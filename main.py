import os
import json
import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from google.cloud import firestore

import vertexai
from vertexai.generative_models import GenerativeModel

from zoneinfo import ZoneInfo


# ---- Config ----
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
REGION = os.environ.get("GCP_REGION", "us-central1")  # Vertex AI region
TIMEZONE = os.environ.get("TIMEZONE", "America/New_York")
DEFAULT_USER_ID = os.environ.get("DEFAULT_USER_ID", "default")

# Firestore multi-database: you said your DB id is productivitydatabase
FIRESTORE_DB = os.environ.get("FIRESTORE_DB", "productivitydatabase")

MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-lite")


app = FastAPI()
templates = Jinja2Templates(directory="templates")
db = firestore.Client(database=FIRESTORE_DB)


# ---- Helpers ----
def now_utc_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def today_local_date() -> datetime.date:
    return datetime.datetime.now(ZoneInfo(TIMEZONE)).date()


def current_week_id() -> str:
    # ISO week, e.g., 2026-W08
    d = today_local_date()
    iso_year, iso_week, _ = d.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def safe_parse_json(text: str) -> Dict[str, Any]:
    """
    Gemini sometimes returns extra text around JSON.
    This tries to extract the first {...} block.
    """
    if not text:
        return {"error": "Empty model response"}

    t = text.strip()

    # Remove ```json fences if present
    if t.startswith("```"):
        t = t.strip("`")
        t = t.replace("json", "", 1).strip()

    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        t = t[start : end + 1]

    try:
        return json.loads(t)
    except Exception:
        return {"error": "Model did not return valid JSON", "raw": text}


def call_gemini_json(prompt: str) -> Dict[str, Any]:
    vertexai.init(project=PROJECT_ID, location=REGION)
    model = GenerativeModel(MODEL_NAME)
    resp = model.generate_content(prompt)
    return safe_parse_json(resp.text)


# ---- Firestore helpers ----
WEEKLY_PLANS_COL = "weekly_plans"
EVENTS_COL = "events_log"


def week_doc_ref(user_id: str, week_id: str):
    return db.collection(WEEKLY_PLANS_COL).document(f"{user_id}__{week_id}")


def get_or_init_week(user_id: str) -> Dict[str, Any]:
    week_id = current_week_id()
    ref = week_doc_ref(user_id, week_id)
    snap = ref.get()

    if snap.exists:
        data = snap.to_dict() or {}
        # Ensure required fields exist
        data.setdefault("user_id", user_id)
        data.setdefault("week_id", week_id)
        data.setdefault("version", 0)
        data.setdefault("tasks", [])
        data.setdefault("weekly_plan", [])
        return data

    # Initialize a new week doc
    data = {
        "user_id": user_id,
        "week_id": week_id,
        "version": 0,
        "tasks": [],
        "weekly_plan": [],
        "created_at": now_utc_iso(),
        "updated_at": now_utc_iso(),
    }
    ref.set(data)
    return data


def save_week(user_id: str, week_doc: Dict[str, Any]) -> None:
    ref = week_doc_ref(user_id, week_doc["week_id"])
    ref.set(week_doc)


def log_event(payload: Dict[str, Any]) -> None:
    payload.setdefault("created_at", now_utc_iso())
    db.collection(EVENTS_COL).add(payload)


# ---- Weekly plan display: add actual dates ----
def weekly_plan_to_by_date(week_id: str, weekly_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert weekly_plan:
      [{"day":"Monday","blocks":[{"start":"09:00","end":"10:00","task":"..."}]}]
    into:
      [{"day":"Monday","date":"YYYY-MM-DD","blocks":[...]}]
    Always includes Monday..Sunday, even if empty.
    """
    if not week_id or "-W" not in week_id:
        return []

    year_str, week_str = week_id.split("-W")
    iso_year = int(year_str)
    iso_week = int(week_str)

    # Monday of ISO week
    week_start = datetime.date.fromisocalendar(iso_year, iso_week, 1)

    day_to_offset = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    normalized = {}
    for item in weekly_plan or []:
        day = item.get("day")
        if day in day_to_offset:
            normalized[day] = item.get("blocks", []) or []

    out = []
    for day, offset in day_to_offset.items():
        d = week_start + datetime.timedelta(days=offset)
        out.append({
            "day": day,
            "date": d.isoformat(),
            "blocks": normalized.get(day, []),
        })

    return out


# ---- Prompt builders ----
def build_extract_prompt(text: str) -> str:
    return f"""
You are a task extraction assistant.

Extract actionable tasks from the user's text. Output ONLY valid JSON with this schema:
{{
  "tasks": [
    {{
      "title": "string",
      "due": "string (optional, e.g., 2026-02-23 or 'Friday' or 'tomorrow')",
      "estimated_minutes": number (optional),
      "priority": "low|medium|high (optional)",
      "category": "string (optional)",
      "notes": "string (optional)"
    }}
  ]
}}

Rules:
- Keep titles short and actionable.
- If user did not provide due date/time, omit "due".
- If you are unsure, omit the field.
- Do not include any extra keys outside this schema.

User text:
{text}
""".strip()


def build_update_week_prompt(existing_weekly_plan: List[Dict[str, Any]],
                             all_tasks: List[Dict[str, Any]],
                             new_tasks: List[Dict[str, Any]]) -> str:
    return f"""
You are a weekly planning assistant.

You will update a weekly schedule (Monday-Sunday) based on tasks. Output ONLY valid JSON with this schema:
{{
  "weekly_plan": [
    {{
      "day": "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday",
      "blocks": [
        {{
          "start": "HH:MM",
          "end": "HH:MM",
          "task": "string",
          "notes": "string (optional)"
        }}
      ]
    }}
  ],
  "changes": ["string"],
  "conflicts": ["string"]
}}

Constraints:
- Keep a realistic schedule (avoid 00:00-06:00).
- Prefer 30-120 min blocks.
- Do NOT delete existing blocks unless necessary; adjust carefully.
- Add new tasks into open times; if cannot fit, put in conflicts.

Existing weekly plan JSON:
{json.dumps(existing_weekly_plan, ensure_ascii=False)}

All tasks (existing + new) JSON:
{json.dumps(all_tasks, ensure_ascii=False)}

New tasks to add (most important):
{json.dumps(new_tasks, ensure_ascii=False)}
""".strip()


# ---- Routes ----
@app.get("/healthz")
def healthz():
    return {"ok": True, "time": now_utc_iso()}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # show current week even on first load
    week_doc = get_or_init_week(DEFAULT_USER_ID)
    week_id = week_doc.get("week_id")
    week_version = week_doc.get("version", 0)
    weekly_by_date = weekly_plan_to_by_date(week_id, week_doc.get("weekly_plan", []))

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "input_text": "",
            "extracted_pretty": None,
            "extracted_tasks": [],
            "pending_tasks_json": None,
            "week_id": week_id,
            "week_version": week_version,
            "weekly_by_date": weekly_by_date,
        }
    )


@app.post("/ui/action", response_class=HTMLResponse)
def ui_action(
    request: Request,
    text: str = Form(""),
    action: str = Form("extract_preview"),
    pending_tasks_json: str = Form("")
):
    input_text = (text or "").strip()

    extracted_pretty: Optional[str] = None
    extracted_tasks: List[Dict[str, Any]] = []
    pending_tasks_json_out: Optional[str] = None

    # always load current week for display
    week_doc = get_or_init_week(DEFAULT_USER_ID)
    week_id = week_doc.get("week_id")
    week_version = week_doc.get("version", 0)
    weekly_by_date = weekly_plan_to_by_date(week_id, week_doc.get("weekly_plan", []))

    try:
        if action == "extract_preview":
            if not input_text:
                extracted_pretty = json.dumps({"error": "Please paste text first."}, indent=2, ensure_ascii=False)
            else:
                extracted = call_gemini_json(build_extract_prompt(input_text))
                extracted_tasks = extracted.get("tasks", []) or []
                pending = {"tasks": extracted_tasks}
                pending_tasks_json_out = json.dumps(pending, ensure_ascii=False)
                extracted_pretty = json.dumps(extracted, indent=2, ensure_ascii=False)

        elif action == "confirm_add":
            if not pending_tasks_json:
                extracted_pretty = json.dumps({"error": "No extracted tasks to add. Please Extract first."}, indent=2, ensure_ascii=False)
            else:
                pending = json.loads(pending_tasks_json)
                new_tasks = pending.get("tasks", [])
                if not isinstance(new_tasks, list) or len(new_tasks) == 0:
                    extracted_pretty = json.dumps({"error": "Extracted task list is empty."}, indent=2, ensure_ascii=False)
                else:
                    existing_plan = week_doc.get("weekly_plan", [])
                    tasks_updated = (week_doc.get("tasks", []) or []) + new_tasks

                    updated = call_gemini_json(build_update_week_prompt(existing_plan, tasks_updated, new_tasks))

                    week_doc["tasks"] = tasks_updated
                    week_doc["weekly_plan"] = updated.get("weekly_plan", [])
                    week_doc["version"] = int(week_doc.get("version", 0)) + 1
                    week_doc["updated_at"] = now_utc_iso()
                    save_week(DEFAULT_USER_ID, week_doc)

                    log_event({
                        "type": "ui_add_to_week",
                        "user_id": DEFAULT_USER_ID,
                        "week_id": week_doc["week_id"],
                        "new_tasks": new_tasks,
                        "changes": updated.get("changes", []),
                        "conflicts": updated.get("conflicts", []),
                    })

                    # refresh plan display after update
                    week_id = week_doc.get("week_id")
                    week_version = week_doc.get("version", 0)
                    weekly_by_date = weekly_plan_to_by_date(week_id, week_doc.get("weekly_plan", []))

                    extracted_pretty = json.dumps({
                        "message": "Tasks added to weekly plan.",
                        "week_id": week_id,
                        "version": week_version,
                        "changes": updated.get("changes", []),
                        "conflicts": updated.get("conflicts", []),
                    }, indent=2, ensure_ascii=False)

        elif action == "view_week":
            week_doc = get_or_init_week(DEFAULT_USER_ID)
            week_id = week_doc.get("week_id")
            week_version = week_doc.get("version", 0)
            weekly_by_date = weekly_plan_to_by_date(week_id, week_doc.get("weekly_plan", []))
            extracted_pretty = json.dumps(week_doc, indent=2, ensure_ascii=False)

        else:
            extracted_pretty = json.dumps({"error": f"Unknown action: {action}"}, indent=2, ensure_ascii=False)

    except Exception as e:
        extracted_pretty = json.dumps({"error": str(e)}, indent=2, ensure_ascii=False)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "input_text": input_text,
            "extracted_pretty": extracted_pretty,
            "extracted_tasks": extracted_tasks,
            "pending_tasks_json": pending_tasks_json_out,  # only set after extract
            "week_id": week_id,
            "week_version": week_version,
            "weekly_by_date": weekly_by_date,
        }
    )


# ---- Optional API endpoints (useful for demo/report) ----
@app.post("/api/extract")
def api_extract(payload: Dict[str, Any]):
    text = (payload.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "Missing text"}, status_code=400)
    extracted = call_gemini_json(build_extract_prompt(text))
    return extracted


@app.get("/api/weekly/get")
def api_weekly_get():
    week_doc = get_or_init_week(DEFAULT_USER_ID)
    return week_doc


@app.post("/api/weekly/add_text")
def api_weekly_add_text(payload: Dict[str, Any]):
    text = (payload.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "Missing text"}, status_code=400)

    extracted = call_gemini_json(build_extract_prompt(text))
    new_tasks = extracted.get("tasks", []) or []

    week_doc = get_or_init_week(DEFAULT_USER_ID)
    existing_plan = week_doc.get("weekly_plan", [])
    tasks_updated = (week_doc.get("tasks", []) or []) + new_tasks

    updated = call_gemini_json(build_update_week_prompt(existing_plan, tasks_updated, new_tasks))

    week_doc["tasks"] = tasks_updated
    week_doc["weekly_plan"] = updated.get("weekly_plan", [])
    week_doc["version"] = int(week_doc.get("version", 0)) + 1
    week_doc["updated_at"] = now_utc_iso()
    save_week(DEFAULT_USER_ID, week_doc)

    log_event({
        "type": "api_add_text",
        "user_id": DEFAULT_USER_ID,
        "week_id": week_doc["week_id"],
        "new_tasks": new_tasks,
        "changes": updated.get("changes", []),
        "conflicts": updated.get("conflicts", []),
    })

    return {
        "message": "Added tasks and updated weekly plan.",
        "week_id": week_doc["week_id"],
        "version": week_doc["version"],
        "changes": updated.get("changes", []),
        "conflicts": updated.get("conflicts", []),
        "weekly_plan": week_doc.get("weekly_plan", []),
    }
