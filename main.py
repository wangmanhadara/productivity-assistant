import os, json, datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from google.cloud import firestore
import vertexai
from vertexai.generative_models import GenerativeModel

# ---- Config ----
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
REGION = os.environ.get("GCP_REGION", "us-central1")
TIMEZONE = "America/New_York"
DEFAULT_USER_ID = "default"

app = FastAPI()
templates = Jinja2Templates(directory="templates")
db = firestore.Client(database="productivitydatabase")

# ---- Helpers ----
def now_utc_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def current_week_id() -> str:
    # ISO week, e.g., 2026-W08
    today = datetime.date.today()
    iso_year, iso_week, _ = today.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"

def safe_parse_json(text: str) -> dict:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    return json.loads(text)

def call_gemini_json(prompt: str) -> dict:
    vertexai.init(project=PROJECT_ID, location=REGION)
    model = GenerativeModel("gemini-2.0-flash-lite")
    resp = model.generate_content(prompt)
    return safe_parse_json(resp.text)

def build_extract_prompt(user_text: str) -> str:
    return f"""
You are a task extraction assistant.
Extract actionable tasks from the user's text.

Return STRICT JSON only (no markdown), schema:
{{
  "tasks": [
    {{
      "title": "...",
      "due_date": "YYYY-MM-DD or null",
      "estimated_minutes": number or null,
      "priority": "low|medium|high",
      "notes": "..."
    }}
  ],
  "questions": ["..."]
}}

Rules:
- Only include actionable tasks (things the user needs to do).
- If no due date is mentioned, use null (do NOT invent dates).
- If estimated time is not stated, set null.
- Priority: high if deadline soon or clearly important, else medium/low.
- Include 0-3 clarifying questions if needed.
- If the user text is already a single task, output it as one task.

User text:
{user_text}
""".strip()

def build_update_week_prompt(existing_plan: List[dict], all_tasks: List[dict], new_tasks: List[dict]) -> str:
    # existing_plan: weekly_plan array
    # all_tasks: cumulative tasks stored for the week
    # new_tasks: tasks just added this time
    return f"""
You are a weekly scheduling assistant that updates an existing plan.

Return STRICT JSON only (no markdown), schema:
{{
  "weekly_plan": [
    {{
      "day": "Monday",
      "blocks": [
        {{"start":"09:00","end":"10:00","task":"...","notes":"..."}}
      ]
    }}
  ],
  "changes": ["..."],
  "conflicts": ["..."]
}}

Rules:
- Minimize changes to existing blocks unless necessary.
- Insert new tasks into open slots when possible.
- If a task has a deadline, schedule it before the due date.
- If due_date is null, schedule it in reasonable open time.
- Use realistic blocks (30–120 minutes).
- If conflicts exist (overlapping/too many tasks), list them in conflicts and suggest what to move.

Existing weekly_plan (may be empty):
{json.dumps(existing_plan, ensure_ascii=False)}

All tasks for this week:
{json.dumps(all_tasks, ensure_ascii=False)}

New tasks being added now:
{json.dumps(new_tasks, ensure_ascii=False)}

Timezone: {TIMEZONE}
""".strip()

# ---- Firestore operations ----
def week_doc_ref(user_id: str, week_id: str):
    doc_id = f"{user_id}_{week_id}"
    return db.collection("weekly_plans").document(doc_id)

def get_or_init_week(user_id: str) -> Dict[str, Any]:
    week_id = current_week_id()
    ref = week_doc_ref(user_id, week_id)
    snap = ref.get()
    if snap.exists:
        return snap.to_dict()
    # init empty
    doc = {
        "user_id": user_id,
        "week_id": week_id,
        "timezone": TIMEZONE,
        "tasks": [],
        "weekly_plan": [],
        "version": 0,
        "updated_at": now_utc_iso(),
        "created_at": now_utc_iso(),
    }
    ref.set(doc)
    return doc

def save_week(user_id: str, week_doc: Dict[str, Any]):
    week_id = week_doc["week_id"]
    ref = week_doc_ref(user_id, week_id)
    ref.set(week_doc)

def log_event(event: Dict[str, Any]):
    db.collection("events_log").add(event)

# ---- API endpoints ----
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/api/extract")
def api_extract(payload: Dict[str, Any]):
    user_text = (payload.get("text") or "").strip()
    if not user_text:
        return JSONResponse({"error": "text is required"}, status_code=400)

    result = call_gemini_json(build_extract_prompt(user_text))
    log_event({
        "type": "extract",
        "user_id": payload.get("user_id", DEFAULT_USER_ID),
        "week_id": current_week_id(),
        "user_input": user_text,
        "ai_output": result,
        "created_at": now_utc_iso(),
    })
    return result

@app.get("/api/weekly/get")
def api_weekly_get(user_id: str = DEFAULT_USER_ID):
    week_doc = get_or_init_week(user_id)
    return week_doc

@app.post("/api/weekly/add_text")
def api_weekly_add_text(payload: Dict[str, Any]):
    user_id = payload.get("user_id", DEFAULT_USER_ID)
    text = (payload.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)

    # 1) extract tasks from text
    extracted = call_gemini_json(build_extract_prompt(text))
    new_tasks = extracted.get("tasks", [])
    if not isinstance(new_tasks, list) or len(new_tasks) == 0:
        return JSONResponse({"error": "no tasks extracted"}, status_code=400)

    # 2) load existing week
    week_doc = get_or_init_week(user_id)
    existing_plan = week_doc.get("weekly_plan", [])
    tasks = week_doc.get("tasks", [])

    # 3) append new tasks (simple append; could dedupe, but keep simple for 3-day project)
    tasks_updated = tasks + new_tasks

    # 4) ask Gemini to update weekly plan incrementally
    updated = call_gemini_json(build_update_week_prompt(existing_plan, tasks_updated, new_tasks))

    # 5) persist
    week_doc["tasks"] = tasks_updated
    week_doc["weekly_plan"] = updated.get("weekly_plan", [])
    week_doc["version"] = int(week_doc.get("version", 0)) + 1
    week_doc["updated_at"] = now_utc_iso()

    save_week(user_id, week_doc)

    # 6) log event
    log_event({
        "type": "add_text",
        "user_id": user_id,
        "week_id": week_doc["week_id"],
        "new_tasks": new_tasks,
        "changes": updated.get("changes", []),
        "conflicts": updated.get("conflicts", []),
        "created_at": now_utc_iso(),
    })

    return {
        "week_id": week_doc["week_id"],
        "version": week_doc["version"],
        "new_tasks": new_tasks,
        "changes": updated.get("changes", []),
        "conflicts": updated.get("conflicts", []),
        "weekly_plan": week_doc["weekly_plan"],
    }

@app.get("/healthz")
def healthz():
    return {"ok": True}

# ---- Simple UI handler (posts back to same page) ----
@app.post("/ui/action", response_class=HTMLResponse)
def ui_action(request: Request, text: str = Form(""), action: str = Form("extract")):
    text = (text or "").strip()

    try:
        if action == "extract":
            if not text:
                result = {"error": "Please paste text first."}
            else:
                result = call_gemini_json(build_extract_prompt(text))

        elif action == "add_to_week":
            if not text:
                result = {"error": "Please paste text first."}
            else:
                # reuse the API logic inline for UI
                extracted = call_gemini_json(build_extract_prompt(text))
                new_tasks = extracted.get("tasks", [])

                week_doc = get_or_init_week(DEFAULT_USER_ID)
                existing_plan = week_doc.get("weekly_plan", [])
                tasks_updated = week_doc.get("tasks", []) + new_tasks

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
                    "created_at": now_utc_iso(),
                })

                result = {
                    "week_id": week_doc["week_id"],
                    "version": week_doc["version"],
                    "new_tasks": new_tasks,
                    "changes": updated.get("changes", []),
                    "conflicts": updated.get("conflicts", []),
                    "weekly_plan": week_doc["weekly_plan"],
                }

        elif action == "view_week":
            week_doc = get_or_init_week(DEFAULT_USER_ID)
            result = week_doc

        else:
            result = {"error": f"Unknown action: {action}"}

    except Exception as e:
        result = {"error": str(e)}

    pretty = json.dumps(result, indent=2, ensure_ascii=False)
    return templates.TemplateResponse("index.html", {"request": request, "result": pretty})
