from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from agent.graph import email_agent
from agent.templates import TEMPLATES
from mock_data import MOCK_EMAILS

app = FastAPI(title="AI Email Reply Assistant")
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory history store — list of processed emails (newest first)
history: list[dict] = []


# ── Models ────────────────────────────────────────────────────────────────────

class EmailRequest(BaseModel):
    email: str


class DraftResponse(BaseModel):
    id: int
    email_type: str
    draft_reply: str
    checklist: list[str]
    confidence_score: int
    confidence_reason: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/history-page")
async def history_page():
    return FileResponse("static/history.html")


@app.post("/generate-draft", response_model=DraftResponse)
async def generate_draft(request: EmailRequest):
    result = email_agent.invoke({
        "inbound_email": request.email,
        "email_type": "",
        "template": "",
        "required_fields": [],
        "draft_reply": "",
        "checklist": [],
        "confidence_score": 0,
        "confidence_reason": "",
    })

    record = {
        "id": len(history) + 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inbound_email": request.email,
        "email_type": result["email_type"],
        "draft_reply": result["draft_reply"],
        "checklist": result["checklist"],
        "confidence_score": result["confidence_score"],
        "confidence_reason": result["confidence_reason"],
        "status": "Drafted",
    }
    history.insert(0, record)

    return DraftResponse(
        id=record["id"],
        email_type=result["email_type"],
        draft_reply=result["draft_reply"],
        checklist=result["checklist"],
        confidence_score=result["confidence_score"],
        confidence_reason=result["confidence_reason"],
    )


@app.get("/history")
async def get_history():
    return history


@app.patch("/history/{record_id}/status")
async def update_status(record_id: int, body: dict):
    for record in history:
        if record["id"] == record_id:
            record["status"] = body.get("status", record["status"])
            return record
    return {"error": "not found"}


@app.get("/templates-page")
async def templates_page():
    return FileResponse("static/templates.html")


@app.get("/templates")
async def get_templates():
    return [
        {
            "email_type": k,
            "required_fields": v["required_fields"],
            "template": v["template"],
        }
        for k, v in TEMPLATES.items()
    ]


@app.get("/mock-emails")
async def mock_emails():
    return MOCK_EMAILS


@app.get("/health")
async def health():
    return {"status": "ok"}
