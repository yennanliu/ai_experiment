from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from agent.graph import email_agent
from mock_data import MOCK_EMAILS

app = FastAPI(title="AI Email Reply Assistant")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Models ────────────────────────────────────────────────────────────────────

class EmailRequest(BaseModel):
    email: str


class DraftResponse(BaseModel):
    email_type: str
    draft_reply: str
    checklist: list[str]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/generate-draft", response_model=DraftResponse)
async def generate_draft(request: EmailRequest):
    result = email_agent.invoke({
        "inbound_email": request.email,
        "email_type": "",
        "template": "",
        "required_fields": [],
        "draft_reply": "",
        "checklist": [],
    })
    return DraftResponse(
        email_type=result["email_type"],
        draft_reply=result["draft_reply"],
        checklist=result["checklist"],
    )


@app.get("/mock-emails")
async def mock_emails():
    return MOCK_EMAILS


@app.get("/health")
async def health():
    return {"status": "ok"}
