from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from agent.graph import email_agent

app = FastAPI(title="AI Email Reply Assistant")


class EmailRequest(BaseModel):
    email: str


class DraftResponse(BaseModel):
    email_type: str
    draft_reply: str
    checklist: list[str]


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


@app.get("/health")
async def health():
    return {"status": "ok"}
