from dotenv import load_dotenv
load_dotenv()

import json
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.graph import email_agent
from agent.templates import TEMPLATES
from agent.retriever import count as rag_count
from mock_data import MOCK_EMAILS
from db import init_db, insert_record, get_all_records, get_record, update_status

app = FastAPI(title="AI Email Reply Assistant")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env before starting the server."
        )
    init_db()


# ── Models ────────────────────────────────────────────────────────────────────

class EmailRequest(BaseModel):
    email: str = Field(..., min_length=1, max_length=4000)


class DraftResponse(BaseModel):
    id: int
    email_type: str
    draft_reply: str
    checklist: list[str]
    retrieved_count: int
    confidence_score: int
    confidence_reason: str


class StatusUpdate(BaseModel):
    status: Literal["Drafted", "Reviewed", "Sent", "Rejected"]


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
        "retrieved_examples": [],
        "confidence_score": 0,
        "confidence_reason": "",
    })

    record = insert_record(
        inbound_email=request.email,
        email_type=result["email_type"],
        draft_reply=result["draft_reply"],
        checklist=result["checklist"],
        confidence_score=result["confidence_score"],
        confidence_reason=result["confidence_reason"],
    )

    return DraftResponse(
        id=record["id"],
        email_type=result["email_type"],
        draft_reply=result["draft_reply"],
        checklist=result["checklist"],
        retrieved_count=len(result.get("retrieved_examples") or []),
        confidence_score=result["confidence_score"],
        confidence_reason=result["confidence_reason"],
    )


@app.post("/generate-draft-stream")
async def generate_draft_stream(request: EmailRequest):
    """SSE endpoint — streams a per-node event after each LangGraph node completes,
    then a final 'done' event with the full result. The UI uses this to drive the
    real-time pipeline visualization."""
    initial_state = {
        "inbound_email": request.email,
        "email_type": "",
        "template": "",
        "required_fields": [],
        "draft_reply": "",
        "checklist": [],
        "retrieved_examples": [],
        "confidence_score": 0,
        "confidence_reason": "",
    }

    def event_stream():
        try:
            final_state = None
            for chunk in email_agent.stream(initial_state):
                node_name = next(iter(chunk))
                state = chunk[node_name]
                final_state = state

                payload: dict = {"node": node_name}
                if node_name == "classify":
                    payload["email_type"] = state.get("email_type", "")
                elif node_name == "retrieve_examples":
                    payload["count"] = len(state.get("retrieved_examples") or [])
                elif node_name == "build_checklist":
                    payload["flags"] = len(state.get("checklist") or [])
                elif node_name == "score_draft":
                    payload["score"] = state.get("confidence_score", 0)

                yield f"event: node\ndata: {json.dumps(payload)}\n\n"

            if final_state:
                record = insert_record(
                    inbound_email=request.email,
                    email_type=final_state["email_type"],
                    draft_reply=final_state["draft_reply"],
                    checklist=final_state["checklist"],
                    confidence_score=final_state["confidence_score"],
                    confidence_reason=final_state["confidence_reason"],
                )
                result = {
                    "id": record["id"],
                    "email_type": final_state["email_type"],
                    "draft_reply": final_state["draft_reply"],
                    "checklist": final_state["checklist"],
                    "retrieved_count": len(final_state.get("retrieved_examples") or []),
                    "confidence_score": final_state["confidence_score"],
                    "confidence_reason": final_state["confidence_reason"],
                }
                yield f"event: done\ndata: {json.dumps(result)}\n\n"

        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/history")
async def get_history():
    return get_all_records()


@app.patch("/history/{record_id}/status")
async def update_record_status(record_id: int, body: StatusUpdate):
    record = update_status(record_id, body.status)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return record


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


@app.get("/rag-status")
async def rag_status():
    return {"documents": rag_count()}


@app.get("/health")
async def health():
    return {"status": "ok"}
