from dotenv import load_dotenv
load_dotenv()

import json
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.graph import resume_agent
from db import init_db, insert_record, get_all_records, get_record, update_status

app = FastAPI(title="AI Resume Tailor")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env before starting.")
    init_db()


# ── Models ────────────────────────────────────────────────────────────────────

class TailorRequest(BaseModel):
    resume: str = Field(..., min_length=1, max_length=8000)
    job_description: str = Field(..., min_length=1, max_length=4000)


class TailorResponse(BaseModel):
    id: int
    tailored_resume: str
    cover_letter: str
    ats_report: dict
    recruiter_feedback: str
    hiring_manager_feedback: str
    final_score: int


class StatusUpdate(BaseModel):
    status: Literal["Draft", "Applied", "Interviewing", "Rejected", "Offer"]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/history-page")
async def history_page():
    return FileResponse("static/history.html")


@app.post("/tailor", response_model=TailorResponse)
async def tailor(request: TailorRequest):
    result = resume_agent.invoke({
        "raw_resume": request.resume,
        "job_description": request.job_description,
    })
    record = insert_record(
        raw_resume=request.resume,
        job_description=request.job_description,
        tailored_resume=result["tailored_resume"],
        cover_letter=result["cover_letter"],
        ats_report=result["ats_report"],
        recruiter_feedback=result["recruiter_feedback"],
        hiring_manager_feedback=result["hiring_manager_feedback"],
        final_score=result["final_score"],
    )
    return TailorResponse(id=record["id"], **{k: result[k] for k in TailorResponse.model_fields if k != "id"})


@app.post("/tailor-stream")
async def tailor_stream(request: TailorRequest):
    """SSE — emits one 'node' event per LangGraph node, then a final 'done' event."""
    initial_state = {
        "raw_resume": request.resume,
        "job_description": request.job_description,
    }

    def event_stream():
        try:
            final_state = None
            for chunk in resume_agent.stream(initial_state):
                node_name = next(iter(chunk))
                state = chunk[node_name]
                final_state = state

                payload: dict = {"node": node_name}
                if node_name == "ats_simulate":
                    payload["ats_score"] = state.get("ats_report", {}).get("score", 0)
                    payload["missing"] = len(state.get("ats_report", {}).get("missing_keywords", []))
                elif node_name == "score_output":
                    payload["final_score"] = state.get("final_score", 0)

                yield f"event: node\ndata: {json.dumps(payload)}\n\n"

            if final_state:
                record = insert_record(
                    raw_resume=request.resume,
                    job_description=request.job_description,
                    tailored_resume=final_state["tailored_resume"],
                    cover_letter=final_state["cover_letter"],
                    ats_report=final_state["ats_report"],
                    recruiter_feedback=final_state["recruiter_feedback"],
                    hiring_manager_feedback=final_state["hiring_manager_feedback"],
                    final_score=final_state["final_score"],
                )
                result = {
                    "id": record["id"],
                    "tailored_resume": final_state["tailored_resume"],
                    "cover_letter": final_state["cover_letter"],
                    "ats_report": final_state["ats_report"],
                    "recruiter_feedback": final_state["recruiter_feedback"],
                    "hiring_manager_feedback": final_state["hiring_manager_feedback"],
                    "final_score": final_state["final_score"],
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


@app.get("/health")
async def health():
    return {"status": "ok"}
