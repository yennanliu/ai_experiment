from dotenv import load_dotenv
load_dotenv()

import io
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
    materials: str = Field(default="", max_length=4000)   # optional extra context


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


class ExportRequest(BaseModel):
    content: str = Field(..., min_length=1)
    title: str = Field(default="Resume")


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
        "materials": request.materials,
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
        "materials": request.materials,
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


@app.post("/export-pdf")
async def export_pdf(request: ExportRequest):
    """PDF Agent — converts resume/cover letter text to a clean downloadable PDF."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.enums import TA_LEFT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2.5 * cm, rightMargin=2.5 * cm,
        topMargin=2.5 * cm, bottomMargin=2.5 * cm,
    )

    styles = getSampleStyleSheet()
    style_normal = ParagraphStyle(
        "body", fontName="Helvetica", fontSize=10, leading=15,
        textColor=colors.HexColor("#1d1d1f"), spaceAfter=6,
    )
    style_heading = ParagraphStyle(
        "heading", fontName="Helvetica-Bold", fontSize=13, leading=18,
        textColor=colors.HexColor("#111827"), spaceBefore=14, spaceAfter=4,
    )
    style_section = ParagraphStyle(
        "section", fontName="Helvetica-Bold", fontSize=10, leading=14,
        textColor=colors.HexColor("#4F46E5"), spaceBefore=12, spaceAfter=2,
    )

    story = []
    lines = request.content.split("\n")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 6))
            continue

        # Heuristic: all-caps short lines → section header
        if stripped.isupper() and len(stripped) < 40:
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e0e0e0"), spaceAfter=4))
            story.append(Paragraph(stripped, style_section))
        # Lines starting with # → heading
        elif stripped.startswith("# "):
            story.append(Paragraph(stripped[2:], style_heading))
        elif stripped.startswith("## "):
            story.append(Paragraph(stripped[3:], style_section))
        # Bullet lines
        elif stripped.startswith("- ") or stripped.startswith("• "):
            bullet_text = "• " + stripped[2:]
            story.append(Paragraph(bullet_text, style_normal))
        else:
            story.append(Paragraph(stripped, style_normal))

    doc.build(story)
    buf.seek(0)

    safe_title = "".join(c for c in request.title if c.isalnum() or c in " _-").strip() or "document"
    filename = safe_title.replace(" ", "_") + ".pdf"

    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
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
