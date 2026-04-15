import asyncio
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agent.config import get_settings
from agent.graph import resume_agent
from db import delete_record, get_all_records, init_db, insert_record, update_status


# ── Structured JSON logging ───────────────────────────────────────────────────

class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        obj: dict = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(obj)


_handler = logging.StreamHandler()
_handler.setFormatter(_JSONFormatter())
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


# ── Rate limiter ──────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()   # validates all env vars at startup; raises if missing
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    init_db()
    logger.info("App started")
    yield
    logger.info("App shutting down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="AI Resume Tailor", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Auth dependency ───────────────────────────────────────────────────────────

def verify_api_key(x_api_key: str = Header(default="")) -> None:
    key = get_settings().app_api_key
    if key and x_api_key != key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


# ── Token estimate helper ─────────────────────────────────────────────────────

_MAX_RESUME_TOKENS = 6000
_MAX_JD_TOKENS = 3000


def _check_token_estimate(resume: str, jd: str) -> None:
    errors = []
    if len(resume) // 4 > _MAX_RESUME_TOKENS:
        errors.append(f"Resume too long (~{len(resume)//4} tokens, max {_MAX_RESUME_TOKENS})")
    if len(jd) // 4 > _MAX_JD_TOKENS:
        errors.append(f"Job description too long (~{len(jd)//4} tokens, max {_MAX_JD_TOKENS})")
    if errors:
        raise HTTPException(status_code=422, detail="; ".join(errors))


# ── Models ────────────────────────────────────────────────────────────────────

class TailorRequest(BaseModel):
    resume: str = Field(..., min_length=100, max_length=24000)
    job_description: str = Field(..., min_length=50, max_length=12000)
    materials: str = Field(default="", max_length=4000)


class TailorResponse(BaseModel):
    id: int
    tailored_resume: str
    cover_letter: str
    ats_report: dict
    recruiter_feedback: dict
    hiring_manager_feedback: dict
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


@app.post("/tailor", response_model=TailorResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def tailor(request: Request, body: TailorRequest):
    _check_token_estimate(body.resume, body.job_description)
    initial_state = {
        "raw_resume": body.resume,
        "job_description": body.job_description,
        "materials": body.materials,
    }
    result = await resume_agent.ainvoke(initial_state)
    record = await asyncio.to_thread(
        insert_record,
        raw_resume=body.resume,
        job_description=body.job_description,
        tailored_resume=result["tailored_resume"],
        cover_letter=result["cover_letter"],
        ats_report=result["ats_report"],
        recruiter_feedback=result["recruiter_feedback"],
        hiring_manager_feedback=result["hiring_manager_feedback"],
        final_score=result["final_score"],
    )
    logger.info(f"Tailor complete id={record['id']} score={result['final_score']}")
    return TailorResponse(id=record["id"], **{k: result[k] for k in TailorResponse.model_fields if k != "id"})


@app.post("/tailor-stream", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def tailor_stream(request: Request, body: TailorRequest):
    """SSE: one node event per LangGraph node, then a final done event."""
    _check_token_estimate(body.resume, body.job_description)
    initial_state = {
        "raw_resume": body.resume,
        "job_description": body.job_description,
        "materials": body.materials,
    }

    async def event_stream():
        try:
            accumulated: dict = {}
            async for chunk in resume_agent.astream(initial_state):
                node_name = next(iter(chunk))
                state = chunk[node_name]
                accumulated = {**accumulated, **state}

                payload: dict = {"node": node_name}

                if node_name == "ats_simulate":
                    ats = state.get("ats_report", {})
                    payload["ats_score"] = ats.get("score", 0)
                    payload["missing"] = len(ats.get("missing_keywords", []))
                    payload["ats_report"] = ats
                elif node_name == "rewrite_resume":
                    payload["tailored_resume"] = state.get("tailored_resume", "")
                elif node_name == "write_cover_letter":
                    payload["cover_letter"] = state.get("cover_letter", "")
                elif node_name == "recruiter_review":
                    payload["recruiter_feedback"] = state.get("recruiter_feedback", {})
                elif node_name == "hiring_manager_review":
                    payload["hiring_manager_feedback"] = state.get("hiring_manager_feedback", {})
                elif node_name == "score_output":
                    payload["final_score"] = state.get("final_score", 0)

                yield f"event: node\ndata: {json.dumps(payload)}\n\n"

            if accumulated:
                record = await asyncio.to_thread(
                    insert_record,
                    raw_resume=body.resume,
                    job_description=body.job_description,
                    tailored_resume=accumulated.get("tailored_resume", ""),
                    cover_letter=accumulated.get("cover_letter", ""),
                    ats_report=accumulated.get("ats_report", {}),
                    recruiter_feedback=accumulated.get("recruiter_feedback", {}),
                    hiring_manager_feedback=accumulated.get("hiring_manager_feedback", {}),
                    final_score=accumulated.get("final_score", 0),
                )
                result = {
                    "id": record["id"],
                    "tailored_resume": accumulated.get("tailored_resume", ""),
                    "cover_letter": accumulated.get("cover_letter", ""),
                    "ats_report": accumulated.get("ats_report", {}),
                    "recruiter_feedback": accumulated.get("recruiter_feedback", {}),
                    "hiring_manager_feedback": accumulated.get("hiring_manager_feedback", {}),
                    "final_score": accumulated.get("final_score", 0),
                }
                logger.info(f"Stream complete id={record['id']}")
                yield f"event: done\ndata: {json.dumps(result)}\n\n"

        except Exception as exc:
            logger.exception("Stream error")
            yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/export-pdf")
async def export_pdf(req: ExportRequest):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

    _FONT_CANDIDATES = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    body_font = "Helvetica"
    bold_font = "Helvetica-Bold"
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("UnicodeFont", path))
                body_font = bold_font = "UnicodeFont"
                break
            except Exception:
                pass

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2.5 * cm, rightMargin=2.5 * cm,
        topMargin=2.5 * cm, bottomMargin=2.5 * cm,
    )
    style_normal = ParagraphStyle("body", fontName=body_font, fontSize=10, leading=15,
                                  textColor=colors.HexColor("#1d1d1f"), spaceAfter=6)
    style_heading = ParagraphStyle("heading", fontName=bold_font, fontSize=13, leading=18,
                                   textColor=colors.HexColor("#111827"), spaceBefore=14, spaceAfter=4)
    style_section = ParagraphStyle("section", fontName=bold_font, fontSize=10, leading=14,
                                   textColor=colors.HexColor("#4F46E5"), spaceBefore=12, spaceAfter=2)

    story = []
    for line in req.content.split("\n"):
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 6))
        elif stripped.isupper() and len(stripped) < 40:
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=colors.HexColor("#e0e0e0"), spaceAfter=4))
            story.append(Paragraph(stripped, style_section))
        elif stripped.startswith("# "):
            story.append(Paragraph(stripped[2:], style_heading))
        elif stripped.startswith("## "):
            story.append(Paragraph(stripped[3:], style_section))
        elif stripped.startswith(("- ", "• ")):
            story.append(Paragraph("• " + stripped[2:], style_normal))
        else:
            story.append(Paragraph(stripped, style_normal))

    doc.build(story)
    buf.seek(0)

    safe_title = "".join(c for c in req.title if c.isalnum() or c in " _-").strip() or "document"
    filename = safe_title.replace(" ", "_") + ".pdf"
    return StreamingResponse(
        buf, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/history")
async def get_history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    return await asyncio.to_thread(get_all_records, limit, offset)


@app.patch("/history/{record_id}/status")
async def update_record_status(record_id: int, body: StatusUpdate):
    record = await asyncio.to_thread(update_status, record_id, body.status)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return record


@app.delete("/history/{record_id}", status_code=204)
async def delete_history_record(record_id: int):
    ok = await asyncio.to_thread(delete_record, record_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Record not found")


@app.get("/health")
async def health():
    return {"status": "ok"}
