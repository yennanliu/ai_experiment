import asyncio
import io
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agent.config import get_settings
from agent.designer import parse_resume_structure
from agent.graph import resume_agent
from db import delete_record, get_all_records, init_db, insert_record, update_notes, update_status


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


def _build_cl_intent(body: "TailorRequest") -> str:
    """Build a cover letter intent string to inject into the agent state."""
    parts = []
    if body.cover_letter_tone != "professional":
        parts.append(f"Tone: {body.cover_letter_tone}")
    if body.cover_letter_length != "standard":
        length_map = {"brief": "150-200 words", "detailed": "350-450 words"}
        parts.append(f"Length: {length_map.get(body.cover_letter_length, 'standard')}")
    if body.cover_letter_emphasis.strip():
        parts.append(f"Emphasize: {body.cover_letter_emphasis.strip()}")
    return "; ".join(parts)


async def _extract_job_meta(jd: str) -> tuple[str, str]:
    """Extract company name and job title from a JD using a fast LLM call."""
    from agent.nodes import _chat
    from agent.config import get_settings
    try:
        raw = await _chat(
            model=get_settings().openai_model_fast,
            system=(
                "Extract the company name and job title from the job description. "
                'Respond ONLY with JSON: {"company_name": "<name or empty string>", "job_title": "<title or empty string>"}. '
                "If not found, use empty strings."
            ),
            user=jd[:1500],
            json_mode=True,
        )
        import json as _json
        data = _json.loads(raw)
        return data.get("company_name", ""), data.get("job_title", "")
    except Exception:
        return "", ""


def _build_refinement_context(accumulated: dict) -> str:
    """Summarise previous-iteration feedback for the resume rewriter."""
    parts = []
    rec = accumulated.get("recruiter_feedback", {})
    hm  = accumulated.get("hiring_manager_feedback", {})
    ats = accumulated.get("ats_report", {})
    missing = ats.get("missing_keywords", [])[:6]
    if rec.get("feedback"):
        parts.append(f"Recruiter said: {rec['feedback']}")
    if hm.get("rationale"):
        parts.append(f"Hiring manager said: {hm['rationale']}")
    if missing:
        parts.append(f"Still missing keywords: {', '.join(missing)}")
    return " | ".join(parts)


async def _run_iterations(body: "TailorRequest", emit) -> dict:
    """
    Core pipeline: run body.iterations rounds of the agent.
    emit(event_type, data_dict) is called for every SSE event.
    Returns the final result dict (including DB record id).
    """
    _check_token_estimate(body.resume, body.job_description)
    cl_intent = _build_cl_intent(body)
    meta_task = asyncio.create_task(_extract_job_meta(body.job_description))

    current_resume = body.resume
    accumulated: dict = {}

    for iteration in range(1, body.iterations + 1):
        if body.iterations > 1:
            await emit("iteration_start", {"iteration": iteration, "total": body.iterations})

        iter_ctx = _build_refinement_context(accumulated) if iteration > 1 else ""

        initial_state = {
            "raw_resume": current_resume,
            "job_description": body.job_description,
            "materials": body.materials,
            "cover_letter_intent": cl_intent,
            "iteration_context": iter_ctx,
        }
        iter_acc: dict = {**initial_state}

        async for chunk in resume_agent.astream(initial_state):
            for node_name, delta in chunk.items():
                iter_acc = {**iter_acc, **delta}
                accumulated = {**accumulated, **iter_acc}

                payload: dict = {"node": node_name, "iteration": iteration, "total_iterations": body.iterations}
                if node_name == "ats_simulate":
                    ats = delta.get("ats_report", {})
                    payload.update(ats_score=ats.get("score", 0),
                                   missing=len(ats.get("missing_keywords", [])),
                                   ats_report=ats)
                elif node_name == "rewrite_resume":
                    payload["tailored_resume"] = delta.get("tailored_resume", "")
                elif node_name == "write_cover_letter":
                    payload["cover_letter"] = delta.get("cover_letter", "")
                elif node_name == "recruiter_review":
                    payload["recruiter_feedback"] = delta.get("recruiter_feedback", {})
                elif node_name == "hiring_manager_review":
                    payload["hiring_manager_feedback"] = delta.get("hiring_manager_feedback", {})
                elif node_name == "score_output":
                    payload["final_score"] = delta.get("final_score", 0)

                await emit("node", payload)

        score = iter_acc.get("final_score", 0)
        if body.iterations > 1:
            await emit("iteration_done", {"iteration": iteration, "total": body.iterations, "score": score})

        current_resume = iter_acc.get("tailored_resume", current_resume)

    company_name, job_title = await meta_task
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
        company_name=company_name,
        job_title=job_title,
    )
    return {
        "id": record["id"],
        "company_name": company_name,
        "job_title": job_title,
        "tailored_resume": accumulated.get("tailored_resume", ""),
        "cover_letter": accumulated.get("cover_letter", ""),
        "ats_report": accumulated.get("ats_report", {}),
        "recruiter_feedback": accumulated.get("recruiter_feedback", {}),
        "hiring_manager_feedback": accumulated.get("hiring_manager_feedback", {}),
        "final_score": accumulated.get("final_score", 0),
        "iterations_completed": body.iterations,
    }


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
    # Cover letter intent
    cover_letter_tone: str = Field(default="professional")   # professional | conversational | concise
    cover_letter_length: str = Field(default="standard")     # brief | standard | detailed
    cover_letter_emphasis: str = Field(default="", max_length=300)
    # Iterative tailoring
    iterations: int = Field(default=1, ge=1, le=5)


class TailorResponse(BaseModel):
    id: int
    tailored_resume: str
    cover_letter: str
    ats_report: dict
    recruiter_feedback: dict
    hiring_manager_feedback: dict
    final_score: int
    company_name: str = ""
    job_title: str = ""


class StatusUpdate(BaseModel):
    status: Literal["Draft", "Applied", "Interviewing", "Rejected", "Offer"]


class NotesUpdate(BaseModel):
    notes: str = Field(default="", max_length=2000)


class ExportRequest(BaseModel):
    content: str = Field(..., min_length=1)
    title: str = Field(default="Resume")
    # PDF options
    style: str = Field(default="classic")          # classic | modern | minimal
    page_size: str = Field(default="A4")           # A4 | Letter
    margin: str = Field(default="normal")          # narrow | normal | wide
    font_scale: float = Field(default=1.0, ge=0.7, le=1.5)
    accent_color: str = Field(default="#4F46E5")   # hex color
    max_pages: int = Field(default=0, ge=0, le=10)  # 0 = unlimited


# ── Job store ─────────────────────────────────────────────────────────────────

@dataclass
class _Job:
    id: str
    label: str
    iterations: int
    status: str = "queued"   # queued | running | done | failed | cancelled
    current_iter: int = 0
    final_score: int = 0
    error: str = ""
    created_at: str = ""
    result: dict | None = None
    events: list = field(default_factory=list)   # SSE strings buffered for replay
    _done: asyncio.Event = field(default_factory=asyncio.Event)


class JobStore:
    _MAX = 30

    def __init__(self):
        self._jobs: dict[str, _Job] = {}
        self._order: list[str] = []

    def create(self, label: str, iterations: int) -> _Job:
        job_id = uuid.uuid4().hex[:8]
        job = _Job(
            id=job_id, label=label, iterations=iterations,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._jobs[job_id] = job
        self._order.append(job_id)
        if len(self._order) > self._MAX:
            old = self._order.pop(0)
            self._jobs.pop(old, None)
        return job

    def get(self, job_id: str) -> _Job | None:
        return self._jobs.get(job_id)

    def push(self, job: _Job, event_type: str, data: dict) -> None:
        job.events.append(f"event: {event_type}\ndata: {json.dumps(data)}\n\n")

    def finish(self, job: _Job) -> None:
        job._done.set()

    def summary(self) -> list[dict]:
        out = []
        for jid in reversed(self._order):
            j = self._jobs.get(jid)
            if j:
                out.append({
                    "id": j.id, "label": j.label, "status": j.status,
                    "iterations": j.iterations, "current_iter": j.current_iter,
                    "final_score": j.final_score, "error": j.error,
                    "created_at": j.created_at,
                })
        return out


job_store = JobStore()


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
    cl_intent = _build_cl_intent(body)
    initial_state = {
        "raw_resume": body.resume,
        "job_description": body.job_description,
        "materials": body.materials,
        "cover_letter_intent": cl_intent,
    }
    result, (company_name, job_title) = await asyncio.gather(
        resume_agent.ainvoke(initial_state),
        _extract_job_meta(body.job_description),
    )
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
        company_name=company_name,
        job_title=job_title,
    )
    logger.info(f"Tailor complete id={record['id']} score={result['final_score']}")
    return TailorResponse(
        id=record["id"],
        company_name=company_name,
        job_title=job_title,
        **{k: result[k] for k in TailorResponse.model_fields if k not in ("id", "company_name", "job_title")},
    )


@app.post("/tailor-stream", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def tailor_stream(request: Request, body: TailorRequest):
    """SSE stream: runs body.iterations rounds, emitting node + iteration events."""
    async def event_stream():
        q: asyncio.Queue = asyncio.Queue()

        async def emit(event_type: str, data: dict):
            await q.put(f"event: {event_type}\ndata: {json.dumps(data)}\n\n")

        async def run():
            try:
                result = await _run_iterations(body, emit)
                logger.info(f"Stream complete id={result['id']} score={result['final_score']}")
                await q.put(f"event: done\ndata: {json.dumps(result)}\n\n")
            except Exception as exc:
                logger.exception("Stream error")
                await q.put(f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n")
            finally:
                await q.put(None)

        asyncio.create_task(run())
        while True:
            item = await q.get()
            if item is None:
                break
            yield item

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Background job endpoints ──────────────────────────────────────────────────

async def _run_job_background(job: _Job, body: TailorRequest) -> None:
    job.status = "running"

    async def emit(event_type: str, data: dict):
        if event_type == "iteration_start":
            job.current_iter = data["iteration"]
        elif event_type == "node" and data.get("node") == "score_output":
            job.final_score = data.get("final_score", 0)
        job_store.push(job, event_type, data)

    try:
        result = await _run_iterations(body, emit)
        job.result = result
        job.final_score = result.get("final_score", 0)
        job.current_iter = body.iterations
        job.status = "done"
        job_store.push(job, "done", result)
        logger.info(f"Job {job.id} done score={job.final_score}")
    except Exception as exc:
        logger.exception(f"Job {job.id} failed")
        job.error = str(exc)
        job.status = "failed"
        job_store.push(job, "error", {"message": str(exc)})
    finally:
        job_store.finish(job)


@app.post("/tailor-job", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def create_tailor_job(request: Request, body: TailorRequest):
    """Create a background tailor job. Returns immediately with job_id."""
    _check_token_estimate(body.resume, body.job_description)
    label = f"{body.job_description[:60].strip()}…" if len(body.job_description) > 60 else body.job_description
    job = job_store.create(label=label, iterations=body.iterations)
    asyncio.create_task(_run_job_background(job, body))
    return {"job_id": job.id, "status": job.status, "label": job.label}


@app.get("/jobs")
async def list_jobs():
    return job_store.summary()


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    """SSE: replay buffered events then tail live events for a background job."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def gen():
        idx = 0
        while True:
            while idx < len(job.events):
                yield job.events[idx]
                idx += 1
            if job._done.is_set() and idx >= len(job.events):
                break
            try:
                await asyncio.wait_for(asyncio.shield(job._done.wait()), timeout=5)
            except asyncio.TimeoutError:
                yield "event: ping\ndata: {}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/jobs/{job_id}", status_code=204)
async def dismiss_job(job_id: str):
    """Remove a job from the dashboard (cancel if still running)."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in ("queued", "running"):
        job.status = "cancelled"
        job_store.finish(job)
    # Remove from store
    job_store._jobs.pop(job_id, None)
    if job_id in job_store._order:
        job_store._order.remove(job_id)


@app.post("/export-pdf")
async def export_pdf(req: ExportRequest):
    """Designer Agent parses the resume into a structured layout, then renders a styled PDF."""
    # Step 1: call designer agent to parse free-form text into structured JSON
    if req.title.lower() in ("cover_letter", "cover letter"):
        # Cover letters don't need structural parsing — render as plain formatted text
        structure = None
    else:
        structure = await parse_resume_structure(req.content)

    # Step 2: render PDF
    pdf_opts = {
        "style": req.style,
        "page_size": req.page_size,
        "margin": req.margin,
        "font_scale": req.font_scale,
        "accent_color": req.accent_color,
        "max_pages": req.max_pages,
    }
    buf = await asyncio.to_thread(_render_pdf, req.content, req.title, structure, pdf_opts)

    safe_title = "".join(c for c in req.title if c.isalnum() or c in " _-").strip() or "document"
    filename = safe_title.replace(" ", "_") + ".pdf"
    return StreamingResponse(
        buf, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _get_pdf_fonts() -> tuple[str, str, str]:
    """Return (body_font, bold_font, italic_font) — Unicode-capable if a system font is found."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        # (regular, bold, italic) — macOS
        (
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
        ),
        # DejaVu — Linux
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        ),
        (
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Oblique.ttf",
        ),
    ]
    for regular, bold, italic in candidates:
        if os.path.exists(regular):
            try:
                pdfmetrics.registerFont(TTFont("UF", regular))
                pdfmetrics.registerFont(TTFont("UF-Bold", bold if os.path.exists(bold) else regular))
                pdfmetrics.registerFont(TTFont("UF-Italic", italic if os.path.exists(italic) else regular))
                return "UF", "UF-Bold", "UF-Italic"
            except Exception:
                pass
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


def _render_pdf(content: str, title: str, structure: dict | None, opts: dict | None = None) -> io.BytesIO:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4, LETTER
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    opts = opts or {}
    style      = opts.get("style", "classic")          # classic | modern | minimal
    page_size  = opts.get("page_size", "A4")
    margin_key = opts.get("margin", "normal")
    font_scale = float(opts.get("font_scale", 1.0))
    max_pages  = int(opts.get("max_pages", 0))         # 0 = unlimited
    accent_hex = opts.get("accent_color", "#4F46E5")

    # Page size
    pagesize = LETTER if page_size == "Letter" else A4
    PAGE_W, PAGE_H = pagesize

    # Margins
    margin_cm = {"narrow": 1.2, "normal": 2.0, "wide": 2.8}.get(margin_key, 2.0)
    L_MARGIN = R_MARGIN = margin_cm * cm
    T_MARGIN = B_MARGIN = margin_cm * cm
    BODY_W = PAGE_W - L_MARGIN - R_MARGIN

    # Colors — vary by style
    ACCENT_CLR = colors.HexColor(accent_hex)
    DARK       = colors.HexColor("#111827")
    MEDIUM     = colors.HexColor("#374151")
    MUTED      = colors.HexColor("#6B7280")
    RULE_CLR   = colors.HexColor("#D1D5DB")

    if style == "minimal":
        heading_color = DARK
        section_color = MEDIUM
        rule_color    = RULE_CLR
    elif style == "modern":
        heading_color = ACCENT_CLR
        section_color = ACCENT_CLR
        rule_color    = ACCENT_CLR
    else:  # classic (default)
        heading_color = ACCENT_CLR
        section_color = ACCENT_CLR
        rule_color    = RULE_CLR

    body_f, bold_f, ital_f = _get_pdf_fonts()

    def fs(base: float) -> float:
        return round(base * font_scale, 1)

    def S(name, **kw) -> ParagraphStyle:
        return ParagraphStyle(name, **kw)

    s_name    = S("name",    fontName=bold_f, fontSize=fs(22), leading=fs(26), textColor=DARK,         spaceAfter=2)
    s_job_ttl = S("jt",      fontName=body_f, fontSize=fs(11), leading=fs(14), textColor=heading_color, spaceAfter=4)
    s_contact = S("contact", fontName=body_f, fontSize=fs(9),  leading=fs(12), textColor=MUTED,         spaceAfter=0)
    s_section = S("sec",     fontName=bold_f, fontSize=fs(9),  leading=fs(12), textColor=section_color,
                  spaceBefore=14, spaceAfter=3, textTransform="uppercase", letterSpacing=0.8)
    s_body    = S("body",    fontName=body_f, fontSize=fs(10), leading=fs(14), textColor=MEDIUM, spaceAfter=4)
    s_bullet  = S("bullet",  fontName=body_f, fontSize=fs(10), leading=fs(14), textColor=MEDIUM,
                  leftIndent=12, firstLineIndent=0, spaceAfter=3)
    s_exp_ttl = S("etitle",  fontName=bold_f, fontSize=fs(10), leading=fs(13), textColor=DARK,   spaceAfter=1)
    s_exp_co  = S("eco",     fontName=ital_f, fontSize=fs(9),  leading=fs(12), textColor=MUTED,  spaceAfter=3)
    s_dates   = S("dates",   fontName=body_f, fontSize=fs(9),  leading=fs(12), textColor=MUTED,  alignment=TA_RIGHT)
    s_cover   = S("cover",   fontName=body_f, fontSize=fs(10.5), leading=fs(16), textColor=MEDIUM, spaceAfter=8)

    buf   = io.BytesIO()
    story = []

    header_rule_thickness = 1.2 if style != "minimal" else 0.5
    header_rule_color = heading_color if style in ("classic", "modern") else RULE_CLR

    def hr(thickness=0.5, color=None, space_before=6, space_after=6):
        return HRFlowable(width="100%", thickness=thickness, color=color or rule_color,
                          spaceBefore=space_before, spaceAfter=space_after)

    def section_header(text: str):
        story.append(hr(thickness=0.5, space_before=10, space_after=0))
        story.append(Paragraph(text.upper(), s_section))

    def _count_pages(flowables) -> int:
        """Build into a scratch buffer and return page count."""
        scratch = io.BytesIO()
        tmp = SimpleDocTemplate(scratch, pagesize=pagesize,
                                leftMargin=L_MARGIN, rightMargin=R_MARGIN,
                                topMargin=T_MARGIN,  bottomMargin=B_MARGIN)
        tmp.build(flowables)
        return tmp.page

    def _build_doc(flowables, out: io.BytesIO):
        d = SimpleDocTemplate(out, pagesize=pagesize,
                              leftMargin=L_MARGIN, rightMargin=R_MARGIN,
                              topMargin=T_MARGIN,  bottomMargin=B_MARGIN)
        d.build(flowables)

    # ── Cover letter (no structure) ──────────────────────────────────────────
    if structure is None:
        for line in content.split("\n"):
            stripped = line.strip()
            if not stripped:
                story.append(Spacer(1, 8))
            else:
                story.append(Paragraph(stripped, s_cover))
        _build_doc(story, buf)
        buf.seek(0)
        return buf

    # ── Structured resume ────────────────────────────────────────────────────

    # Header — name
    if structure["name"]:
        story.append(Paragraph(structure["name"], s_name))

    # Professional title
    if structure["title"]:
        story.append(Paragraph(structure["title"], s_job_ttl))

    # Contact line  email · phone · linkedin · website · location
    c = structure["contact"]
    contact_parts = [v for v in [c["email"], c["phone"], c["linkedin"], c["website"], c["location"]] if v]
    if contact_parts:
        story.append(Paragraph("  ·  ".join(contact_parts), s_contact))

    story.append(hr(thickness=header_rule_thickness, color=header_rule_color, space_before=8, space_after=4))

    # Summary
    if structure["summary"]:
        section_header("Summary")
        story.append(Paragraph(structure["summary"], s_body))

    # Skills
    if structure["skills"]:
        section_header("Skills")
        for skill_group in structure["skills"]:
            cat   = skill_group.get("category", "")
            items = skill_group.get("items") or []
            line  = (f"<b>{cat}:</b>  " if cat else "") + ",  ".join(items)
            story.append(Paragraph(line, s_body))

    # Experience
    if structure["experience"]:
        section_header("Experience")
        for exp in structure["experience"]:
            # Two-column row: title/company on left, location | dates on right
            right_text = "  |  ".join(filter(None, [exp["location"], exp["dates"]]))
            title_para = Paragraph(exp["title"], s_exp_ttl)
            dates_para = Paragraph(right_text, s_dates)
            tbl = Table(
                [[title_para, dates_para]],
                colWidths=[BODY_W * 0.65, BODY_W * 0.35],
            )
            tbl.setStyle(TableStyle([
                ("VALIGN",  (0, 0), (-1, -1), "BOTTOM"),
                ("LEFTPADDING",  (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING",   (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
            ]))
            story.append(tbl)

            if exp["company"]:
                story.append(Paragraph(exp["company"], s_exp_co))

            for bullet in exp["bullets"]:
                story.append(Paragraph(f"•  {bullet}", s_bullet))

            story.append(Spacer(1, 6))

    # Education
    if structure["education"]:
        section_header("Education")
        for edu in structure["education"]:
            right_text = "  |  ".join(filter(None, [edu["location"], edu["dates"]]))
            deg_para   = Paragraph(edu["degree"], s_exp_ttl)
            date_para  = Paragraph(right_text, s_dates)
            tbl = Table(
                [[deg_para, date_para]],
                colWidths=[BODY_W * 0.65, BODY_W * 0.35],
            )
            tbl.setStyle(TableStyle([
                ("VALIGN",  (0, 0), (-1, -1), "BOTTOM"),
                ("LEFTPADDING",  (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING",   (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
            ]))
            story.append(tbl)
            if edu["institution"]:
                story.append(Paragraph(edu["institution"], s_exp_co))
            story.append(Spacer(1, 4))

    # Extra sections (certifications, projects, etc.)
    for extra in structure.get("extra_sections") or []:
        section_header(extra.get("title", ""))
        for item in (extra.get("items") or []):
            story.append(Paragraph(f"•  {item}", s_bullet))

    # ── Page-count enforcement (shrink font until content fits) ──────────────
    if max_pages > 0:
        current_scale = font_scale
        while current_scale >= 0.6:
            pages = _count_pages(story)
            if pages <= max_pages:
                break
            # Reduce by 3% and rebuild styles
            current_scale = round(current_scale - 0.03, 4)

            def _fs(base: float) -> float:
                return round(base * current_scale, 1)

            s_name.fontSize    = _fs(22); s_name.leading    = _fs(26)
            s_job_ttl.fontSize = _fs(11); s_job_ttl.leading = _fs(14)
            s_contact.fontSize = _fs(9);  s_contact.leading = _fs(12)
            s_section.fontSize = _fs(9);  s_section.leading = _fs(12)
            s_body.fontSize    = _fs(10); s_body.leading    = _fs(14)
            s_bullet.fontSize  = _fs(10); s_bullet.leading  = _fs(14)
            s_exp_ttl.fontSize = _fs(10); s_exp_ttl.leading = _fs(13)
            s_exp_co.fontSize  = _fs(9);  s_exp_co.leading  = _fs(12)
            s_dates.fontSize   = _fs(9);  s_dates.leading   = _fs(12)
            s_cover.fontSize   = _fs(10.5); s_cover.leading = _fs(16)

    _build_doc(story, buf)
    buf.seek(0)
    return buf


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


@app.patch("/history/{record_id}/notes")
async def update_record_notes(record_id: int, body: NotesUpdate):
    record = await asyncio.to_thread(update_notes, record_id, body.notes)
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
