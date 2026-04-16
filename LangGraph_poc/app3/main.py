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
from agent.designer import parse_resume_structure
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
    # PDF options
    style: str = Field(default="classic")          # classic | modern | minimal
    page_size: str = Field(default="A4")           # A4 | Letter
    margin: str = Field(default="normal")          # narrow | normal | wide
    font_scale: float = Field(default=1.0, ge=0.7, le=1.5)
    accent_color: str = Field(default="#4F46E5")   # hex color
    max_pages: int = Field(default=0, ge=0, le=10)  # 0 = unlimited


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
            # Seed accumulated with the inputs so insert_record always has them,
            # even though nodes now return only their changed fields (partial dicts).
            accumulated: dict = {**initial_state}
            async for chunk in resume_agent.astream(initial_state):
                # A chunk may contain multiple keys when parallel nodes complete
                # in the same superstep — iterate all of them.
                for node_name, delta in chunk.items():
                    accumulated = {**accumulated, **delta}

                    payload: dict = {"node": node_name}

                    if node_name == "ats_simulate":
                        ats = delta.get("ats_report", {})
                        payload["ats_score"] = ats.get("score", 0)
                        payload["missing"] = len(ats.get("missing_keywords", []))
                        payload["ats_report"] = ats
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


@app.delete("/history/{record_id}", status_code=204)
async def delete_history_record(record_id: int):
    ok = await asyncio.to_thread(delete_record, record_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Record not found")


@app.get("/health")
async def health():
    return {"status": "ok"}
