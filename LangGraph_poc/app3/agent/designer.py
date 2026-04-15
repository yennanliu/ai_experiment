"""Resume Designer Agent — parses free-form resume text into structured JSON for PDF rendering."""
from __future__ import annotations

import json

from .config import get_settings
from .nodes import _chat
from .prompts import RESUME_DESIGNER


async def parse_resume_structure(resume_text: str) -> dict:
    """Call the designer LLM and return a validated structure dict."""
    cfg = get_settings()
    raw = await _chat(
        model=cfg.openai_model_writer,
        system=RESUME_DESIGNER,
        user=resume_text,
        json_mode=True,
        temperature=0,
    )
    data = json.loads(raw)

    # Normalise/default every field so the renderer never KeyErrors
    return {
        "name": data.get("name", ""),
        "title": data.get("title", ""),
        "contact": {
            "email":    data.get("contact", {}).get("email") or "",
            "phone":    data.get("contact", {}).get("phone") or "",
            "linkedin": data.get("contact", {}).get("linkedin") or "",
            "website":  data.get("contact", {}).get("website") or "",
            "location": data.get("contact", {}).get("location") or "",
        },
        "summary": data.get("summary") or "",
        "skills": data.get("skills") or [],
        "experience": [
            {
                "title":   e.get("title", ""),
                "company": e.get("company", ""),
                "location": e.get("location") or "",
                "dates":   e.get("dates", ""),
                "bullets": e.get("bullets") or [],
            }
            for e in (data.get("experience") or [])
        ],
        "education": [
            {
                "degree":      e.get("degree", ""),
                "institution": e.get("institution", ""),
                "location":    e.get("location") or "",
                "dates":       e.get("dates", ""),
            }
            for e in (data.get("education") or [])
        ],
        "extra_sections": data.get("extra_sections") or [],
    }
