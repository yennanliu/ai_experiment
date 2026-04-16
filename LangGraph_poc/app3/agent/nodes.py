from __future__ import annotations

import json

from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import get_settings
from .prompts import ATS_SIMULATOR, COVER_LETTER_WRITER, HIRING_MANAGER, RECRUITER, RESUME_WRITER
from .state import ResumeState

# Module-level async client — created once, reused across all requests
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=get_settings().openai_api_key)
    return _client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _chat(*, model: str, system: str, user: str, json_mode: bool = False, temperature: float = 0) -> str:
    kwargs: dict = dict(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = await _get_client().chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


async def parse_inputs(state: ResumeState) -> dict:
    """Validate inputs and initialize empty output fields."""
    # Return ONLY changed/new fields — never re-emit unchanged fields.
    # LangGraph merges partial returns automatically; returning the full
    # state via {**state, ...} causes InvalidUpdateError on parallel branches
    # because multiple nodes try to write the same channel in one superstep.
    return {
        "materials": state.get("materials") or "",
        "cover_letter_intent": state.get("cover_letter_intent") or "",
        "tailored_resume": "",
        "cover_letter": "",
        "ats_report": {},
        "recruiter_feedback": {},
        "hiring_manager_feedback": {},
        "final_score": 0,
    }


async def ats_simulate(state: ResumeState) -> ResumeState:
    """ATS Simulator — score keyword match between resume and JD."""
    cfg = get_settings()
    raw = await _chat(
        model=cfg.openai_model_fast,
        system=ATS_SIMULATOR,
        user=f"Job Description:\n{state['job_description']}\n\nResume:\n{state['raw_resume']}",
        json_mode=True,
    )
    data = json.loads(raw)
    ats_report = {
        "score": int(data.get("score", 0)),
        "missing_keywords": data.get("missing_keywords") or [],
        "suggestions": data.get("suggestions") or [],
    }
    return {"ats_report": ats_report}


async def rewrite_resume(state: ResumeState) -> ResumeState:
    """Resume Writer — tailor resume to JD using ATS report."""
    cfg = get_settings()
    ats = state["ats_report"]
    missing = ", ".join(ats.get("missing_keywords", [])) or "None"
    suggestions = ", ".join(ats.get("suggestions", [])) or "None"

    tailored = await _chat(
        model=cfg.openai_model_writer,
        system=RESUME_WRITER,
        user=(
            f"Job Description:\n{state['job_description']}\n\n"
            f"ATS Missing Keywords: {missing}\n"
            f"ATS Suggestions: {suggestions}\n\n"
            f"Original Resume:\n{state['raw_resume']}"
        ),
        temperature=0.3,
    )
    return {"tailored_resume": tailored}


async def write_cover_letter(state: ResumeState) -> ResumeState:
    """Cover Letter Writer — generate a tailored cover letter."""
    cfg = get_settings()
    extras = []
    if state.get("materials", "").strip():
        extras.append(f"Additional Materials:\n{state['materials']}")
    if state.get("cover_letter_intent", "").strip():
        extras.append(f"Writer Instructions: {state['cover_letter_intent']}")
    extras_block = ("\n\n" + "\n\n".join(extras)) if extras else ""

    cover_letter = await _chat(
        model=cfg.openai_model_writer,
        system=COVER_LETTER_WRITER,
        user=(
            f"Job Description:\n{state['job_description']}\n\n"
            f"Tailored Resume:\n{state['tailored_resume']}"
            f"{extras_block}"
        ),
        temperature=0.4,
    )
    return {"cover_letter": cover_letter}


async def recruiter_review(state: ResumeState) -> ResumeState:
    """Recruiter agent — 30-second scan feedback."""
    cfg = get_settings()
    raw = await _chat(
        model=cfg.openai_model_fast,
        system=RECRUITER,
        user=(
            f"Role applied for (from JD):\n{state['job_description'][:500]}\n\n"
            f"Resume:\n{state['tailored_resume']}"
        ),
        json_mode=True,
    )
    data = json.loads(raw)
    feedback = {
        "verdict": data.get("verdict", ""),
        "feedback": data.get("feedback", ""),
    }
    return {"recruiter_feedback": feedback}


async def hiring_manager_review(state: ResumeState) -> ResumeState:
    """Hiring Manager agent — technical fit evaluation."""
    cfg = get_settings()
    raw = await _chat(
        model=cfg.openai_model_fast,
        system=HIRING_MANAGER,
        user=(
            f"Job Description:\n{state['job_description']}\n\n"
            f"Resume:\n{state['tailored_resume']}"
        ),
        json_mode=True,
    )
    data = json.loads(raw)
    feedback = {
        "score": int(data.get("score", 0)),
        "verdict": data.get("verdict", ""),
        "rationale": data.get("rationale", ""),
    }
    return {"hiring_manager_feedback": feedback}


async def score_output(state: ResumeState) -> ResumeState:
    """Aggregate a final score from ATS + hiring manager scores."""
    ats_score = state["ats_report"].get("score", 0)
    hm_score = state["hiring_manager_feedback"].get("score", 0)
    final_score = round(ats_score * 0.4 + hm_score * 0.6)
    return {"final_score": final_score}
