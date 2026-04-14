from __future__ import annotations
from functools import lru_cache
from openai import OpenAI
from .state import ResumeState
from .prompts import ATS_SIMULATOR, RESUME_WRITER, COVER_LETTER_WRITER, RECRUITER, HIRING_MANAGER


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return OpenAI()


def parse_inputs(state: ResumeState) -> ResumeState:
    """Validate inputs and initialize empty output fields."""
    return {
        **state,
        "materials": state.get("materials") or "",
        "tailored_resume": "",
        "cover_letter": "",
        "ats_report": {},
        "recruiter_feedback": "",
        "hiring_manager_feedback": "",
        "final_score": 0,
    }


def ats_simulate(state: ResumeState) -> ResumeState:
    """ATS Simulator — score keyword match between resume and JD."""
    response = get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": ATS_SIMULATOR},
            {
                "role": "user",
                "content": (
                    f"Job Description:\n{state['job_description']}\n\n"
                    f"Resume:\n{state['raw_resume']}"
                ),
            },
        ],
        temperature=0,
    )
    text = response.choices[0].message.content.strip()
    ats_report = {"score": 0, "missing_keywords": [], "suggestions": []}
    for line in text.splitlines():
        if line.startswith("SCORE:"):
            try:
                ats_report["score"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("MISSING:"):
            raw = line.split(":", 1)[1].strip()
            ats_report["missing_keywords"] = [] if raw == "None" else [k.strip() for k in raw.split(",")]
        elif line.startswith("SUGGESTIONS:"):
            raw = line.split(":", 1)[1].strip()
            ats_report["suggestions"] = [] if raw == "None" else [k.strip() for k in raw.split(",")]
    return {**state, "ats_report": ats_report}


def rewrite_resume(state: ResumeState) -> ResumeState:
    """Resume Writer — tailor resume to JD using ATS report."""
    ats = state["ats_report"]
    missing = ", ".join(ats.get("missing_keywords", [])) or "None"
    suggestions = ", ".join(ats.get("suggestions", [])) or "None"

    response = get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": RESUME_WRITER},
            {
                "role": "user",
                "content": (
                    f"Job Description:\n{state['job_description']}\n\n"
                    f"ATS Missing Keywords: {missing}\n"
                    f"ATS Suggestions: {suggestions}\n\n"
                    f"Original Resume:\n{state['raw_resume']}"
                ),
            },
        ],
        temperature=0.3,
    )
    tailored = response.choices[0].message.content.strip()
    return {**state, "tailored_resume": tailored}


def write_cover_letter(state: ResumeState) -> ResumeState:
    """Resume Writer — generate a tailored cover letter, optionally using extra materials."""
    materials_block = ""
    if state.get("materials", "").strip():
        materials_block = f"\n\nAdditional Materials (bio, past cover letters, achievements):\n{state['materials']}"

    response = get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": COVER_LETTER_WRITER},
            {
                "role": "user",
                "content": (
                    f"Job Description:\n{state['job_description']}\n\n"
                    f"Tailored Resume:\n{state['tailored_resume']}"
                    f"{materials_block}"
                ),
            },
        ],
        temperature=0.4,
    )
    cover_letter = response.choices[0].message.content.strip()
    return {**state, "cover_letter": cover_letter}


def recruiter_review(state: ResumeState) -> ResumeState:
    """Recruiter agent — 30-second scan feedback."""
    response = get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": RECRUITER},
            {
                "role": "user",
                "content": (
                    f"Role applied for (from JD):\n{state['job_description'][:500]}\n\n"
                    f"Resume:\n{state['tailored_resume']}"
                ),
            },
        ],
        temperature=0,
    )
    return {**state, "recruiter_feedback": response.choices[0].message.content.strip()}


def hiring_manager_review(state: ResumeState) -> ResumeState:
    """Hiring Manager agent — technical fit evaluation."""
    response = get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": HIRING_MANAGER},
            {
                "role": "user",
                "content": (
                    f"Job Description:\n{state['job_description']}\n\n"
                    f"Resume:\n{state['tailored_resume']}"
                ),
            },
        ],
        temperature=0,
    )
    return {**state, "hiring_manager_feedback": response.choices[0].message.content.strip()}


def score_output(state: ResumeState) -> ResumeState:
    """Aggregate a final score from ATS + hiring manager scores."""
    ats_score = state["ats_report"].get("score", 0)

    hm_score = 0
    for line in state["hiring_manager_feedback"].splitlines():
        if line.startswith("SCORE:"):
            try:
                hm_score = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass

    final_score = round(ats_score * 0.4 + hm_score * 0.6)
    return {**state, "final_score": final_score}
