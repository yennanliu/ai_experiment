from typing import TypedDict


class ResumeState(TypedDict):
    raw_resume: str               # Original resume text (markdown or plain)
    job_description: str          # Full JD text
    materials: str                # Optional extra context: bio, past cover letters, achievements
    cover_letter_intent: str      # Optional tone/length/emphasis directives
    tailored_resume: str          # Rewritten resume from Resume Writer
    cover_letter: str             # Generated cover letter
    ats_report: dict              # {score: int, missing_keywords: list, suggestions: list}
    recruiter_feedback: dict      # {verdict: str, feedback: str}
    hiring_manager_feedback: dict # {score: int, verdict: str, rationale: str}
    final_score: int              # 0–100 overall confidence
