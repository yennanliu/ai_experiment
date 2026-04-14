from typing import TypedDict


class ResumeState(TypedDict):
    raw_resume: str               # Original resume text (markdown or plain)
    job_description: str          # Full JD text
    materials: str                # Optional extra context: bio, past cover letters, achievements
    tailored_resume: str          # Rewritten resume from Resume Writer
    cover_letter: str             # Generated cover letter
    ats_report: dict              # {score: int, missing_keywords: list, suggestions: list}
    recruiter_feedback: str       # Tone/format feedback
    hiring_manager_feedback: str  # Fit score + rationale
    final_score: int              # 0–100 overall confidence
