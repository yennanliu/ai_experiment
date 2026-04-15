"""Unit tests for agent node logic (no LLM calls)."""
import pytest

from agent.nodes import score_output
from agent.state import ResumeState


def _state(**overrides) -> ResumeState:
    base: ResumeState = {
        "raw_resume": "resume text",
        "job_description": "jd text",
        "materials": "",
        "tailored_resume": "tailored",
        "cover_letter": "cover",
        "ats_report": {"score": 0, "missing_keywords": [], "suggestions": []},
        "recruiter_feedback": {"verdict": "Pass", "feedback": "Looks good."},
        "hiring_manager_feedback": {"score": 0, "verdict": "Yes", "rationale": "Solid."},
        "final_score": 0,
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_score_output_calculation():
    state = _state(
        ats_report={"score": 80, "missing_keywords": [], "suggestions": []},
        hiring_manager_feedback={"score": 90, "verdict": "Strong Yes", "rationale": "Perfect fit."},
    )
    result = await score_output(state)
    # 80 * 0.4 + 90 * 0.6 = 32 + 54 = 86
    assert result["final_score"] == 86


@pytest.mark.asyncio
async def test_score_output_zeros():
    state = _state(
        ats_report={"score": 0, "missing_keywords": [], "suggestions": []},
        hiring_manager_feedback={"score": 0, "verdict": "No", "rationale": "Not a fit."},
    )
    result = await score_output(state)
    assert result["final_score"] == 0


@pytest.mark.asyncio
async def test_score_output_preserves_state():
    state = _state(
        ats_report={"score": 50, "missing_keywords": ["Python"], "suggestions": []},
        hiring_manager_feedback={"score": 60, "verdict": "Maybe", "rationale": "Close."},
    )
    result = await score_output(state)
    assert result["tailored_resume"] == "tailored"
    assert result["cover_letter"] == "cover"
    assert result["ats_report"]["missing_keywords"] == ["Python"]


@pytest.mark.asyncio
async def test_score_output_missing_hm_score():
    """Missing hiring manager score defaults to 0."""
    state = _state(
        ats_report={"score": 100, "missing_keywords": [], "suggestions": []},
        hiring_manager_feedback={},  # no score key
    )
    result = await score_output(state)
    # 100 * 0.4 + 0 * 0.6 = 40
    assert result["final_score"] == 40
