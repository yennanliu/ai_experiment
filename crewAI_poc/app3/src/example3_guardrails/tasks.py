from typing import Any, Tuple
from crewai import Agent, Task, TaskOutput


# ── Guardrail functions ──────────────────────────────────────────────────────

def validate_analysis(result: TaskOutput) -> Tuple[bool, Any]:
    """Reject analysis that is too brief or lacks evidence-based language."""
    word_count = len(result.raw.split())
    if word_count < 80:
        return (
            False,
            f"Analysis too brief ({word_count} words). "
            "Expand to at least 80 words with specific findings and data.",
        )
    analytical_terms = ["because", "therefore", "however", "data", "shows",
                        "indicates", "suggests", "evidence", "analysis"]
    if not any(term in result.raw.lower() for term in analytical_terms):
        return (
            False,
            "Analysis lacks evidence-based reasoning. "
            "Include terms like 'data shows', 'therefore', or 'however'.",
        )
    return (True, result.raw.strip())


def validate_report(result: TaskOutput) -> Tuple[bool, Any]:
    """Reject reports missing required markdown sections or below minimum length."""
    raw_lower = result.raw.lower()
    for section in ["## introduction", "## conclusion"]:
        if section not in raw_lower:
            return (
                False,
                f"Report missing '{section}'. Required sections: "
                "## Introduction, at least one ## body section, ## Conclusion.",
            )
    word_count = len(result.raw.split())
    if word_count < 150:
        return (False, f"Report too short ({word_count} words). Need at least 150 words.")
    return (True, result.raw.strip())


# ── Task factories ────────────────────────────────────────────────────────────

def analysis_task(agent: Agent, topic: str) -> Task:
    return Task(
        description=(
            f"Analyze: **{topic}**\n"
            "Identify 3–4 key trends or findings. Back each one with specific evidence "
            "and analytical reasoning. Avoid vague summaries."
        ),
        expected_output="Analysis of 3–4 findings with evidence-based reasoning (80+ words).",
        agent=agent,
        guardrail=validate_analysis,
    )


def report_task(agent: Agent, analysis: Task, topic: str) -> Task:
    return Task(
        description=(
            f"Using the analysis on '{topic}', write a structured markdown report.\n"
            "Required sections: # Title, ## Introduction, ## Key Findings, "
            "## Implications, ## Conclusion."
        ),
        expected_output=(
            "Structured markdown report (150+ words) with all required sections present."
        ),
        agent=agent,
        context=[analysis],
        guardrail=validate_report,
    )
