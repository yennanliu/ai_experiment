from crewai import Task, Agent
from src.example2_structured_output.models import MarketReport


def research_task(analyst: Agent, market: str) -> Task:
    return Task(
        description=(
            f"Research the '{market}' market thoroughly. Identify current trends, "
            "market size signals, key players, opportunities, and risks."
        ),
        expected_output=(
            "Research brief: current trends, market size signals, major players, "
            "2–3 opportunities, and 2–3 risks."
        ),
        agent=analyst,
    )


def strategy_task(strategist: Agent, research_task: Task, market: str) -> Task:
    return Task(
        description=(
            f"Using the market research on '{market}', produce a structured market report. "
            "Include: topic, key_facts (3–5 bullet points), opportunities (2–3), "
            "risks (2–3), a strategic recommendation, and a confidence_score (0.0–1.0)."
        ),
        expected_output=(
            "A complete MarketReport object with all fields populated: "
            "topic, key_facts, opportunities, risks, recommendation, confidence_score."
        ),
        agent=strategist,
        context=[research_task],
        output_pydantic=MarketReport,
    )
