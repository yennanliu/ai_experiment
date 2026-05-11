from typing import List
from pydantic import BaseModel, Field


class MarketReport(BaseModel):
    topic: str = Field(description="The analyzed market topic")
    key_facts: List[str] = Field(description="3–5 key facts about the market")
    opportunities: List[str] = Field(description="2–3 market opportunities")
    risks: List[str] = Field(description="2–3 key risks")
    recommendation: str = Field(description="Concise strategic recommendation (1–2 sentences)")
    confidence_score: float = Field(
        description="Analyst confidence in the report, from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
