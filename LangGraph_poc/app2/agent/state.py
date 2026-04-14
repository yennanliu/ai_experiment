from typing import TypedDict


class AgentState(TypedDict):
    inbound_email: str
    email_type: str
    template: str
    required_fields: list[str]
    retrieved_examples: list[dict]  # [{email, reply, email_type}, ...]
    draft_reply: str
    checklist: list[str]
    confidence_score: int           # 0–100, LLM self-assessment of reply quality
    confidence_reason: str          # brief explanation of the score
