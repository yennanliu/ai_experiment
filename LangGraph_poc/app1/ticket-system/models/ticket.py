"""Pydantic schemas for the HTTP API layer"""

from typing import Literal

from pydantic import BaseModel


class SubmitRequest(BaseModel):
    message: str


class TicketResult(BaseModel):
    category: str
    confidence: float
    priority: str
    sla_hours: int
    department: str
    research_notes: str = ""
    response: str
    quality_score: float
    retry_count: int
    history: list[dict]


class TicketRecord(BaseModel):
    ticket_id: str
    status: Literal["pending", "processing", "done", "error"]
    submitted_at: str
    message: str
    result: TicketResult | None = None
    error: str | None = None
