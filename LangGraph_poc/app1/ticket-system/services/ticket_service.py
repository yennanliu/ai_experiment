"""Ticket service — owns the in-memory store and async pipeline execution"""

import asyncio
import uuid
from datetime import datetime

from graph import process_ticket
from models.ticket import TicketRecord, TicketResult

# In-memory store: ticket_id → TicketRecord (as dict for easy mutation)
_store: dict[str, dict] = {}


def create_ticket(message: str) -> dict:
    ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"
    record = TicketRecord(
        ticket_id=ticket_id,
        status="pending",
        submitted_at=datetime.now().isoformat(),
        message=message,
    ).model_dump()
    _store[ticket_id] = record
    return record


def get_ticket(ticket_id: str) -> dict | None:
    return _store.get(ticket_id)


def list_tickets() -> list[dict]:
    return list(_store.values())


def reprocess_ticket(ticket_id: str) -> dict | None:
    """Reset a ticket to pending and schedule a fresh pipeline run."""
    record = _store.get(ticket_id)
    if not record:
        return None
    record["status"] = "pending"
    record["result"] = None
    record["error"] = None
    return record


async def run_pipeline(ticket_id: str, message: str) -> None:
    _store[ticket_id]["status"] = "processing"
    try:
        state = await asyncio.to_thread(process_ticket, ticket_id, message)
        _store[ticket_id]["status"] = "done"
        _store[ticket_id]["result"] = TicketResult(
            category=state.category,
            confidence=round(state.confidence, 2),
            priority=state.priority,
            sla_hours=state.sla_hours,
            department=state.department,
            research_notes=state.research_notes,
            response=state.response,
            quality_score=round(state.quality_score, 2),
            retry_count=state.retry_count,
            history=state.history,
        ).model_dump()
    except Exception as exc:
        _store[ticket_id]["status"] = "error"
        _store[ticket_id]["error"] = str(exc)
