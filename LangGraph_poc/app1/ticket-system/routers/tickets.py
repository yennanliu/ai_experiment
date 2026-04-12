"""Tickets router — HTTP controller layer"""

import asyncio
from collections import Counter

from fastapi import APIRouter, HTTPException

from models.ticket import SubmitRequest, TicketRecord
from services.ticket_service import create_ticket, get_ticket, list_tickets, run_pipeline

router = APIRouter(prefix="/tickets", tags=["tickets"])


@router.post("", response_model=TicketRecord, status_code=202)
async def submit_ticket(body: SubmitRequest):
    record = create_ticket(body.message)
    asyncio.create_task(run_pipeline(record["ticket_id"], body.message))
    return record


@router.get("", response_model=list[TicketRecord])
async def get_all_tickets():
    return list_tickets()


@router.get("/stats", tags=["tickets"])
async def get_stats():
    tickets = list_tickets()
    by_status   = Counter(t["status"] for t in tickets)
    by_category = Counter(t["result"]["category"] for t in tickets if t.get("result"))
    by_priority = Counter(t["result"]["priority"] for t in tickets if t.get("result"))
    return {
        "total": len(tickets),
        "by_status":   dict(by_status),
        "by_category": dict(by_category),
        "by_priority": dict(by_priority),
    }


@router.get("/{ticket_id}", response_model=TicketRecord)
async def get_one_ticket(ticket_id: str):
    record = get_ticket(ticket_id)
    if not record:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return record
