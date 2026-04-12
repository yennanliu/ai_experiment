"""Tests for services/ticket_service.py"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import services.ticket_service as svc
from graph.state import TicketState


@pytest.fixture(autouse=True)
def clear_store():
    """Reset the in-memory store before each test."""
    svc._store.clear()
    yield
    svc._store.clear()


# ── create_ticket ─────────────────────────────────────────────────────────────

def test_create_ticket_returns_pending_record():
    record = svc.create_ticket("my app crashed")
    assert record["status"] == "pending"
    assert record["message"] == "my app crashed"
    assert record["ticket_id"].startswith("TKT-")
    assert record["result"] is None
    assert record["error"] is None


def test_create_ticket_persists_to_store():
    record = svc.create_ticket("billing issue")
    assert svc._store[record["ticket_id"]] == record


def test_create_ticket_generates_unique_ids():
    ids = {svc.create_ticket("msg")["ticket_id"] for _ in range(10)}
    assert len(ids) == 10


# ── get_ticket ────────────────────────────────────────────────────────────────

def test_get_ticket_returns_none_for_missing():
    assert svc.get_ticket("TKT-NOPE") is None


def test_get_ticket_returns_existing():
    record = svc.create_ticket("help")
    assert svc.get_ticket(record["ticket_id"]) == record


# ── list_tickets ──────────────────────────────────────────────────────────────

def test_list_tickets_empty():
    assert svc.list_tickets() == []


def test_list_tickets_returns_all():
    svc.create_ticket("one")
    svc.create_ticket("two")
    assert len(svc.list_tickets()) == 2


# ── run_pipeline ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_pipeline_sets_done_on_success():
    record = svc.create_ticket("I was double-charged")
    ticket_id = record["ticket_id"]

    fake_state = TicketState(
        ticket_id=ticket_id,
        user_message="I was double-charged",
        category="billing",
        confidence=0.95,
        priority="high",
        sla_hours=4,
        department="Finance",
        response="We'll refund you within 4 hours.",
        quality_score=0.9,
        retry_count=1,
        history=[],
    )

    with patch("services.ticket_service.process_ticket", return_value=fake_state):
        await svc.run_pipeline(ticket_id, "I was double-charged")

    updated = svc._store[ticket_id]
    assert updated["status"] == "done"
    assert updated["result"]["category"] == "billing"
    assert updated["result"]["priority"] == "high"
    assert updated["result"]["response"] == "We'll refund you within 4 hours."


@pytest.mark.asyncio
async def test_run_pipeline_sets_error_on_exception():
    record = svc.create_ticket("bad ticket")
    ticket_id = record["ticket_id"]

    with patch("services.ticket_service.process_ticket", side_effect=RuntimeError("LLM timeout")):
        await svc.run_pipeline(ticket_id, "bad ticket")

    updated = svc._store[ticket_id]
    assert updated["status"] == "error"
    assert "LLM timeout" in updated["error"]
