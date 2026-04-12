"""Tests for models/ticket.py — Pydantic schema validation"""

import pytest
from pydantic import ValidationError

from models.ticket import SubmitRequest, TicketRecord, TicketResult


def test_submit_request_valid():
    req = SubmitRequest(message="I need help")
    assert req.message == "I need help"


def test_submit_request_requires_message():
    with pytest.raises(ValidationError):
        SubmitRequest()


def test_ticket_record_minimal():
    record = TicketRecord(
        ticket_id="TKT-001",
        status="pending",
        submitted_at="2024-01-01T00:00:00",
        message="help",
    )
    assert record.result is None
    assert record.error is None


def test_ticket_record_invalid_status():
    with pytest.raises(ValidationError):
        TicketRecord(
            ticket_id="TKT-001",
            status="unknown",
            submitted_at="2024-01-01T00:00:00",
            message="help",
        )


def test_ticket_result_valid():
    result = TicketResult(
        category="billing",
        confidence=0.9,
        priority="high",
        sla_hours=4,
        department="Finance",
        response="We'll look into it.",
        quality_score=0.88,
        retry_count=1,
        history=[{"node": "classify", "time": "10:00:00", "detail": "category=billing"}],
    )
    assert result.category == "billing"
    assert result.sla_hours == 4
