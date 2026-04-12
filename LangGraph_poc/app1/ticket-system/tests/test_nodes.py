"""Tests for graph/nodes.py — all LLM calls are mocked"""

from unittest.mock import patch

import pytest

from graph.nodes import (
    SLA_MAP,
    classify,
    generate_response,
    prioritize,
    quality_check,
    route,
    should_retry,
)
from graph.state import TicketState


# ── classify ──────────────────────────────────────────────────────────────────

def test_classify_sets_category_and_confidence():
    state = TicketState(user_message="my payment failed")
    with patch("graph.nodes.chat", return_value='{"category": "billing", "confidence": 0.95}'):
        result = classify(state)
    assert result.category == "billing"
    assert result.confidence == 0.95


def test_classify_appends_history():
    state = TicketState(user_message="crash")
    with patch("graph.nodes.chat", return_value='{"category": "technical", "confidence": 0.9}'):
        result = classify(state)
    assert len(result.history) == 1
    assert result.history[0]["node"] == "classify"


# ── prioritize ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("priority,expected_sla", list(SLA_MAP.items()))
def test_prioritize_sets_sla(priority, expected_sla):
    state = TicketState(category="billing", user_message="help")
    with patch("graph.nodes.chat", return_value=f'{{"priority": "{priority}"}}'):
        result = prioritize(state)
    assert result.priority == priority
    assert result.sla_hours == expected_sla


def test_prioritize_unknown_priority_defaults_to_24h():
    state = TicketState(category="other", user_message="hi")
    with patch("graph.nodes.chat", return_value='{"priority": "unknown"}'):
        result = prioritize(state)
    assert result.sla_hours == 24


# ── route ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("category,department", [
    ("technical", "Engineering"),
    ("billing", "Finance"),
    ("account", "Customer Success"),
    ("feature_request", "Product"),
    ("bug_report", "Engineering"),
    ("other", "General Support"),
])
def test_route_maps_correctly(category, department):
    state = TicketState(category=category)
    result = route(state)
    assert result.department == department


def test_route_unknown_category_falls_back():
    state = TicketState(category="mystery")
    result = route(state)
    assert result.department == "General Support"


def test_route_appends_history():
    state = TicketState(category="billing")
    result = route(state)
    assert any(h["node"] == "route" for h in result.history)


# ── generate_response ─────────────────────────────────────────────────────────

def test_generate_response_sets_response():
    state = TicketState(
        user_message="I need help",
        category="technical",
        priority="high",
        department="Engineering",
        sla_hours=4,
    )
    with patch("graph.nodes.chat", return_value="We're on it, expect a fix within 4 hours."):
        result = generate_response(state)
    assert result.response == "We're on it, expect a fix within 4 hours."
    assert any(h["node"] == "generate_response" for h in result.history)


# ── quality_check ─────────────────────────────────────────────────────────────

def test_quality_check_sets_score_and_increments_retry():
    state = TicketState(user_message="help", response="Here is your answer.")
    with patch("graph.nodes.chat", return_value='{"score": 0.92}'):
        result = quality_check(state)
    assert result.quality_score == 0.92
    assert result.retry_count == 1


def test_quality_check_increments_on_each_call():
    state = TicketState(user_message="help", response="ok", retry_count=1)
    with patch("graph.nodes.chat", return_value='{"score": 0.5}'):
        result = quality_check(state)
    assert result.retry_count == 2


# ── should_retry ──────────────────────────────────────────────────────────────

def test_should_retry_when_score_low_and_retries_remaining():
    state = TicketState(quality_score=0.5, retry_count=1)
    assert should_retry(state) == "generate_response"


def test_should_not_retry_when_score_meets_threshold():
    state = TicketState(quality_score=0.85, retry_count=1)
    assert should_retry(state) == "__end__"


def test_should_not_retry_when_max_retries_reached():
    state = TicketState(quality_score=0.3, retry_count=3)
    assert should_retry(state) == "__end__"


def test_should_retry_at_exactly_threshold_boundary():
    # score == 0.8 should NOT retry (condition is strict <)
    state = TicketState(quality_score=0.8, retry_count=1)
    assert should_retry(state) == "__end__"
