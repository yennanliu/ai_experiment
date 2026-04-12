"""Tests for graph/state.py"""

from graph.state import TicketState


def test_default_values():
    state = TicketState()
    assert state.ticket_id == ""
    assert state.user_message == ""
    assert state.category == ""
    assert state.confidence == 0.0
    assert state.priority == ""
    assert state.sla_hours == 24
    assert state.department == ""
    assert state.response == ""
    assert state.quality_score == 0.0
    assert state.retry_count == 0
    assert state.history == []


def test_history_is_independent_per_instance():
    s1 = TicketState()
    s2 = TicketState()
    s1.history.append("event")
    assert s2.history == []


def test_explicit_init():
    state = TicketState(ticket_id="TKT-001", user_message="help", priority="high")
    assert state.ticket_id == "TKT-001"
    assert state.user_message == "help"
    assert state.priority == "high"
