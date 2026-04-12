"""Integration tests for the FastAPI router — no real LLM calls"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import services.ticket_service as svc
from server import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_store():
    svc._store.clear()
    yield
    svc._store.clear()


# ── POST /tickets ─────────────────────────────────────────────────────────────

def test_submit_ticket_returns_202():
    with patch("routers.tickets.run_pipeline"):  # don't actually run the pipeline
        resp = client.post("/tickets", json={"message": "my screen is black"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "pending"
    assert body["ticket_id"].startswith("TKT-")
    assert body["message"] == "my screen is black"


def test_submit_ticket_missing_message_returns_422():
    resp = client.post("/tickets", json={})
    assert resp.status_code == 422


# ── GET /tickets ──────────────────────────────────────────────────────────────

def test_list_tickets_empty():
    resp = client.get("/tickets")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_tickets_returns_submitted():
    with patch("routers.tickets.run_pipeline"):
        client.post("/tickets", json={"message": "ticket one"})
        client.post("/tickets", json={"message": "ticket two"})
    resp = client.get("/tickets")
    assert len(resp.json()) == 2


# ── GET /tickets/{id} ─────────────────────────────────────────────────────────

def test_get_ticket_not_found():
    resp = client.get("/tickets/TKT-NOPE")
    assert resp.status_code == 404


def test_get_ticket_returns_correct_record():
    with patch("routers.tickets.run_pipeline"):
        post_resp = client.post("/tickets", json={"message": "help me"})
    ticket_id = post_resp.json()["ticket_id"]

    resp = client.get(f"/tickets/{ticket_id}")
    assert resp.status_code == 200
    assert resp.json()["ticket_id"] == ticket_id
    assert resp.json()["message"] == "help me"


# ── GET / ─────────────────────────────────────────────────────────────────────

def test_ui_endpoint_returns_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
