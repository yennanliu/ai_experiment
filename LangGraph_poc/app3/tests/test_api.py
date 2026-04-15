"""Integration tests for FastAPI routes (no real LLM calls)."""
import json
import os
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def set_env(monkeypatch, tmp_path):
    """Provide required env vars and redirect DB to temp file."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("APP_API_KEY", "")
    # Redirect DB
    import db as db_module
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(db_module, "DB_PATH", db_file)
    db_module.init_db()


@pytest.fixture
def client(set_env):
    from httpx import AsyncClient, ASGITransport
    from main import app
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    async with client as c:
        r = await c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_history_empty(client):
    async with client as c:
        r = await c.get("/history")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 0
    assert data["items"] == []


@pytest.mark.asyncio
async def test_history_pagination(client):
    import db as db_module
    for i in range(5):
        db_module.insert_record(
            raw_resume=f"resume {i}", job_description=f"jd {i}",
            tailored_resume="t", cover_letter="c",
            ats_report={"score": i * 10, "missing_keywords": [], "suggestions": []},
            recruiter_feedback={"verdict": "Pass", "feedback": "ok"},
            hiring_manager_feedback={"score": 50, "verdict": "Yes", "rationale": "fine"},
            final_score=i * 10,
        )
    async with client as c:
        r = await c.get("/history?limit=3&offset=0")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 5
    assert len(data["items"]) == 3


@pytest.mark.asyncio
async def test_update_status(client):
    import db as db_module
    db_module.insert_record(
        raw_resume="r", job_description="jd", tailored_resume="t", cover_letter="c",
        ats_report={}, recruiter_feedback={}, hiring_manager_feedback={}, final_score=0,
    )
    async with client as c:
        r = await c.patch("/history/1/status", json={"status": "Applied"})
    assert r.status_code == 200
    assert r.json()["status"] == "Applied"


@pytest.mark.asyncio
async def test_update_status_not_found(client):
    async with client as c:
        r = await c.patch("/history/999/status", json={"status": "Applied"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_delete_record(client):
    import db as db_module
    db_module.insert_record(
        raw_resume="r", job_description="jd", tailored_resume="t", cover_letter="c",
        ats_report={}, recruiter_feedback={}, hiring_manager_feedback={}, final_score=0,
    )
    async with client as c:
        r = await c.delete("/history/1")
    assert r.status_code == 204
    assert db_module.get_record(1) is None


@pytest.mark.asyncio
async def test_delete_nonexistent(client):
    async with client as c:
        r = await c.delete("/history/999")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_tailor_validation_too_short(client):
    async with client as c:
        r = await c.post("/tailor", json={"resume": "short", "job_description": "jd"})
    assert r.status_code == 422
