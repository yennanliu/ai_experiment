"""Unit tests for db.py using a temporary in-memory SQLite database."""
import json
import sqlite3
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """Redirect DB_PATH to a temp file for each test."""
    db_file = tmp_path / "test.db"
    import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", db_file)
    db_module.init_db()
    return db_module


def test_insert_and_get(tmp_db):
    record = tmp_db.insert_record(
        raw_resume="raw",
        job_description="jd",
        tailored_resume="tailored",
        cover_letter="cover",
        ats_report={"score": 70, "missing_keywords": ["Python"], "suggestions": []},
        recruiter_feedback={"verdict": "Pass", "feedback": "Good."},
        hiring_manager_feedback={"score": 80, "verdict": "Yes", "rationale": "Fit."},
        final_score=76,
    )
    assert record["id"] == 1
    assert record["final_score"] == 76
    assert record["ats_report"]["score"] == 70
    assert record["recruiter_feedback"]["verdict"] == "Pass"
    assert record["hiring_manager_feedback"]["score"] == 80


def test_get_all_records_pagination(tmp_db):
    for i in range(5):
        tmp_db.insert_record(
            raw_resume=f"resume {i}", job_description=f"jd {i}",
            tailored_resume="t", cover_letter="c",
            ats_report={}, recruiter_feedback={}, hiring_manager_feedback={},
            final_score=i * 10,
        )
    page1 = tmp_db.get_all_records(limit=3, offset=0)
    assert page1["total"] == 5
    assert len(page1["items"]) == 3

    page2 = tmp_db.get_all_records(limit=3, offset=3)
    assert len(page2["items"]) == 2


def test_update_status(tmp_db):
    tmp_db.insert_record(
        raw_resume="r", job_description="jd", tailored_resume="t", cover_letter="c",
        ats_report={}, recruiter_feedback={}, hiring_manager_feedback={}, final_score=0,
    )
    updated = tmp_db.update_status(1, "Applied")
    assert updated["status"] == "Applied"


def test_delete_record(tmp_db):
    tmp_db.insert_record(
        raw_resume="r", job_description="jd", tailored_resume="t", cover_letter="c",
        ats_report={}, recruiter_feedback={}, hiring_manager_feedback={}, final_score=0,
    )
    assert tmp_db.delete_record(1) is True
    assert tmp_db.get_record(1) is None


def test_delete_nonexistent(tmp_db):
    assert tmp_db.delete_record(999) is False


def test_parse_json_field_legacy_string(tmp_db):
    """_parse_json_field should store non-JSON legacy strings under _raw key."""
    val = tmp_db._parse_json_field("VERDICT: Pass\nFEEDBACK: Looks good", {})
    assert "_raw" in val


def test_parse_json_field_valid_json(tmp_db):
    val = tmp_db._parse_json_field('{"verdict":"Pass","feedback":"ok"}', {})
    assert val["verdict"] == "Pass"


def test_parse_json_field_empty(tmp_db):
    val = tmp_db._parse_json_field("", {"score": 0})
    assert val == {"score": 0}
