"""SQLite persistence layer for resume tailoring history."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "history.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # WAL mode: better concurrent read performance, safer writes
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id                       INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp                TEXT    NOT NULL,
                raw_resume               TEXT    NOT NULL,
                job_description          TEXT    NOT NULL,
                tailored_resume          TEXT    NOT NULL,
                cover_letter             TEXT    NOT NULL,
                ats_report               TEXT    NOT NULL,
                recruiter_feedback       TEXT    NOT NULL DEFAULT '{}',
                hiring_manager_feedback  TEXT    NOT NULL DEFAULT '{}',
                final_score              INTEGER NOT NULL DEFAULT 0,
                status                   TEXT    NOT NULL DEFAULT 'Draft'
            )
        """)


def insert_record(
    raw_resume: str,
    job_description: str,
    tailored_resume: str,
    cover_letter: str,
    ats_report: dict,
    recruiter_feedback: dict,
    hiring_manager_feedback: dict,
    final_score: int,
) -> dict:
    ts = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO history
               (timestamp, raw_resume, job_description, tailored_resume, cover_letter,
                ats_report, recruiter_feedback, hiring_manager_feedback, final_score, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Draft')""",
            (
                ts, raw_resume, job_description, tailored_resume, cover_letter,
                json.dumps(ats_report),
                json.dumps(recruiter_feedback),
                json.dumps(hiring_manager_feedback),
                final_score,
            ),
        )
        row_id = cur.lastrowid
    return get_record(row_id)


def get_all_records(limit: int = 50, offset: int = 0) -> dict:
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM history ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return {"total": total, "items": [_to_dict(r) for r in rows]}


def get_record(record_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM history WHERE id = ?", (record_id,)).fetchone()
    return _to_dict(row) if row else None


def update_status(record_id: int, status: str) -> dict | None:
    with _connect() as conn:
        conn.execute("UPDATE history SET status = ? WHERE id = ?", (status, record_id))
    return get_record(record_id)


def delete_record(record_id: int) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM history WHERE id = ?", (record_id,))
    return cur.rowcount > 0


def _parse_json_field(value: str, fallback: dict) -> dict:
    """Parse a JSON TEXT field; handle legacy plain-text format gracefully."""
    if not value:
        return fallback
    try:
        result = json.loads(value)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    # Legacy: plain-text key:value format (old records)
    return {"_raw": value}


def _to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["ats_report"] = _parse_json_field(d["ats_report"], {"score": 0, "missing_keywords": [], "suggestions": []})
    d["recruiter_feedback"] = _parse_json_field(d["recruiter_feedback"], {})
    d["hiring_manager_feedback"] = _parse_json_field(d["hiring_manager_feedback"], {})
    return d
