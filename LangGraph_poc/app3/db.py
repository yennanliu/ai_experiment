"""SQLite persistence layer for resume tailoring history."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "history.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
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
                recruiter_feedback       TEXT    NOT NULL DEFAULT '',
                hiring_manager_feedback  TEXT    NOT NULL DEFAULT '',
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
    recruiter_feedback: str,
    hiring_manager_feedback: str,
    final_score: int,
) -> dict:
    ts = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO history
               (timestamp, raw_resume, job_description, tailored_resume, cover_letter,
                ats_report, recruiter_feedback, hiring_manager_feedback, final_score, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Draft')""",
            (ts, raw_resume, job_description, tailored_resume, cover_letter,
             json.dumps(ats_report), recruiter_feedback, hiring_manager_feedback, final_score),
        )
        row_id = cur.lastrowid
    return get_record(row_id)


def get_all_records() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM history ORDER BY id DESC").fetchall()
    return [_to_dict(r) for r in rows]


def get_record(record_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM history WHERE id = ?", (record_id,)).fetchone()
    return _to_dict(row) if row else None


def update_status(record_id: int, status: str) -> dict | None:
    with _connect() as conn:
        conn.execute("UPDATE history SET status = ? WHERE id = ?", (status, record_id))
    return get_record(record_id)


def _to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["ats_report"] = json.loads(d["ats_report"])
    return d
