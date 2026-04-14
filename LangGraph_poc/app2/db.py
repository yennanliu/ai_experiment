"""
SQLite persistence layer for email draft history.
Replaces the in-memory list so history survives server restarts.
"""
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
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp         TEXT    NOT NULL,
                inbound_email     TEXT    NOT NULL,
                email_type        TEXT    NOT NULL,
                draft_reply       TEXT    NOT NULL,
                checklist         TEXT    NOT NULL,
                confidence_score  INTEGER NOT NULL DEFAULT 0,
                confidence_reason TEXT    NOT NULL DEFAULT '',
                status            TEXT    NOT NULL DEFAULT 'Drafted'
            )
        """)


def insert_record(
    inbound_email: str,
    email_type: str,
    draft_reply: str,
    checklist: list,
    confidence_score: int,
    confidence_reason: str,
) -> dict:
    ts = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO history
               (timestamp, inbound_email, email_type, draft_reply,
                checklist, confidence_score, confidence_reason, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'Drafted')""",
            (ts, inbound_email, email_type, draft_reply,
             json.dumps(checklist), confidence_score, confidence_reason),
        )
        row_id = cur.lastrowid
    return get_record(row_id)


def get_all_records() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM history ORDER BY id DESC"
        ).fetchall()
    return [_to_dict(r) for r in rows]


def get_record(record_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM history WHERE id = ?", (record_id,)
        ).fetchone()
    return _to_dict(row) if row else None


def update_status(record_id: int, status: str) -> dict | None:
    with _connect() as conn:
        conn.execute(
            "UPDATE history SET status = ? WHERE id = ?", (status, record_id)
        )
    return get_record(record_id)


def _to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["checklist"] = json.loads(d["checklist"])
    return d
