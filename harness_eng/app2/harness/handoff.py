"""
Handoff artifact management — multi-session continuity.

Harness pattern: after each component is generated, progress is written to a
JSON file. The next session reads this file first, skips completed components,
and resumes from where it left off — eliminating redundant work and context loss.
"""

import json
import time
from pathlib import Path

_HANDOFF_DIR = Path(__file__).parent.parent / "artifacts" / "handoffs"


def save_progress(session_id: str, data: dict) -> None:
    _HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    path = _HANDOFF_DIR / f"progress_{session_id}.json"
    data["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps(data, indent=2))


def load_progress(session_id: str) -> dict | None:
    path = _HANDOFF_DIR / f"progress_{session_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def get_latest_session() -> str | None:
    _HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        _HANDOFF_DIR.glob("progress_*.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not files:
        return None
    # filename is "progress_{session_id}.json"
    return files[0].stem[len("progress_"):]


def list_sessions() -> list[dict]:
    _HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    sessions = []
    for f in sorted(_HANDOFF_DIR.glob("progress_*.json")):
        data = json.loads(f.read_text())
        sessions.append({
            "session_id": f.stem[len("progress_"):],
            "task": data.get("task", "unknown"),
            "saved_at": data.get("saved_at"),
            "completed": data.get("completed_components", []),
            "total": len(data.get("plan", {}).get("components", [])),
        })
    return sessions
