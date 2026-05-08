"""
Planner agent — breaks a high-level task into a structured JSON plan.

Harness pattern: the Planner runs once per task; its JSON output drives
the Generator and Constraint Checker, so all three agents share one schema.
"""

import json
import re
from pathlib import Path

from . import provider

_PLANNER_MD = Path(__file__).parent.parent / "PLANNER.md"

_FALLBACK_CONSTRAINTS = [
    "must address security",
    "must define public interfaces",
    "must document error handling",
    "must state scalability characteristics",
]


def plan(task: str) -> dict:
    """Return a structured plan dict for the given task."""
    system = _PLANNER_MD.read_text()
    raw = provider.simple_complete(system=system, user=f"Task: {task}")

    # Strip markdown fences if the model added them
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Best-effort extraction if the model wrapped JSON in prose
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    # Ensure required keys with sensible defaults
    data.setdefault("task", task)
    if not data.get("components"):
        data["components"] = ["design", "implementation", "error_handling"]
    if not data.get("artifacts"):
        data["artifacts"] = [c.replace(" ", "_").lower() for c in data["components"]]
    data.setdefault("constraints", _FALLBACK_CONSTRAINTS)
    data.setdefault("success_criteria", ["all components documented", "no constraint violations"])

    return data
