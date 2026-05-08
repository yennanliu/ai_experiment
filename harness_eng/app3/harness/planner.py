"""
Planner agent — identical to app2; breaks a task into a structured JSON plan.
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
    system = _PLANNER_MD.read_text()
    raw = provider.simple_complete(system=system, user=f"Task: {task}")

    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    data.setdefault("task", task)
    if not data.get("components"):
        data["components"] = ["design", "implementation", "error_handling"]
    if not data.get("artifacts"):
        data["artifacts"] = [c.replace(" ", "_").lower() for c in data["components"]]
    data.setdefault("constraints", _FALLBACK_CONSTRAINTS)
    data.setdefault("success_criteria", ["all components documented", "no constraint violations"])

    return data
