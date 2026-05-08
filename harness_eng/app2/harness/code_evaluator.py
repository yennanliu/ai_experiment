"""
Code evaluator — scores generated Python source files.

Harness pattern: a dedicated evaluator reads workspace/*.py files and
scores them on a code-specific rubric (correctness, completeness,
code_quality, testability) — separate from the design evaluator's rubric.
"""

import json
import re
from pathlib import Path

from . import config, provider

_WORKSPACE = Path(__file__).parent.parent / "workspace"

_SYSTEM = (
    "You are a strict Python code reviewer. "
    "Score the provided source files on four dimensions (each 1–5):\n"
    "  correctness   — logic is sound; no obvious bugs or runtime errors\n"
    "  completeness  — all required functionality is implemented; no stubs\n"
    "  code_quality  — idiomatic Python: type hints, docstrings, error handling\n"
    "  testability   — tests exist, are meaningful, and would pass against the code\n\n"
    "Respond with valid JSON only — no prose, no fences:\n"
    '{"correctness": <int>, "completeness": <int>, '
    '"code_quality": <int>, "testability": <int>, '
    '"overall": <float>, "feedback": "<one sentence>", '
    '"suggestions": ["<fix1>", "<fix2>"]}'
)


def evaluate(task: str, files: list[Path]) -> dict:
    """Score a list of workspace Python files against the given task."""
    sections = []
    for f in files:
        body = f.read_text() if f.exists() else "[MISSING — file was never written]"
        lines = body.count("\n") + 1
        sections.append(f"=== {f.name} ({lines} lines) ===\n{body}")

    if not sections:
        return {
            "correctness": 0, "completeness": 0, "code_quality": 0, "testability": 0,
            "overall": 0.0, "feedback": "No files were written to workspace.",
            "suggestions": ["Ensure write_code is called for each component"],
        }

    user = f"Task: {task}\n\n" + "\n\n".join(sections)

    raw = provider.simple_complete(system=_SYSTEM, user=user, max_tokens=config.MAX_TOKENS)
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(match.group()) if match else {}

    if "overall" not in result:
        dims = [result.get(k, 3) for k in ("correctness", "completeness", "code_quality", "testability")]
        result["overall"] = round(sum(dims) / len(dims), 1)

    result.setdefault("feedback", "")
    result.setdefault("suggestions", [])
    return result


def list_workspace_files() -> list[Path]:
    """Return all .py files currently in the workspace."""
    _WORKSPACE.mkdir(exist_ok=True)
    return sorted(_WORKSPACE.glob("*.py"))
