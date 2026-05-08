"""
Evaluator agent — multi-dimensional rubric scoring.

Harness pattern: never let the generator grade its own work. A separate
evaluator agent, designed specifically to find flaws, produces honest scores
across four dimensions. The overall average drives the retry/feedback loop.
"""

import json
import re
from pathlib import Path

from . import provider

_ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

_SYSTEM = (
    "You are a strict technical quality evaluator. Score the delivered artifacts "
    "on four dimensions (each 1–5):\n"
    "  completeness        — covers all plan components?\n"
    "  correctness         — is the technical content accurate?\n"
    "  constraint_adherence — are all CONSTRAINTS.md rules followed?\n"
    "  quality             — well-structured, actionable, no stubs?\n\n"
    "Respond with valid JSON only — no prose, no fences:\n"
    '{"completeness": <int>, "correctness": <int>, '
    '"constraint_adherence": <int>, "quality": <int>, '
    '"overall": <float>, '
    '"feedback": "<one sentence summary>", '
    '"suggestions": ["<suggestion1>", "<suggestion2>"]}'
)


def evaluate(task: str, plan: dict, violations: list[dict]) -> dict:
    """
    Score the artifacts produced for a given plan.

    Returns a dict with keys:
      completeness, correctness, constraint_adherence, quality,
      overall (float), feedback (str), suggestions (list[str])
    """
    artifact_sections = []
    for name in plan.get("artifacts", []):
        path = _ARTIFACTS_DIR / f"{name}.md"
        body = path.read_text() if path.exists() else "[MISSING — never written]"
        artifact_sections.append(f"=== {name} ===\n{body}")

    violation_summary = (
        "\n".join(
            f"  [{v.get('severity', 'warning').upper()}] {v['artifact']}: {v['violation']}"
            for v in violations
        )
        if violations
        else "  None"
    )

    user = (
        f"Task: {task}\n\n"
        f"Plan components : {', '.join(plan.get('components', []))}\n"
        f"Success criteria: {', '.join(plan.get('success_criteria', []))}\n\n"
        f"Constraint violations detected:\n{violation_summary}\n\n"
        f"Artifacts produced:\n\n" + "\n\n".join(artifact_sections)
    )

    raw = provider.simple_complete(system=_SYSTEM, user=user)
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(match.group()) if match else {}

    # Compute overall if the model omitted it
    if "overall" not in result:
        dims = [result.get(k, 3) for k in ("completeness", "correctness", "constraint_adherence", "quality")]
        result["overall"] = round(sum(dims) / len(dims), 1)

    result.setdefault("feedback", "")
    result.setdefault("suggestions", [])
    return result
