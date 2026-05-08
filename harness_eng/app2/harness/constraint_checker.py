"""
Constraint checker — scans each artifact against CONSTRAINTS.md.

Harness pattern: a dedicated agent trained to *find problems* catches issues
that the Generator (trained to *produce output*) routinely misses. Separating
these roles is the key insight from Anthropic's three-agent architecture.
"""

import json
import re
from pathlib import Path

from . import provider

_CONSTRAINTS_MD = Path(__file__).parent.parent / "CONSTRAINTS.md"
_ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

_SYSTEM = (
    "You are an architectural constraint checker. "
    "Given a set of constraints and one artifact, identify every violation. "
    "Be strict — if a constraint signal word or section is absent, flag it. "
    "Respond with valid JSON only, no prose:\n"
    '{"violations": [{"constraint": "<rule name>", "violation": "<what is missing or wrong>", '
    '"severity": "critical|warning"}]}\n'
    'If there are no violations, return: {"violations": []}'
)


def check(artifact_names: list[str]) -> list[dict]:
    """
    Check each artifact against CONSTRAINTS.md.

    Returns a flat list of violation dicts, each with keys:
      artifact, constraint, violation, severity
    """
    constraints = _CONSTRAINTS_MD.read_text()
    all_violations: list[dict] = []

    for name in artifact_names:
        path = _ARTIFACTS_DIR / f"{name}.md"

        if not path.exists():
            all_violations.append({
                "artifact": name,
                "constraint": "Existence",
                "violation": f"Artifact '{name}.md' was never written by the generator",
                "severity": "critical",
            })
            continue

        content = path.read_text()
        user = f"Constraints:\n{constraints}\n\nArtifact '{name}':\n{content}"
        raw = provider.simple_complete(system=_SYSTEM, user=user)

        # Strip markdown fences defensively
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            result = json.loads(match.group()) if match else {"violations": []}

        for v in result.get("violations", []):
            all_violations.append({"artifact": name, **v})

    return all_violations
