"""
Merger — reads all fan-out artifacts and writes a consolidated integration summary.

Harness pattern: after parallel agents produce independent documents, a dedicated
merger agent synthesises them into one artifact that surfaces cross-cutting
concerns (interface alignment, gaps, conflicts, integration risks).
"""

from pathlib import Path

from . import provider
from .fan_out import ComponentResult

_ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

_SYSTEM = (
    "You are a senior technical architect reviewing designs from parallel specialist agents. "
    "Each agent independently designed one component of the same system.\n\n"
    "Write a concise INTEGRATION SUMMARY that:\n"
    "  1. Shows how the components fit together (data flow, shared interfaces)\n"
    "  2. Flags any gaps or conflicts between the designs\n"
    "  3. Lists 3–5 integration risks with mitigations\n\n"
    "Use markdown with clear headers. Be specific — name the components and artifacts."
)


def merge(task: str, results: list[ComponentResult]) -> str:
    """
    Synthesise parallel artifacts into one _merged.md integration summary.

    Failed components are noted but don't block the merge.
    Returns the merged content string.
    """
    sections = []
    for r in results:
        if not r.ok:
            sections.append(
                f"=== {r.component} [FAILED: {r.error}] ===\n"
                "(no artifact produced — treat as a gap)"
            )
            continue
        path = _ARTIFACTS_DIR / f"{r.artifact}.md"
        body = path.read_text() if path.exists() else r.text or "(empty)"
        sections.append(f"=== {r.component} ({r.elapsed:.1f}s) ===\n{body}")

    user = f"Task: {task}\n\n" + "\n\n".join(sections)
    summary = provider.simple_complete(system=_SYSTEM, user=user, max_tokens=1500)

    _ARTIFACTS_DIR.mkdir(exist_ok=True)
    (_ARTIFACTS_DIR / "_merged.md").write_text(summary)
    return summary
