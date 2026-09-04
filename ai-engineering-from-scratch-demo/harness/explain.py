"""`--explain`: concept plus a link back to the lesson (`DESIGN D6`).

Zero dependencies, and the URL is *derived* from the mirrored path rather than
stored, so D1's path-identity is what keeps the links correct.
"""

from __future__ import annotations

SITE = "https://yennj12.js.org/ai-engineering-from-scratch"


def lesson_url(phase: str, lesson: str) -> str:
    return f"{SITE}/phases/{phase}/{lesson}/"


def render(phase: str, lesson: str, exercise=None, concept: str = "") -> str:
    lines = [f"lesson : {phase}/{lesson}", f"source : {lesson_url(phase, lesson)}"]
    if exercise is not None:
        lines.append(f"exercise {exercise.index} ({exercise.kind}, {exercise.tier})")
        lines.append("")
        lines.append("  en: " + exercise.en)
        lines.append("  zh: " + exercise.zh)
        if exercise.verifies:
            lines.append("")
            lines.append("  verifies: " + exercise.verifies)
        if exercise.uses_reference:
            lines.append("  uses    : " + ", ".join(exercise.uses_reference))
    if concept:
        lines += ["", concept.strip()]
    return "\n".join(lines)
