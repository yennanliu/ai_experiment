"""Coverage: which lessons have a demo, computed from the tree. (D1)

The rule from DESIGN.md is that coverage is never hand-maintained. Because the
demo tree mirrors the reference tree byte-for-byte (D1), "what is built" is a
`diff` of two directory listings, and it cannot drift from reality the way a
table in a README does.

This module also detects the rot risk from the risk table: a lesson whose
`docs/en.md` changed since its demo recorded that doc's hash is flagged as
STALE, so the demo gets revisited instead of quietly describing an older lesson.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .manifest import Demo, discover
from .parity import reference_repo

BUILT = "built"
STALE = "stale"
MISSING = "missing"

GLYPH = {BUILT: "✅", STALE: "⚠️", MISSING: "⬚"}


def doc_hash(path: Path) -> str:
    """sha256 of a lesson doc, or "" if there is no doc."""
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


@dataclass(frozen=True)
class LessonStatus:
    lesson: str
    state: str
    demo: Demo | None = None
    note: str = ""

    @property
    def phase(self) -> str:
        return self.lesson.split("/")[1]


def reference_lessons(reference: Path | None = None) -> list[str]:
    """Every `phases/<phase>/<lesson>` path in the reference repo, sorted."""
    root = reference or reference_repo()
    lessons = [
        f"phases/{p.parent.name}/{p.name}"
        for p in sorted((root / "phases").glob("*/*"))
        if p.is_dir()
    ]
    return sorted(lessons)


def survey(demos_root: Path, reference: Path | None = None) -> list[LessonStatus]:
    """Join the reference lesson list against the demos that exist."""
    root = reference or reference_repo()
    built = {d.lesson: d for d in discover(demos_root)}

    statuses: list[LessonStatus] = []
    for lesson in reference_lessons(root):
        demo = built.pop(lesson, None)
        if demo is None:
            statuses.append(LessonStatus(lesson, MISSING))
            continue
        note = ""
        state = BUILT
        if demo.reference_doc and demo.reference_doc_sha256:
            current = doc_hash(root / demo.reference_doc)
            if current and current != demo.reference_doc_sha256:
                state = STALE
                note = "lesson doc changed since this demo was written"
        statuses.append(LessonStatus(lesson, state, demo, note))

    # A demo whose lesson no longer exists upstream is a real finding, not a
    # rounding error -- surface it rather than dropping it on the floor.
    for lesson, demo in sorted(built.items()):
        statuses.append(
            LessonStatus(lesson, STALE, demo, "no such lesson in the reference repo")
        )
    return sorted(statuses, key=lambda s: s.lesson)


def by_phase(statuses: list[LessonStatus]) -> dict[str, list[LessonStatus]]:
    grouped: dict[str, list[LessonStatus]] = {}
    for status in statuses:
        grouped.setdefault(status.phase, []).append(status)
    return grouped


def phase_table(statuses: list[LessonStatus]) -> str:
    """The generated coverage table that becomes the README body."""
    rows = [
        "| Phase | Lessons | Demos | Parity | Tiers | Coverage |",
        "|---|---:|---:|---:|---|---:|",
    ]
    total_lessons = total_demos = total_parity = 0
    for phase, group in sorted(by_phase(statuses).items()):
        demos = [s.demo for s in group if s.demo]
        parity = sum(1 for d in demos if d.has_parity)
        tiers = ", ".join(sorted({d.tier for d in demos})) or "—"
        pct = 100 * len(demos) / len(group) if group else 0
        rows.append(
            f"| {phase} | {len(group)} | {len(demos)} | {parity} | {tiers} | {pct:.0f}% |"
        )
        total_lessons += len(group)
        total_demos += len(demos)
        total_parity += parity
    pct = 100 * total_demos / total_lessons if total_lessons else 0
    rows.append(
        f"| **all** | **{total_lessons}** | **{total_demos}** | "
        f"**{total_parity}** | | **{pct:.1f}%** |"
    )
    return "\n".join(rows)


def summary(statuses: list[LessonStatus]) -> dict[str, int]:
    counts = {BUILT: 0, STALE: 0, MISSING: 0}
    for status in statuses:
        counts[status.state] += 1
    return counts
