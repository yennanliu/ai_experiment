"""Reference tree vs demo tree, plus spec-drift detection (`DESIGN D15`).

Coverage is measured against the reference checkout, never against a stored
list, so a lesson added upstream shows up as unbuilt rather than as nothing.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import re

from . import manifest, parity

DEMOS = pathlib.Path(__file__).resolve().parent.parent / "demos" / "phases"
_EX_HEAD = re.compile(r"^##\s+(Exercises|練習)\s*$", re.M)
_NUMBERED = re.compile(r"^\s*(\d+)\.\s+(.*)$")


def exercise_block(markdown: str) -> list:
    """The `## Exercises` / `## 練習` list, wrapped lines folded back together."""
    match = _EX_HEAD.search(markdown)
    if not match:
        return []
    body = markdown[match.end():]
    end = re.search(r"^##\s+", body, re.M)
    if end:
        body = body[: end.start()]
    items, current = [], None
    for line in body.splitlines():
        numbered = _NUMBERED.match(line)
        if numbered:
            if current is not None:
                items.append(current.strip())
            current = numbered.group(2)
        elif current is not None:
            if line.strip():
                current += " " + line.strip()
            else:
                items.append(current.strip())
                current = None
    if current is not None:
        items.append(current.strip())
    return items


def spec_hash(text: str) -> str:
    """Hash the exercise *text*, so unrelated lesson edits do not read as drift."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()[:12]


@dataclasses.dataclass
class LessonRow:
    phase: str
    lesson: str
    upstream: int
    built: int
    drifted: tuple = ()

    @property
    def status(self) -> str:
        if self.upstream == 0:
            return "—"
        if self.drifted:
            return "⚠ drifted"
        if self.built == 0:
            return "⬚ unbuilt"
        return "✅ verified" if self.built >= self.upstream else f"{self.built}/{self.upstream}"


def scan(phase_filter: str | None = None) -> list:
    root = parity.find_reference_root() / "phases"
    rows = []
    for phase_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if phase_filter and not phase_dir.name.startswith(phase_filter):
            continue
        for lesson_dir in sorted(p for p in phase_dir.iterdir() if p.is_dir()):
            doc = lesson_dir / "docs" / "en.md"
            if not doc.is_file():
                continue
            upstream = exercise_block(doc.read_text(encoding="utf-8"))
            built, drifted = 0, []
            man = DEMOS / phase_dir.name / lesson_dir.name / "practice" / "practice.yaml"
            if man.is_file():
                practice = manifest.load_practice(man)
                built = len(practice.exercises)
                for ex in practice.exercises:
                    if ex.index <= len(upstream):
                        if spec_hash(ex.en) != spec_hash(upstream[ex.index - 1]):
                            drifted.append(ex.index)
                    else:
                        drifted.append(ex.index)
            rows.append(LessonRow(phase_dir.name, lesson_dir.name,
                                  len(upstream), built, tuple(drifted)))
    return rows


def table(rows: list) -> str:
    total_up = sum(r.upstream for r in rows)
    total_built = sum(r.built for r in rows)
    out = [f"{'lesson':<58}{'built':>7}{'upstream':>10}  status", "-" * 92]
    for row in rows:
        if row.upstream == 0 and row.built == 0:
            continue
        out.append(f"{row.phase + '/' + row.lesson:<58}{row.built:>7}{row.upstream:>10}  {row.status}")
    pct = (100.0 * total_built / total_up) if total_up else 0.0
    out += ["-" * 92, f"{'TOTAL':<58}{total_built:>7}{total_up:>10}  {pct:.1f}%"]
    return "\n".join(out)
