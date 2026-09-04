#!/usr/bin/env python3
"""Re-derive every number in `DESIGN §1` and `DESIGN §5` from the reference tree.

The design doc's rule is that its figures stay re-checkable rather than becoming
folklore. This is the script that makes that true: run it and diff.
"""

from __future__ import annotations

import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from harness import coverage, parity  # noqa: E402


def census():
    root = parity.find_reference_root() / "phases"
    per_phase = collections.OrderedDict()
    totals = collections.Counter()
    for phase_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        lessons = sorted(p for p in phase_dir.iterdir() if p.is_dir())
        counts = collections.Counter()
        for lesson in lessons:
            doc = lesson / "docs" / "en.md"
            if not doc.is_file():
                counts["no_docs"] += 1
                continue
            text = doc.read_text(encoding="utf-8")
            items = coverage.exercise_block(text)
            counts["lessons"] += 1
            counts["exercises"] += len(items)
            has_lab = "## Practice Lab" in text
            if has_lab:
                # DESIGN §1 counts these two ways; report both so the figure is
                # unambiguous. 13/10-mcp-resources-and-prompts has both headings.
                counts["lab_heading"] += 1
            if items:
                counts["with_exercises"] += 1
            elif has_lab:
                counts["lab_only"] += 1
            else:
                counts["no_practice"] += 1
            if (lesson / "docs" / "zh.md").is_file():
                counts["bilingual"] += 1
        per_phase[phase_dir.name] = counts
        totals.update(counts)
    return per_phase, totals


def main() -> int:
    per_phase, totals = census()
    print(f"{'phase':<36}{'lessons':>8}{'exercises':>11}{'w/ex':>7}{'labs':>6}{'none':>6}")
    print("-" * 74)
    for phase, counts in per_phase.items():
        print(f"{phase:<36}{counts['lessons']:>8}{counts['exercises']:>11}"
              f"{counts['with_exercises']:>7}{counts['lab_only']:>6}{counts['no_practice']:>6}")
    print("-" * 74)
    print(f"{'TOTAL':<36}{totals['lessons']:>8}{totals['exercises']:>11}"
          f"{totals['with_exercises']:>7}{totals['lab_only']:>6}{totals['no_practice']:>6}")
    print(f"\nphases                                  : {len(per_phase)}")
    print(f"lessons with a ## Practice Lab heading  : {totals['lab_heading']}")
    print(f"  ...of those, lab instead of exercises : {totals['lab_only']}")
    print(f"bilingual lessons (docs/zh.md present)  : {totals['bilingual']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
