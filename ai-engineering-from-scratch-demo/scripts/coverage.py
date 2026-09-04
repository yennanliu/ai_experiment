#!/usr/bin/env python3
"""Generate the root README's coverage table from the reference tree.

The table is never hand-maintained: a lesson added upstream appears here as
unbuilt on the next run. `--check` exits non-zero if the committed README is
stale, which is what stops it drifting.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from harness import coverage  # noqa: E402

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"
START = "<!-- coverage:start -->"
END = "<!-- coverage:end -->"


def render(rows) -> str:
    by_phase = {}
    for row in rows:
        counts = by_phase.setdefault(row.phase, [0, 0, 0])
        counts[0] += row.upstream
        counts[1] += row.built
        counts[2] += 1 if row.built else 0
    built = sum(c[1] for c in by_phase.values())
    upstream = sum(c[0] for c in by_phase.values())
    lines = [START, "",
             f"**{built} / {upstream} exercises** "
             f"({100.0 * built / upstream:.1f}%) across {len(by_phase)} phases.", "",
             "| Phase | Exercises | Solved | Lessons started |", "|---|---:|---:|---:|"]
    for phase, (up, done, lessons) in by_phase.items():
        mark = "✅" if done and done >= up else ("🚧" if done else "⬚")
        lines.append(f"| {mark} `{phase}` | {up} | {done} | {lessons} |")
    lines += ["", f"Regenerate with `uv run python scripts/coverage.py`.", "", END]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if README.md is stale")
    args = parser.parse_args(argv)
    table = render(coverage.scan())
    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print(f"{README}: no {START} / {END} markers", file=sys.stderr)
        return 1
    head, _, rest = text.partition(START)
    _, _, tail = rest.partition(END)
    updated = head + table + tail
    if args.check:
        if updated != text:
            print("README.md coverage table is stale; run scripts/coverage.py")
            return 1
        print("coverage table: up to date")
        return 0
    README.write_text(updated, encoding="utf-8")
    print(f"updated {README.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
