#!/usr/bin/env python3
"""The mechanical gate. (Section 7, step 3)

426 demos cannot be reviewed by eyeball, so the gate is not review -- it is a
set of checks a machine can run: the manifest validates, the README says
something, there are at least three real tests, and no `TODO` from `scaffold.py`
survived. Anything this script rejects does not merge, with no human in the loop.

It deliberately does *not* run the demos. `demo verify` does that; keeping the
two separate means the cheap structural check still runs on a machine with no
torch installed.

    python scripts/audit_demos.py [--fix-hint] [lesson-substring]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.manifest import Demo, ManifestError, discover, load  # noqa: E402
from harness.parity import reference_repo  # noqa: E402
from harness.runner import DEMOS_ROOT  # noqa: E402

MIN_TESTS = 3
MIN_README_WORDS = 40
TODO = re.compile(r"\bTODO\b")

# A demo may declare a budget, but not an implausible one for its tier (D2).
TIER_CEILING_SECONDS = {"T0": 10, "T1": 300, "T2": 60, "T3": 1800}


def audit(demo: Demo) -> list[str]:
    """Every reason this demo is not ready, or an empty list."""
    problems: list[str] = []

    if TODO.search(demo.proves):
        problems.append("`proves` is still a scaffold TODO")
    if demo.runtime_seconds > TIER_CEILING_SECONDS[demo.tier]:
        problems.append(
            f"runtime_seconds {demo.runtime_seconds} exceeds the "
            f"{demo.tier} ceiling of {TIER_CEILING_SECONDS[demo.tier]}s -- "
            "either speed it up or move it to a slower tier"
        )

    entry = demo.entrypoint_path
    if not entry.exists():
        problems.append(f"no entrypoint at {demo.entrypoint}")
    else:
        source = entry.read_text(encoding="utf-8")
        if TODO.search(source):
            problems.append(f"{demo.entrypoint} still contains a scaffold TODO")
        if "NotImplementedError" in source:
            problems.append(f"{demo.entrypoint} raises NotImplementedError")
        if "--explain" not in source and "explain(" not in source:
            problems.append("entrypoint does not support --explain (D6)")

    readme = demo.path / "README.md"
    if not readme.is_file():
        problems.append("no README.md")
    else:
        text = readme.read_text(encoding="utf-8")
        if TODO.search(text):
            problems.append("README.md still contains a scaffold TODO")
        elif len(text.split()) < MIN_README_WORDS:
            problems.append(f"README.md is under {MIN_README_WORDS} words")

    tests = sorted((demo.path / "tests").glob("test_*.py"))
    if not tests:
        problems.append("no tests/test_*.py")
    else:
        bodies = [t.read_text(encoding="utf-8") for t in tests]
        count = sum(len(re.findall(r"^def test_", b, re.MULTILINE)) for b in bodies)
        if count < MIN_TESTS:
            problems.append(f"only {count} test(s); at least {MIN_TESTS} required")
        if any(TODO.search(b) for b in bodies):
            problems.append("tests still contain a scaffold TODO")

    if demo.tier == "T2" and not (demo.path / "cassettes" / demo.cassette).exists():
        # Not fatal: a demo can legitimately land before its tape is cut. It is
        # reported so the gap is visible rather than forgotten.
        problems.append(f"NOTE: cassette not recorded yet ({demo.cassette})")

    if demo.parity_with and not (reference_repo() / demo.parity_with).exists():
        problems.append(f"parity_with points at a file that no longer exists: "
                        f"{demo.parity_with}")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="?", help="only audit demos matching this")
    args = parser.parse_args()

    try:
        demos = discover(DEMOS_ROOT)
    except ManifestError as exc:
        print(f"FAIL  {exc}")
        return 1
    if args.target:
        demos = [d for d in demos if args.target in d.lesson]

    failures = 0
    notes = 0
    for demo in demos:
        problems = audit(demo)
        blocking = [p for p in problems if not p.startswith("NOTE:")]
        if not problems:
            print(f"ok    {demo.lesson}")
            continue
        failures += bool(blocking)
        notes += len(problems) - len(blocking)
        print(f"{'FAIL' if blocking else 'note'}  {demo.lesson}")
        for problem in problems:
            print(f"        {problem}")

    print(f"\n{len(demos)} demo(s) audited, {failures} failing, {notes} note(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
