"""Exercise 4 — every file that decides the score is inside the agent's repo.

    Design an evaluator firewall for a DGM-style loop in a repo you know.
    Identify every file the agent could edit that would change the
    evaluator's output.

Reading of the exercise: "a repo you know" is taken as the one this file is
committed to, because the enumeration then has to be exhaustive rather than
illustrative -- every path is checked to exist rather than named from memory.
The DGM-style loop is the obvious one: the agent's artifact is a solution
file, and the evaluator is `demo verify`.

**ANSWER: eight paths, and all eight are in the agent's own tree.** The
solution file (which contains `verify()`, the grading predicate itself),
`practice.yaml`, `tests/test_practice.py`, `harness/practice.py`,
`harness/tiers.py`, `harness/parity.py`, `scripts/audit_practice.py` and
`scripts/check_deps.py`. **8** of 8 exist here; **0** of 8 exist in the
reference checkout the same run reads.

**FINDING: the cheapest edit is one word in a YAML file.** `practice.yaml`
carries `tier`, and `tiers.selected` returns **True** for `T0` and
**False** for `T3` under the T0 ceiling CI sets. Changing one token makes the
exercise skip rather than fail, and a skip is a pass to `demo verify`. No
code is touched and no gate objects.

**FINDING: the design is already half built.** One input is outside reach:
the lesson's own `docs/en.md`, in a separate repository checked out beside
this one, is what `test_exercise_text_matches_upstream` hashes. That is the
firewall, working, on exactly **1** of the inputs that decide a score --
which is what makes the other eight legible as holes rather than as
inevitabilities.

**FINDING: the grading predicate ships with the artifact.** `DESIGN D13`
puts `verify()` in the same file as `solve()`, so of the eight paths the
first is the one the agent is *asked* to write. A firewall that moved the
harness upstream and left that arrangement would still be scoring the agent
with the agent's own test.

Structure: `reach()` resolves each path in both checkouts; `tier_bypass()`
runs the one-token edit against the real tier gate.
"""

from __future__ import annotations

import os
import pathlib

from harness import parity, practice, tiers

PHASE, LESSON = "15-autonomous-systems", "04-darwin-godel-machine"

HERE = pathlib.Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "harness").is_dir())
LOCAL = HERE.relative_to(ROOT)
REACHABLE = (
    LOCAL / pathlib.Path(__file__).name,          # the solution, which holds verify()
    LOCAL / "practice.yaml",
    LOCAL / "tests" / "test_practice.py",
    pathlib.Path("harness/practice.py"),
    pathlib.Path("harness/tiers.py"),
    pathlib.Path("harness/parity.py"),
    pathlib.Path("scripts/audit_practice.py"),
    pathlib.Path("scripts/check_deps.py"),
)
UPSTREAM_INPUT = pathlib.Path("docs/en.md")


def reach():
    """Each scoring input, and whether it sits in the agent's tree or upstream."""
    here = [path for path in REACHABLE if (ROOT / path).is_file()]
    reference = parity.lesson_dir(PHASE, LESSON)
    outside = [path for path in REACHABLE if (reference / path.name).is_file()]
    return here, outside, (reference / UPSTREAM_INPUT).is_file()


def tier_bypass():
    """The one-token edit: retag the exercise above the ceiling CI runs."""
    previous = os.environ.get("DEMO_TIER")
    os.environ["DEMO_TIER"] = "T0"
    try:
        return tiers.selected("T0"), tiers.selected("T3")
    finally:
        os.environ.pop("DEMO_TIER", None)
        if previous is not None:
            os.environ["DEMO_TIER"] = previous


def solve():
    here, outside, upstream = reach()
    kept, skipped = tier_bypass()
    manifest = (ROOT / LOCAL / "practice.yaml").read_text(encoding="utf-8")
    return {
        "paths": len(REACHABLE),
        "in_reach": len(here),
        "outside": len(outside),
        "upstream_input": upstream,
        "tier_declared": manifest.count("tier:"),
        "runs_at_t0": kept,
        "runs_at_t3": skipped,
        "predicate_in_artifact": "PRACTICE_IMPL" in pathlib.Path(__file__).read_text(
            encoding="utf-8"),
        "first_path": REACHABLE[0].name,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: eight paths, all eight in the agent's own tree",
            all([result["paths"] == 8, result["in_reach"] == 8,
                 result["outside"] == 0]),
            f"{result['in_reach']} of {result['paths']} scoring inputs resolve inside "
            f"this repository and {result['outside']} in the reference checkout the "
            "same run reads -- the solution file, its manifest, its test, three "
            "harness modules and both gate scripts",
        ),
        practice.Check(
            "FINDING: the cheapest edit is one word in a YAML file",
            all([result["runs_at_t0"], not result["runs_at_t3"],
                 result["tier_declared"] == 5]),
            f"practice.yaml declares tier {result['tier_declared']} times, and under "
            f"the T0 ceiling CI sets, tiers.selected is {result['runs_at_t0']} for T0 "
            f"and {result['runs_at_t3']} for T3 -- one token turns a failure into a "
            "skip, and a skip passes",
        ),
        practice.Check(
            "FINDING: the design is already half built",
            result["upstream_input"],
            "the lesson's own docs/en.md sits in a separate checkout and is what the "
            "spec-drift test hashes -- one input already outside reach, which is what "
            "makes the other eight legible as holes",
        ),
        practice.Check(
            "FINDING: the grading predicate ships with the artifact",
            all([result["predicate_in_artifact"],
                 result["first_path"] == pathlib.Path(__file__).name]),
            f"the first of the eight paths is {result['first_path']} -- the file the "
            "agent is asked to write, which carries verify(); moving the harness "
            "upstream would still leave the agent holding its own test",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
