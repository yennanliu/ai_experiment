"""Exercise 3 — the scaffold gap is widest for mediocre models.

    Pick one task from your bug backlog that would require 10+ lines of
    change across two files. Estimate the end-to-end success probability for
    a frontier model under (a) JSON tool calls and (b) CodeAct. Justify the
    gap.

Reading of the exercise: a task recalled from a backlog cannot be checked, so
the task is one this repository actually has, and the two files are resolved
rather than named. The estimate then uses the turn counts the lesson's own
scaffolds produce -- `bugs + 1` against `2` -- rather than an intuition about
how many turns each would take.

**ANSWER: 0.857 against 0.902 at 95% per-step, and the gap is p^2(1 - p).**
The task is enforcing the manifest's reference-use field, which
`harness/manifest.py` parses **4** times and both gates read **0** times. Two
files change: `scripts/audit_practice.py` gains the check, and this lesson's
`practice.yaml` -- **5** exercise entries -- gains the field. Two files means
**3** JSON turns and **2** CodeAct turns, so end-to-end is `p^3` against
`p^2`.

**FINDING: the gap is a hump, not a slope.** `p^2 - p^3` peaks at
`p = 2/3`, where it is **0.148**. At **0.99** it is **0.010** and at
**0.50** it is **0.125**. The scaffold choice matters most for models in the
middle and washes out at the frontier -- which is the opposite of how the
choice is usually argued.

**FINDING: the gap grows with the file count, not the line count.** For a
`k`-file change the difference is `p^2 (1 - p^(k-1))`: at 95% per-step it is
**0.045** across two files, **0.167** across five and **0.334** across ten.
Ten lines or ten thousand changes nothing, because the scaffolds are paid per
action and an action is a file.

**FINDING: the price is stated in the other column.** CodeAct's observed
blast radius for this task is **2** files against JSON's **1**, and the
failure it buys is a partial edit -- the check landing without the field, or
the field without the check. The JSON scaffold cannot produce that state,
because each of its turns is validated before the next begins.

Structure: `task()` resolves the two files and the field they are about;
`odds()` prices both scaffolds at one per-step reliability.
"""

from __future__ import annotations

import pathlib

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "09-coding-agent-landscape"

HERE = pathlib.Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "harness").is_dir())
FIELD = "uses_reference"
PARSER = "harness/manifest.py"
GATES = ("scripts/audit_practice.py", "scripts/check_deps.py")
RELIABILITY, FILES = 0.95, 2


def mentions(path):
    return (ROOT / path).read_text(encoding="utf-8").count(FIELD)


def task():
    """The two files the change touches, and how much of each it reaches."""
    manifest = HERE / "practice.yaml"
    return {
        "parsed": mentions(PARSER),
        "gated": [mentions(gate) for gate in GATES],
        "entries": manifest.read_text(encoding="utf-8").count("  - index:"),
        "files": [(ROOT / GATES[0]).is_file(), manifest.is_file()],
    }


def odds(per_step, files, json_turns=None):
    """End-to-end success for both scaffolds at a per-step reliability."""
    turns = files + 1 if json_turns is None else json_turns
    return round(per_step ** turns, 4), round(per_step ** 2, 4)


def gap(per_step, files=FILES):
    return round(per_step ** 2 - per_step ** (files + 1), 4)


def peak(steps=1000):
    """Where p^2 - p^3 is largest, searched on the unrounded value."""
    best = max(range(1, steps), key=lambda index: (index / steps) ** 2 - (index / steps) ** 3)
    return round(best / steps, 3), gap(best / steps)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped = ref.JsonScaffold()
    shipped.run()
    resolved = task()
    return {
        **resolved,
        "reliability": RELIABILITY,
        "turns": [FILES + 1, 2],
        "end_to_end": list(odds(RELIABILITY, FILES)),
        "gap": gap(RELIABILITY),
        "peak": list(peak()),
        "at_high": gap(0.99),
        "at_half": gap(0.50),
        "by_files": [gap(RELIABILITY, count) for count in (2, 5, 10)],
        "json_radius": shipped.blast_radius(),
        "codeact_radius": FILES,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 0.857 against 0.902 at 95% per-step",
            all([result["parsed"] == 4, result["gated"] == [0, 0],
                 result["entries"] == 5, result["files"] == [True, True],
                 result["end_to_end"] == [0.8574, 0.9025], result["turns"] == [3, 2]]),
            f"the field is parsed {result['parsed']} times by the manifest reader and "
            f"{result['gated']} times by the two gates, and the second file holds "
            f"{result['entries']} exercise entries; at {result['reliability']} per-step "
            f"the {len(result['files'])} files are {result['turns']} turns and "
            f"{result['end_to_end']} end-to-end",
        ),
        practice.Check(
            "FINDING: the gap is a hump, not a slope",
            all([result["gap"] == 0.0451, result["peak"] == [0.667, 0.1481],
                 result["at_high"] == 0.0098, result["at_half"] == 0.125]),
            f"p^2 - p^3 is {result['gap']} at {result['reliability']}, peaks at "
            f"p = {result['peak'][0]} with {result['peak'][1]}, and falls to "
            f"{result['at_high']} at 0.99 -- the scaffold matters most in the middle",
        ),
        practice.Check(
            "FINDING: the gap grows with the file count, not the line count",
            all([result["by_files"] == [0.0451, 0.1674, 0.3337]]),
            f"across 2, 5 and 10 files the difference is {result['by_files']} at "
            f"{result['reliability']} per-step -- the scaffolds are paid per action and "
            "an action is a file, so the line count never enters",
        ),
        practice.Check(
            "FINDING: the price is stated in the other column",
            all([result["json_radius"] == 1, result["codeact_radius"] == 2]),
            f"CodeAct's radius for this task is {result['codeact_radius']} files "
            f"against JSON's {result['json_radius']}, and what it buys is a partial "
            "edit -- the check without the field, or the field without the check",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
