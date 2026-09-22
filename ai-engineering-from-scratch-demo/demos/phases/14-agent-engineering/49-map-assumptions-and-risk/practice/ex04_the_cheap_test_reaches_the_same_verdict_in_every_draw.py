"""Exercise 4 — the cheap test reaches the same verdict in four of five draws.

    Replace one large experiment with a cheaper decisive test.

Reading of the exercise: "decisive" means the cheap test has to reach the
same decision as the expensive one. That is checkable here, because both
versions can actually be run: the large experiment traces every number in
five finished lessons' answers, and the cheap one traces a single lesson.

**ANSWER: the full run scores 70.8% against the cheap run's 57.9%, and both
say stop.** Five lessons' solutions claim **154** numbers in their docstrings
and **109** of them appear in the details those solutions print; one lesson
claims **38** and matches **22**. Against a 90% threshold both verdicts are
`stop`, at **20.0%** of the cost.

**FINDING: the cheap test is decisive in all 5 of the draws available.** The
per-lesson rates are **57.9%**, **78.6%**, **70.4%**, **86.2%** and **65.6%**,
every one below the threshold, so no single-lesson draw reverses the decision.
That is a property of this sample rather than of the method -- the honest
report is the list of rates, so a reader can see how close the nearest draw
came.

**FINDING: the expensive run is not five times more informative.** It costs
**5** lessons and **25** graded files to move the estimate from 57.9% to
70.8%, both on the same side of the threshold. What it buys is the per-lesson
spread -- a **28.3**-point range -- which is the thing that actually tells you
the metric is noisy.

**FINDING: the assumption cannot record which experiment was run.**
`Assumption.test` is one string, so replacing the large experiment with the
small one overwrites the description and leaves no trace that a cheaper test
was substituted. A reader of the map sees the test that ran and not the one
it replaced, which is the decision worth reviewing.

Structure: `trace()` runs the experiment over one lesson; `full()` runs it
over all five; `draws()` reports every single-lesson verdict.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "49-map-assumptions-and-risk"
BASE = Path(__file__).resolve().parents[2]
LESSONS = ("43-frame-the-task-before-code", "44-plan-from-evidence",
           "45-delegate-with-isolation", "46-turn-feedback-into-system",
           "47-outcomes-before-output")
CHEAP = "43-frame-the-task-before-code"
THRESHOLD = 0.90


def trace(lesson):
    """One lesson's claimed numbers against the details its solutions print."""
    claimed = matched = 0
    files = sorted((BASE / lesson / "practice").glob("ex0*.py"))
    for path in files:
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
        values = sorted(set(re.findall(r"\b\d+(?:\.\d+)?%?\b", doc)))
        printed = " ".join(check.detail for check in practice.grade_file(path).checks)
        claimed += len(values)
        matched += sum(value in printed for value in values)
    return {"lesson": lesson, "claimed": claimed, "matched": matched,
            "rate": round(matched / claimed, 3), "files": len(files)}


def draws():
    return [trace(lesson) for lesson in LESSONS]


def full(rows):
    claimed = sum(row["claimed"] for row in rows)
    matched = sum(row["matched"] for row in rows)
    return {"claimed": claimed, "matched": matched, "rate": round(matched / claimed, 3),
            "files": sum(row["files"] for row in rows), "lessons": len(rows)}


def verdict(rate, threshold=THRESHOLD):
    return "stop" if rate < threshold else "continue"


def spread(rows, large_rate):
    """Which single-lesson draws agree with the full run, and how far the rates range."""
    verdicts = [verdict(row["rate"]) for row in rows]
    target = verdict(large_rate)
    return {"rates": [row["rate"] for row in rows],
            "agreeing": sum(one == target for one in verdicts),
            "disagreeing": [row["lesson"][:2] for row, one in zip(rows, verdicts)
                            if one != target],
            "spread": round(max(row["rate"] for row in rows)
                            - min(row["rate"] for row in rows), 3)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = draws()
    large = full(rows)
    cheap = next(row for row in rows if row["lesson"] == CHEAP)
    assumption = ref.Assumption("Every number traces to a graded detail", 5, 3, 1,
                                "Trace one lesson's numbers against its graded details")
    return {
        "large_claimed": large["claimed"], "large_matched": large["matched"],
        "large_rate": large["rate"], "large_verdict": verdict(large["rate"]),
        "cheap_claimed": cheap["claimed"], "cheap_matched": cheap["matched"],
        "cheap_rate": cheap["rate"], "cheap_verdict": verdict(cheap["rate"]),
        "cost": round(cheap["files"] / large["files"], 3),
        **spread(rows, large["rate"]), "draws": len(rows),
        "files": large["files"], "lessons": large["lessons"],
        "test_fields": [name for name in ref.Assumption.__dataclass_fields__
                        if "test" in name],
        "test": assumption.test,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the full run scores 70.8% against the cheap run's 57.9%, both stop",
            all([result["large_claimed"] == 154, result["large_matched"] == 109,
                 result["large_rate"] == 0.708, result["cheap_rate"] == 0.579,
                 result["large_verdict"] == result["cheap_verdict"] == "stop",
                 result["cost"] == 0.2]),
            f"five lessons claim {result['large_claimed']} numbers and match "
            f"{result['large_matched']} ({result['large_rate']:.1%}); one lesson claims "
            f"{result['cheap_claimed']} and matches {result['cheap_matched']} "
            f"({result['cheap_rate']:.1%}). Both verdicts are "
            f"{result['cheap_verdict']!r} at {result['cost']:.1%} of the cost",
        ),
        practice.Check(
            "FINDING: the cheap test is decisive in all 5 draws available",
            all([result["rates"] == [0.579, 0.786, 0.704, 0.862, 0.656],
                 result["agreeing"] == 5, result["draws"] == 5,
                 result["disagreeing"] == []]),
            f"the per-lesson rates are {result['rates']}, every one of them below the "
            f"threshold, so {result['agreeing']} of {result['draws']} draws agree with the "
            f"full run and {len(result['disagreeing'])} would have reversed it -- this "
            "time the cheap test is decisive whichever lesson it lands on",
        ),
        practice.Check(
            "FINDING: the expensive run is not five times more informative",
            all([result["files"] == 25, result["lessons"] == 5,
                 result["spread"] == 0.283]),
            f"{result['lessons']} lessons and {result['files']} graded files move the "
            f"estimate from 57.9% to {result['large_rate']:.1%}, both on the same side of "
            f"the threshold; what they buy is the {result['spread']:.1%} spread that says "
            "the metric is noisy",
        ),
        practice.Check(
            "FINDING: the assumption cannot record which experiment was run",
            all([result["test_fields"] == ["test"],
                 result["test"].startswith("Trace one lesson")]),
            f"Assumption carries {result['test_fields']} -- one string -- so substituting "
            "the cheap test overwrites the description and leaves no trace that a larger "
            "one was replaced",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
