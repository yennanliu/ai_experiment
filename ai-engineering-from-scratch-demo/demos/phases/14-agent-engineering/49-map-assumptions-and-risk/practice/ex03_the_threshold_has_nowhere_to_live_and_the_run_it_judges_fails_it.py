"""Exercise 3 — the threshold has nowhere to live, and the run it judges fails it.

    Define a threshold that would cause you to stop the build.

Reading of the exercise: a stop threshold has to be written before the result
and compared against it afterwards. `Assumption` has **6** fields and none of
them holds a number, so the threshold is defined here and the comparison is
done here -- which is the finding as much as the answer.

**ANSWER: "stop unless 90% of a claimed number traces to a graded check
detail", and the real run scores 70.8%.** Extracting every number from the
docstrings of five finished lessons' solutions gives **154** values, **109**
of which appear verbatim in the details those same solutions print. That is
**70.8%**, below the threshold, so the rule says stop.

**FINDING: the 45 misses are quoted material, which is what a cheap threshold
buys.** They are line numbers inside receipts (`code/main.py:174`), field
counts quoted from a dataclass, and thresholds the prose explains rather than
measures -- not claims about anything the run produced. The threshold fired
correctly and the conclusion is to fix the metric, not the feature: a rule
that stops the build on its first run has told you something either way.

**FINDING: the module cannot record a threshold, a result, or a decision.**
`Assumption` carries a `test` string and an `evidence` string; there is no
field for the number to beat, the number observed, or what to do at pass,
fail and ambiguous. Writing "87.5% of 16" into `evidence` marks the
assumption `tested` and loses the comparison that made it meaningful.

**FINDING: a stop rule only works if the assumption it guards is still open.**
`next_experiment` draws from assumptions with empty evidence, so recording
the failing result removes this assumption from the pool -- **1** fewer
candidate -- and the map moves on to the next experiment as though the
question were settled. Stopping is not a state the document can hold.

Structure: `traced()` reads one lesson's claims; `trace()` pools five of them;
`decide()` applies the threshold.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "49-map-assumptions-and-risk"
BASE = Path(__file__).resolve().parents[2]
SAMPLE = ("43-frame-the-task-before-code", "44-plan-from-evidence",
          "45-delegate-with-isolation", "46-turn-feedback-into-system",
          "47-outcomes-before-output")
THRESHOLD = 0.90
STATEMENT = "Every number in an answer can be sourced from a graded check detail"


def traced(lesson):
    """Every number a lesson's solutions claim in their docstrings, against the
    details those same solutions print when they run."""
    rows = []
    for path in sorted((BASE / lesson / "practice").glob("ex0*.py")):
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
        claimed = sorted(set(re.findall(r"\b\d+(?:\.\d+)?%?\b", doc)))
        printed = " ".join(check.detail for check in practice.grade_file(path).checks)
        rows += [(lesson[:2], value, value in printed) for value in claimed]
    return rows


def trace(lessons=SAMPLE):
    """Every number the five lessons claim, against the details they print."""
    rows = [row for lesson in lessons for row in traced(lesson)]
    matched = [row for row in rows if row[2]]
    return {"claimed": rows, "matched": matched,
            "missing": [(row[0], row[1]) for row in rows if not row[2]],
            "rate": round(len(matched) / len(rows), 3)}


def decide(rate, threshold=THRESHOLD):
    return "continue" if rate >= threshold else "stop"


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    row = trace()
    assumption = ref.Assumption(STATEMENT, 5, 3, 1, "Match the numbers against the details")
    recorded = ref.Assumption(STATEMENT, 5, 3, 1, assumption.test,
                              f"{row['rate']:.1%} of {len(row['claimed'])} numbers")
    pool = [assumption, ref.Assumption("The generator stays correct", 3, 5, 2, "replay")]
    after = [recorded, pool[1]]
    return {
        "claimed": len(row["claimed"]), "matched": len(row["matched"]),
        "rate": row["rate"], "missing": row["missing"],
        "threshold": THRESHOLD, "decision": decide(row["rate"]),
        "fields": list(ref.Assumption.__dataclass_fields__),
        "numeric_fields": [name for name, field in ref.Assumption.__dataclass_fields__.items()
                           if field.type == "int"],
        "threshold_field": any(name in ref.Assumption.__dataclass_fields__
                               for name in ("threshold", "stop_at", "decision")),
        "status_after": ref.prioritize(after)[0]["status"] if
        ref.prioritize(after)[0]["statement"] == STATEMENT else
        next(row2["status"] for row2 in ref.prioritize(after)
             if row2["statement"] == STATEMENT),
        "open_before": len([item for item in pool if not item.evidence]),
        "open_after": len([item for item in after if not item.evidence]),
        "next_after": ref.next_experiment(after).statement,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the threshold is 90% and the real run scores 70.8%",
            all([result["claimed"] == 154, result["matched"] == 109,
                 result["rate"] == 0.708, result["threshold"] == 0.90,
                 result["decision"] == "stop"]),
            f"five finished lessons' solutions claim {result['claimed']} numbers in their "
            f"docstrings and {result['matched']} appear verbatim in the details those same "
            "solutions print -- "
            f"{result['rate']:.1%} against a {result['threshold']:.0%} threshold, so the "
            f"rule says {result['decision']}",
        ),
        practice.Check(
            "FINDING: the 45 misses are quoted material, not measurements",
            all([len(result["missing"]) == 45,
                 ("43", "174") in result["missing"]]),
            f"the {len(result['missing'])} unmatched values are line numbers inside "
            "receipts, field counts quoted from a dataclass and thresholds the prose "
            "explains -- not claims about anything the run measured, so the threshold "
            "fired correctly and the fix belongs to the metric",
        ),
        practice.Check(
            "FINDING: the module cannot record a threshold, a result, or a decision",
            all([len(result["fields"]) == 6, result["threshold_field"] is False,
                 len(result["numeric_fields"]) == 3]),
            f"Assumption carries {result['fields']}; its {len(result['numeric_fields'])} "
            "numeric fields are the risk dimensions, and there is no field for the number "
            "to beat, the number observed, or what to do at pass, fail and ambiguous",
        ),
        practice.Check(
            "FINDING: a stop rule only works if the assumption it guards is still open",
            all([result["status_after"] == "tested", result["open_before"] == 2,
                 result["open_after"] == 1,
                 result["next_after"] == "The generator stays correct"]),
            f"recording the failing result marks the assumption {result['status_after']!r} "
            f"and drops the open pool from {result['open_before']} to "
            f"{result['open_after']}, so the map moves on to "
            f"{result['next_after']!r} as though the question were settled",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
