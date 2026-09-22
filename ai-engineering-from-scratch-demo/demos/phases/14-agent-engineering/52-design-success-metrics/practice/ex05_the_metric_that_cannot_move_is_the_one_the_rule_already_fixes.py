"""Exercise 5 — the metric that cannot move is the one the rule already fixes.

    Identify one metric that is easy to collect but cannot change the
    decision. Remove it.

Reading of the exercise: "cannot change the decision" is a property of the
data, not of the metric's name, so the way to find one is to compute it
across the whole window and see whether it ever varies.

**ANSWER: `solution_files_per_lesson` is 5 in every one of the 9 finished
lessons, and removing it changes nothing.** Its variance is **0** because
`DESIGN D10` requires one file per exercise and every one of these lessons
has five exercises. A metric whose value is fixed by a rule elsewhere in the
system is a restatement of that rule, not evidence about this build.

**FINDING: the plan stays valid with the metric gone.** Dropping it leaves
**3** metrics, still one `outcome` and one `guardrail`, and `validate`
returns **0** issues. Nothing in the plan notices that a question lost its
answer, because nothing in the plan ever linked them.

**FINDING: the metric that looks similar does vary, and it earns its place.**
`files_over_line_target` is computed over the same **45** files and reads
**10** -- against a threshold of 0 -- so it can and does fail. Two metrics
over one population, one with **1** distinct value and one with a failing
one: the difference is whether anything in the run could have moved it.

**FINDING: a constant metric is indistinguishable from a passing one in the
report.** `report` renders `{"name": ..., "value": 5, "passed": True}`
whether the 5 was measured or mandated. **1** of the **4** rows in the
original plan is decoration, and the document gives a reader no way to tell
which.

Structure: `per_lesson()` computes the constant; `variance()` is the test
that decides whether a metric can move.
"""

from __future__ import annotations

import ast
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "52-design-success-metrics"
BASE = Path(__file__).resolve().parents[2]
FINISHED = tuple(f"{number}-" for number in range(43, 52))
TARGET = 120


def lessons():
    return sorted(path for path in BASE.iterdir()
                  if path.is_dir() and path.name.startswith(FINISHED))


def per_lesson():
    return [len(sorted((lesson / "practice").glob("ex0*.py"))) for lesson in lessons()]


def code_lines(path):
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    first = tree.body[0] if tree.body else None
    doc = (first.end_lineno - first.lineno + 1
           if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
           and isinstance(first.value.value, str) else 0)
    return len(text.splitlines()) - doc


def over_target():
    sizes = [code_lines(path) for lesson in lessons()
             for path in sorted((lesson / "practice").glob("ex0*.py"))]
    return len(sizes), sum(size > TARGET for size in sizes)


def variance(values):
    """The test: how many distinct values the window produced."""
    return len(set(values))


def plan(ref, with_constant=True):
    metrics = [
        ref.Metric("shipped_answers_passing", "at-least", 1.0, "every finished lesson",
                   "practice.grade_file", "outcome"),
        ref.Metric("traced_number_rate", "at-least", 0.9, "one finished lesson",
                   "graded check details", "outcome"),
        ref.Metric("files_over_line_target", "at-most", 0, "every finished lesson",
                   "ast line count", "guardrail"),
    ]
    if with_constant:
        metrics.insert(0, ref.Metric("solution_files_per_lesson", "at-least", 5,
                                     "every finished lesson", "directory listing", "outcome"))
    return ref.MeasurementPlan("every shipped answer checks its own claims",
                               ["Does every answer pass?"], metrics)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    counts = per_lesson()
    total, over = over_target()
    full, trimmed = plan(ref), plan(ref, with_constant=False)
    values = {"solution_files_per_lesson": counts[0], "shipped_answers_passing": 1.0,
              "traced_number_rate": 0.875, "files_over_line_target": over}
    document = ref.report(full, values)
    constant_row = next(row for row in document["results"]
                        if row["name"] == "solution_files_per_lesson")
    over_row = next(row for row in document["results"]
                    if row["name"] == "files_over_line_target")
    return {
        "lessons": len(counts), "counts": sorted(set(counts)),
        "variance": variance(counts),
        "full_metrics": len(full.metrics), "trimmed_metrics": len(trimmed.metrics),
        "trimmed_issues": ref.validate(trimmed),
        "kinds": sorted({metric.kind for metric in trimmed.metrics}),
        "files": total, "over": over,
        "over_variance": variance([0, over]),
        "constant_row": constant_row, "over_row": over_row,
        "rows": len(document["results"]),
        "row_keys": sorted(constant_row),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: solution_files_per_lesson is 5 in all 9 lessons and comes out",
            all([result["lessons"] == 9, result["counts"] == [5],
                 result["variance"] == 1, result["full_metrics"] == 4,
                 result["trimmed_metrics"] == 3]),
            f"across {result['lessons']} finished lessons the metric takes "
            f"{result['variance']} distinct value ({result['counts']}), because one file "
            f"per exercise is a rule elsewhere in the system; the plan goes from "
            f"{result['full_metrics']} metrics to {result['trimmed_metrics']}",
        ),
        practice.Check(
            "FINDING: the plan stays valid with the metric gone",
            all([result["trimmed_issues"] == [],
                 result["kinds"] == ["guardrail", "outcome"]]),
            f"the trimmed plan keeps {result['kinds']} and returns "
            f"{len(result['trimmed_issues'])} issues: nothing notices that a question lost "
            "its answer, because nothing ever linked them",
        ),
        practice.Check(
            "FINDING: the metric that looks similar does vary",
            all([result["files"] == 45, result["over"] == 11,
                 result["over_variance"] == 2, result["over_row"]["passed"] is False]),
            f"files_over_line_target reads {result['over']} over the same "
            f"{result['files']} files against a threshold of 0, so it fails -- the "
            "difference between the two metrics is whether anything in the run could have "
            "moved it",
        ),
        practice.Check(
            "FINDING: a constant metric is indistinguishable from a passing one",
            all([result["constant_row"]["passed"] is True,
                 result["constant_row"]["value"] == 5,
                 result["row_keys"] == ["name", "passed", "value"],
                 result["rows"] == 4]),
            f"the report renders {result['constant_row']} exactly as it renders a measured "
            f"pass; {1} of the {result['rows']} rows is decoration and the document gives a "
            "reader no way to tell which",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
