"""Exercise 2 — the counter-metric shows ten of forty-five files over the line target.

    Add a counter-metric that catches cost shifted to another role.

Reading of the exercise: an outcome metric here measures the reader -- every
shipped answer passes. The cost of reaching it lands on two other people: the
maintainer who has to keep each file inside the repository's own size rule,
and the reviewer who has to read the answers section. Both are measurable
from the tree.

**ANSWER: 10 of 45 shipped files sit above the 120-line target, and the
outcome metric cannot see any of them.** The answers sections average
**133.3** lines each. Both numbers come from the same nine lessons whose
answers pass **45** of **45**, so a plan carrying only the outcome metric
reports a perfect result while the cost it created is entirely outside the
report.

**FINDING: the counter-metric has to be `at-most` on a number that improving
the outcome pushes up.** Longer answers make a claim easier to support, so
the metric that protects against them is `answer_section_lines at-most 150`;
measured, the worst lesson sits at **146** and passes by **4** lines. A
counter-metric with slack is still a counter-metric -- it is the one that
will fail first.

**FINDING: `validate` accepts the counter-metric and counts it as neither
kind.** Adding it to a plan that already has an outcome and a guardrail
leaves **0** issues, and the plan now holds **4** metrics of which **3**
kinds are distinct. The lesson's third category exists only in the value of a
string.

**FINDING: the shifted cost is invisible in the pass/fail line.** `report`
returns one row per metric with `passed` per row and no aggregate, so a run
where the outcome passes and the counter-metric fails renders as **3**
passing rows and **1** failing row with no verdict. The reader has to decide
what a mixed report means, which is what Exercise 4 is for.

Structure: `code_lines()` reproduces the repository's own size rule;
`counter()` measures what the outcome metric costs.
"""

from __future__ import annotations

import ast
import statistics
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "52-design-success-metrics"
BASE = Path(__file__).resolve().parents[2]
FINISHED = tuple(f"{number}-" for number in range(43, 52))
TARGET = 120
LINE_BUDGET = 150


def lessons():
    return sorted(path for path in BASE.iterdir()
                  if path.is_dir() and path.name.startswith(FINISHED))


def code_lines(path):
    """The repository's own metric: every line except the module docstring."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    first = tree.body[0] if tree.body else None
    doc = (first.end_lineno - first.lineno + 1
           if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
           and isinstance(first.value.value, str) else 0)
    return len(text.splitlines()) - doc


def counter():
    files = [path for lesson in lessons()
             for path in sorted((lesson / "practice").glob("ex0*.py"))]
    sizes = [code_lines(path) for path in files]
    answers = [len((lesson / "practice" / "README.md").read_text(encoding="utf-8")
                   .split("## Answers", 1)[1].splitlines()) for lesson in lessons()]
    return {"files": len(files), "over": sum(size > TARGET for size in sizes),
            "largest": max(sizes), "answer_mean": round(statistics.mean(answers), 1),
            "answer_max": max(answers)}


def plan(ref, with_counter=True):
    metrics = [
        ref.Metric("shipped_answers_passing", "at-least", 1.0, "every finished lesson",
                   "practice.grade_file", "outcome"),
        ref.Metric("files_over_line_target", "at-most", 0, "every finished lesson",
                   "ast line count", "guardrail"),
    ]
    if with_counter:
        metrics.append(ref.Metric("answer_section_lines", "at-most", LINE_BUDGET,
                                  "every finished lesson", "practice/README.md", "counter"))
    return ref.MeasurementPlan("every shipped answer checks its own claims",
                               ["Does every answer pass?", "What does it cost to keep them?"],
                               metrics)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    measured = counter()
    full = plan(ref)
    values = {"shipped_answers_passing": 1.0,
              "files_over_line_target": measured["over"],
              "answer_section_lines": measured["answer_max"]}
    document = ref.report(full, values)
    outcome_only = ref.report(plan(ref, with_counter=False),
                              {"shipped_answers_passing": 1.0,
                               "files_over_line_target": 0})
    return {
        **measured,
        "issues": ref.validate(full), "kinds": len({m.kind for m in full.metrics}),
        "metrics": len(full.metrics),
        "counter_direction": full.metrics[-1].direction,
        "budget": LINE_BUDGET, "slack": LINE_BUDGET - measured["answer_max"],
        "rows": len(document["results"]),
        "passing_rows": sum(row.get("passed") is True for row in document["results"]),
        "failing_rows": sum(row.get("passed") is False for row in document["results"]),
        "keys": sorted(document),
        "aggregate": any(key in document for key in ("verdict", "decision", "passed")),
        "outcome_only_rows": len(outcome_only["results"]),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 10 of 45 files sit above the line target the outcome cannot see",
            all([result["files"] == 45, result["over"] == 10, result["largest"] == 145,
                 result["answer_mean"] == 133.3, result["outcome_only_rows"] == 2]),
            f"{result['over']} of {result['files']} shipped files exceed the "
            f"{TARGET}-line target, the largest at {result['largest']}, while the answers "
            f"sections average {result['answer_mean']} lines -- none of it visible in a "
            f"plan whose {result['outcome_only_rows']} metrics are the outcome and its "
            "guardrail",
        ),
        practice.Check(
            "FINDING: the counter-metric is at-most on a number the outcome pushes up",
            all([result["counter_direction"] == "at-most", result["budget"] == 150,
                 result["answer_max"] == 146, result["slack"] == 4]),
            f"longer answers make a claim easier to support, so the counter-metric is "
            f"{result['counter_direction']} {result['budget']}; the worst lesson sits at "
            f"{result['answer_max']} and passes by {result['slack']} lines",
        ),
        practice.Check(
            "FINDING: validate accepts the counter-metric and counts it as neither kind",
            all([result["issues"] == [], result["metrics"] == 3, result["kinds"] == 3]),
            f"the plan holds {result['metrics']} metrics across {result['kinds']} distinct "
            f"kinds and validates with {len(result['issues'])} issues: the lesson's third "
            "category exists only in the value of a string",
        ),
        practice.Check(
            "FINDING: the shifted cost is invisible in the pass/fail line",
            all([result["rows"] == 3, result["failing_rows"] == 1,
                 result["passing_rows"] == 2, result["aggregate"] is False,
                 len(result["keys"]) == 4]),
            f"the report renders {result['passing_rows']} passing rows and "
            f"{result['failing_rows']} failing one across {result['keys']} with no "
            "aggregate, so a reader has to decide for themselves what a mixed result means",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
