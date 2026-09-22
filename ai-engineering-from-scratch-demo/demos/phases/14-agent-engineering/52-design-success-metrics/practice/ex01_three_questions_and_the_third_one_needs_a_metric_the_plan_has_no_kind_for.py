"""Exercise 1 — three questions, and the third needs a metric the plan has no kind for.

    Derive three questions from one outcome goal.

Reading of the exercise: the goal is the one this repository has been working
to -- a reader can run an answer to every exercise and see it check its own
claims against the lesson's code. Three questions fall out of it, and the
third is about somebody other than the reader.

**ANSWER: the three questions map to an outcome metric, a guardrail and a
counter-metric, and the plan validates with 2 of the 3.** "Does every shipped
answer still pass?" is **45** of **45** across nine finished lessons. "Does
any answer make a claim its own run does not support?" is the traced-number
rate. "What does this cost the reader?" is the length of the answers section
-- mean **132.7** lines -- and `validate` has no `counter` kind to hold it.

**FINDING: `validate` requires exactly two kinds and ignores the third the
docs name.** It refuses a plan with no `outcome` and no `guardrail` metric;
`kind` is otherwise a free string, so a metric declared `counter` satisfies
neither requirement and raises no issue. **2** of the **3** kinds the lesson
describes are enforced and the third is invisible.

**FINDING: the plan is valid before any value exists.** `validate` reads the
goal, the questions and the metric contracts; `report` adds per-metric
results. A plan whose every value is missing still reports status `valid`
with **3** results marked `missing`, which is correct and worth saying out
loud: validity is a property of the plan, never of the measurement.

**FINDING: the questions are strings and nothing connects them to metrics.**
`MeasurementPlan` holds **3** fields -- goal, questions, metrics -- so a plan
with **3** questions and **1** metric passes, and so does one with **0**
questions answered by **3** metrics. The derivation the lesson is teaching
leaves no trace in the artifact that records it.

Structure: `QUESTIONS` is the derivation; `measured()` reads the real values
out of the repository.
"""

from __future__ import annotations

import inspect
import statistics
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "52-design-success-metrics"
BASE = Path(__file__).resolve().parents[2]
FINISHED = tuple(f"{number}-" for number in range(43, 52))
GOAL = ("a reader can run an answer to every exercise and see it check its own claims "
        "against the lesson's code")
QUESTIONS = ["Does every shipped answer still pass?",
             "Does any answer make a claim its own run does not support?",
             "What does an answers section cost the reader?"]


def lessons():
    return sorted(path for path in BASE.iterdir()
                  if path.is_dir() and path.name.startswith(FINISHED))


def measured():
    """Values read from the finished lessons rather than invented."""
    files = [path for lesson in lessons()
             for path in sorted((lesson / "practice").glob("ex0*.py"))]
    passing = sum(practice.grade_file(path).status == "pass" for path in files)
    answer_lines = []
    for lesson in lessons():
        readme = (lesson / "practice" / "README.md").read_text(encoding="utf-8")
        answer_lines.append(len(readme.split("## Answers", 1)[1].splitlines()))
    return {"files": len(files), "passing": passing,
            "answer_lines": round(statistics.mean(answer_lines), 1),
            "lessons": len(answer_lines)}


def plan(ref, kinds=("outcome", "outcome", "counter")):
    return ref.MeasurementPlan(
        goal=GOAL, questions=list(QUESTIONS),
        metrics=[
            ref.Metric("shipped_answers_passing", "at-least", 1.0,
                       "every finished lesson", "practice.grade_file", kinds[0]),
            ref.Metric("traced_number_rate", "at-least", 0.9,
                       "one finished lesson", "graded check details", kinds[1]),
            ref.Metric("answer_section_lines", "at-most", 150,
                       "every finished lesson", "practice/README.md", kinds[2]),
        ])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    values = measured()
    counter_only = ref.validate(plan(ref))
    with_guardrail = plan(ref, kinds=("outcome", "guardrail", "counter"))
    document = ref.report(with_guardrail, {})
    validator = inspect.getsource(ref.validate)
    return {
        **values,
        "questions": len(QUESTIONS),
        "kinds_enforced": validator.count('not in kinds'),
        "counter_issues": counter_only,
        "counter_named": "counter" in validator,
        "guardrail_issues": ref.validate(with_guardrail),
        "status_without_values": document["status"],
        "missing": sum(row.get("status") == "missing" for row in document["results"]),
        "plan_fields": list(ref.MeasurementPlan.__dataclass_fields__),
        "mismatched_ok": ref.validate(ref.MeasurementPlan(
            GOAL, [], with_guardrail.metrics)) == ["questions are empty"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: three questions, an outcome, a guardrail and a counter-metric",
            all([result["questions"] == 3, result["files"] == 45,
                 result["passing"] == 45, result["lessons"] == 9,
                 result["answer_lines"] == 132.7]),
            f"across {result['lessons']} finished lessons, {result['passing']} of "
            f"{result['files']} shipped answers pass and the answers sections average "
            f"{result['answer_lines']} lines -- the third question's metric, which the "
            "plan has no kind for",
        ),
        practice.Check(
            "FINDING: validate requires two kinds and ignores the third",
            all([result["kinds_enforced"] == 2, result["counter_named"] is False,
                 result["counter_issues"] == ["guardrail metric is missing"],
                 result["guardrail_issues"] == []]),
            f"the validator enforces {result['kinds_enforced']} kinds and never mentions "
            f"'counter'; a plan whose third metric is a counter-metric reports "
            f"{result['counter_issues']}, and relabelling one metric as a guardrail makes "
            "the same plan valid",
        ),
        practice.Check(
            "FINDING: the plan is valid before any value exists",
            all([result["status_without_values"] == "valid", result["missing"] == 3]),
            f"reporting with no values at all returns status "
            f"{result['status_without_values']!r} with {result['missing']} results marked "
            "missing: validity is a property of the plan, never of the measurement",
        ),
        practice.Check(
            "FINDING: the questions are strings and nothing connects them to metrics",
            all([result["plan_fields"] == ["goal", "questions", "metrics"],
                 result["mismatched_ok"] is True]),
            f"MeasurementPlan holds {result['plan_fields']}, so the only rule about "
            "questions is that the list is non-empty; the derivation the lesson teaches "
            "leaves no trace in the artifact that records it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
