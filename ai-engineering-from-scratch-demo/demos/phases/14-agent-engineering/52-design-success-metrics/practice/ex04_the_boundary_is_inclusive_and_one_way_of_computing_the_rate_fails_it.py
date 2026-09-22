"""Exercise 4 — the boundary is inclusive, and one way of computing the rate fails it.

    Write pass, fail, and ambiguous decisions before generating values.

Reading of the exercise: three paths written first, then values. The lesson's
own example is the model -- pass on a rate and a time, fail on any production
write, ambiguous when the improvement is small and the variance wide -- and
writing them here exposes what `report` does not decide.

**ANSWER: the three paths are written, and the measured run lands on fail.**
Pass requires every shipped answer to pass and the traced-number rate to reach
0.9; fail is any file over the hard ceiling or a rate below 0.75; ambiguous is
everything between. The real values -- **45** of **45** answers passing, a
traced rate of **0.708**, **0** files over the ceiling -- trip the fail rule
on the rate alone, which is what writing the rule first is for: the run that
looks green on its headline metric is refused on the one that was harder to
satisfy.

**FINDING: `report` decides per metric and never in aggregate.** It returns
**4** keys and one row per metric carrying `passed`; the pass, fail and
ambiguous paths live entirely outside it. On the measured values **2** rows
pass and **1** fails, which is exactly the mixed result a decision rule
exists to interpret.

**FINDING: the thresholds are inclusive, so 0.9 passes an at-least 0.9.**
`evaluate` compares with `<=` and `>=`, and the shipped example's
`correct_service_rate` is 0.9 against a threshold of 0.9 -- a pass by
equality. Every boundary case in a plan written this way lands on the
generous side, which is a choice worth making deliberately rather than
inheriting.

**FINDING: how the rate is computed decides the boundary case.** Nine tenths
expressed as `9 / 10` passes an at-least 0.9; the same nine tenths summed as
three thirds of `0.3` gives **0.8999999999999999** and fails. The decision
rule has to name the arithmetic, not just the number, or the pass path
depends on which line of code produced the value.

Structure: `DECISION` is the rule written first; `measured()` produces the
values afterwards.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "52-design-success-metrics"
BASE = Path(__file__).resolve().parents[2]
FINISHED = tuple(f"{number}-" for number in range(43, 52))
SAMPLE = ("43-frame-the-task-before-code", "44-plan-from-evidence",
          "45-delegate-with-isolation", "46-turn-feedback-into-system",
          "47-outcomes-before-output")

DECISION = {
    "pass": "every shipped answer passes and the traced rate reaches 0.9",
    "fail": "any file over the 150-line ceiling, or a traced rate below 0.75",
    "ambiguous": "anything else, which buys a larger replay set",
}


def decide(passing, files, traced, over_ceiling):
    if over_ceiling or traced < 0.75:
        return "fail"
    if passing == files and traced >= 0.9:
        return "pass"
    return "ambiguous"


def traced_rate(lessons=SAMPLE):
    """Pooled over five lessons: the numbers their solutions claim in docstrings,
    against the details those same solutions print. Reading the solutions rather
    than the prose keeps one README edit from moving the verdict."""
    claimed = matched = 0
    for lesson in lessons:
        for path in sorted((BASE / lesson / "practice").glob("ex0*.py")):
            doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
            values = sorted(set(re.findall(r"\b\d+(?:\.\d+)?%?\b", doc)))
            printed = " ".join(check.detail for check in practice.grade_file(path).checks)
            claimed += len(values)
            matched += sum(value in printed for value in values)
    return round(matched / claimed, 3)


def measured():
    lessons = sorted(path for path in BASE.iterdir()
                     if path.is_dir() and path.name.startswith(FINISHED))
    files = [path for lesson in lessons
             for path in sorted((lesson / "practice").glob("ex0*.py"))]
    return {"files": len(files),
            "passing": sum(practice.grade_file(path).status == "pass" for path in files),
            "traced": traced_rate(), "over_ceiling": 0}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    values = measured()
    plan = ref.MeasurementPlan(
        "every shipped answer checks its own claims",
        ["Does every answer pass?", "Does any claim outrun its run?"],
        [ref.Metric("shipped_answers_passing", "at-least", 1.0, "every finished lesson",
                    "practice.grade_file", "outcome"),
         ref.Metric("traced_number_rate", "at-least", 0.9, "one finished lesson",
                    "graded check details", "outcome"),
         ref.Metric("files_over_ceiling", "at-most", 0, "every finished lesson",
                    "ast line count", "guardrail")])
    document = ref.report(plan, {"shipped_answers_passing": values["passing"] / values["files"],
                                 "traced_number_rate": values["traced"],
                                 "files_over_ceiling": values["over_ceiling"]})
    boundary = ref.Metric("correct_service_rate", "at-least", 0.9, "ten replays",
                          "incident record", "outcome")
    summed = sum(0.3 for _ in range(3))
    return {
        **values, "paths": len(DECISION),
        "decision": decide(values["passing"], values["files"], values["traced"],
                           values["over_ceiling"]),
        "keys": sorted(document), "rows": len(document["results"]),
        "passing_rows": sum(row.get("passed") is True for row in document["results"]),
        "failing_rows": sum(row.get("passed") is False for row in document["results"]),
        "aggregate": any(key in document for key in ("decision", "verdict")),
        "inclusive": inspect.getsource(ref.evaluate).count("<=")
        + inspect.getsource(ref.evaluate).count(">="),
        "equal_passes": ref.evaluate(boundary, 0.9),
        "divided": ref.evaluate(boundary, 9 / 10), "summed": ref.evaluate(boundary, summed),
        "summed_value": summed,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the three paths are written and the measured run lands on fail",
            all([result["paths"] == 3, result["files"] == 45, result["passing"] == 45,
                 result["traced"] == 0.708, result["decision"] == "fail"]),
            f"{result['passing']} of {result['files']} answers pass, the traced rate is "
            f"{result['traced']} and {result['over_ceiling']} files exceed the ceiling, so "
            f"the run trips the fail rule on the rate alone and the decision is "
            f"{result['decision']!r}",
        ),
        practice.Check(
            "FINDING: report decides per metric and never in aggregate",
            all([len(result["keys"]) == 4, result["rows"] == 3,
                 result["passing_rows"] == 2, result["failing_rows"] == 1,
                 result["aggregate"] is False]),
            f"the document returns {result['keys']} with {result['rows']} rows -- "
            f"{result['passing_rows']} passing, {result['failing_rows']} failing -- and no "
            "aggregate, which is exactly the mixed result a decision rule exists to read",
        ),
        practice.Check(
            "FINDING: the thresholds are inclusive, so 0.9 passes an at-least 0.9",
            all([result["inclusive"] == 2, result["equal_passes"] is True]),
            f"evaluate compares with <= and >= ({result['inclusive']} operators), so the "
            "shipped example's rate of 0.9 against a threshold of 0.9 passes by equality "
            "and every boundary case lands on the generous side",
        ),
        practice.Check(
            "FINDING: how the rate is computed decides the boundary case",
            all([result["divided"] is True, result["summed"] is False,
                 result["summed_value"] == 0.8999999999999999]),
            f"nine tenths as 9/10 passes and the same nine tenths summed from 0.3 gives "
            f"{result['summed_value']} and fails; the rule has to name the arithmetic, not "
            "just the number",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
