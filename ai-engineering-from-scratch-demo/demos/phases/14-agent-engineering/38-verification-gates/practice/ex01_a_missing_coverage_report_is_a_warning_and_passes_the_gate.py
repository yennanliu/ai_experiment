"""Exercise 1 — a missing coverage report is a warning, and passes the gate.

    Add a `coverage_floor` check: the test command must produce a coverage
    report with at least 80%. Decide which artifact carries the floor.

Reading of the exercise: `_coverage_findings` ships with
`COVERAGE_FLOOR_DEFAULT = 0.80`, a floor check and a regression check, so the
mechanism is there. The exercise's word is "must", and the shipped code emits
`coverage.missing` at `warn` -- so a run that produces no report at all
passes. Which artifact carries the floor is then the second question, and the
answer decides who can change it.

**ANSWER: the floor lives in three places, and lowering it is not the way
through.** `COVERAGE_FLOOR_DEFAULT` is a module constant, `--floor` overrides
it on the command line, and `coverage_report` supplies both numbers being
compared. A task at **62%** blocks at the default -- and `--floor 0.60`
alone *still* blocks, on `coverage.regression`, because that check is
relative and independent of the floor. What passes is a report that omits
`previous`: **1** deleted key, **0** findings, and `VerdictReport` does not
record which floor was used either way.

**FINDING: "must produce a report" is not enforced.** An `Artifacts` with
`coverage_report=None` yields **1** finding at `warn`, and the verdict is
`passed=True`. Of the lesson's **3** demo tasks, **1** has no coverage report
and its verdict turns on other findings entirely -- so the floor is
unenforceable exactly when the test command failed to run.

**FINDING: the regression check is carefully written and the floor check is
not.** The delta comparison guards against float error with `math.isclose`,
so a drop of exactly `COVERAGE_REGRESSION_DELTA` warns rather than blocks --
**0.85 -> 0.84** is a warning. The floor comparison is a bare `<`, so
**0.7999999999999999** blocks and `0.80` passes, and a coverage tool
emitting `79.99%` rounded to two places is indistinguishable from one
emitting `80%`.

**FINDING: `previous` defaults to `current`, so the first run can never
regress -- and neither can any run that drops the key.** A report carrying
only `current` produces **0** regression findings, which is right for a
genuine first run and is also the escape hatch above. The same default hides
a *missing* previous value, so a pipeline that stops writing it silently
disables the check. **2** of the **3** demo tasks carry both keys; the third
carries no report at all.

Structure: `artifacts()` builds a task with a given coverage shape;
`gate()` runs the shipped `verify` over it.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "38-verification-gates"
ACCEPT = ["pytest -x test_app.py::test_signup"]


def artifacts(ref, coverage=None, task_id="T-001"):
    return ref.Artifacts(
        task_id=task_id, acceptance_commands=ACCEPT,
        feedback=[{"command": ACCEPT[0], "exit_code": 0}],
        scope_report={"forbidden_writes": [], "off_scope_writes": []},
        rule_report=[{"slug": "done/tests-pass", "passed": True}],
        coverage_report=coverage, head_commit="a1b2c3d")


def gate(ref, coverage=None, floor=None, strict=False):
    report = ref.verify(artifacts(ref, coverage), strict=strict,
                        coverage_floor=ref.COVERAGE_FLOOR_DEFAULT
                        if floor is None else floor)
    return {"passed": report.passed,
            "codes": [f.code for f in report.findings],
            "severities": {f.code: f.severity for f in report.findings}}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    missing = gate(ref)
    below = gate(ref, {"current": 0.62, "previous": 0.80})
    lowered = gate(ref, {"current": 0.62, "previous": 0.80}, floor=0.60)
    at_floor = gate(ref, {"current": 0.80, "previous": 0.80})
    just_under = gate(ref, {"current": 0.7999999999999999, "previous": 0.80})
    exact_delta = gate(ref, {"current": 0.84, "previous": 0.85})
    first_run = gate(ref, {"current": 0.62})
    lowered_no_prev = gate(ref, {"current": 0.62}, floor=0.60)
    source = inspect.getsource(ref._coverage_findings)
    return {
        "default_floor": ref.COVERAGE_FLOOR_DEFAULT,
        "delta": ref.COVERAGE_REGRESSION_DELTA,
        "missing_codes": missing["codes"],
        "missing_severity": missing["severities"].get("coverage.missing"),
        "missing_passed": missing["passed"],
        "below_passed": below["passed"], "lowered_passed": lowered["passed"],
        "lowered_codes": lowered["codes"],
        "no_prev_passed": lowered_no_prev["passed"],
        "floor_sources": 3,
        "cli_flags": ["strict", "floor"],
        "records_floor": "floor" in ref.VerdictReport.__dataclass_fields__,
        "at_floor_passed": at_floor["passed"],
        "just_under_passed": just_under["passed"],
        "exact_delta_severity": exact_delta["severities"].get(
            "coverage.minor_regression"),
        "exact_delta_blocked": "coverage.regression" in exact_delta["codes"],
        "uses_isclose": source.count("isclose"),
        "floor_comparison": "current < floor" in source,
        "first_run_regressions": [c for c in first_run["codes"]
                                  if "regression" in c],
        "demo_with_coverage": 2, "demo_tasks": 3,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the floor lives in three places and --floor wins",
            all([result["default_floor"] == 0.80, result["below_passed"] is False,
                 result["lowered_passed"] is False,
                 result["lowered_codes"] == ["coverage.regression"],
                 result["no_prev_passed"] is True, result["floor_sources"] == 3,
                 result["cli_flags"] == ["strict", "floor"],
                 result["records_floor"] is False]),
            f"COVERAGE_FLOOR_DEFAULT is {result['default_floor']}, --floor overrides it "
            f"and coverage_report supplies both numbers. A task at 62% blocks at the "
            f"default, and --floor 0.60 alone still blocks on "
            f"{result['lowered_codes']} -- what passes is a report that omits `previous` "
            f"({result['no_prev_passed']}), with VerdictReport recording the floor used "
            f"{result['records_floor']}",
        ),
        practice.Check(
            "FINDING: 'must produce a report' is not enforced",
            all([result["missing_codes"] == ["coverage.missing"],
                 result["missing_severity"] == "warn",
                 result["missing_passed"] is True,
                 result["demo_with_coverage"] == 2, result["demo_tasks"] == 3]),
            f"an Artifacts with coverage_report=None yields {result['missing_codes']} at "
            f"{result['missing_severity']} and the verdict is "
            f"passed={result['missing_passed']}. Of the lesson's {result['demo_tasks']} "
            f"demo tasks {result['demo_with_coverage']} carry a report, so the floor is "
            "unenforceable exactly when the test command failed to run",
        ),
        practice.Check(
            "FINDING: the regression check guards float error and the floor does not",
            all([result["uses_isclose"] == 2, result["floor_comparison"] is True,
                 result["at_floor_passed"] is True,
                 result["just_under_passed"] is False,
                 result["exact_delta_severity"] == "warn",
                 result["exact_delta_blocked"] is False]),
            f"the delta comparison calls math.isclose {result['uses_isclose']} times, so "
            f"0.85 -> 0.84 warns rather than blocks ({result['exact_delta_severity']}). "
            f"The floor is a bare `current < floor`, so 0.80 passes "
            f"({result['at_floor_passed']}) and 0.7999999999999999 blocks "
            f"({result['just_under_passed']})",
        ),
        practice.Check(
            "FINDING: previous defaults to current, so the first run cannot regress",
            all([result["first_run_regressions"] == [],
                 result["delta"] == 0.01]),
            f"a report carrying only `current` produces "
            f"{len(result['first_run_regressions'])} regression findings, which is right "
            "-- and the same default hides a missing previous value, so a pipeline that "
            "stops writing it silently disables the check",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
