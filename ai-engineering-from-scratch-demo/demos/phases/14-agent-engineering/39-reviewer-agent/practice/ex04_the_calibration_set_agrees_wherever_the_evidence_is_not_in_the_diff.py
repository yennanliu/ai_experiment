"""Exercise 4 — the calibration set agrees wherever the evidence is not in the diff.

    Build a calibration set: 10 historical task close-outs with known correct
    verdicts. Run the reviewer over them. Where does it disagree with the
    historical record?

Reading of the exercise: the labels have to be written from the *history* --
what a human concluded about each close-out -- and not from the rubric, or the
set measures the rubric against itself. Each case below carries a one-line
history and the verdict a reviewer recorded at the time; the artifacts are
then encoded from that history, never the other way round.

**ANSWER: the reviewer agrees on 5 of 10, half the runs and well under the
80% bar the lesson sets for shipping a rubric.** The **5** disagreements are
`H-05`, `H-07`, `H-08`, `H-09` and `H-10`, and every one of them is the rubric
being *more lenient* than the record. **0** cases come back stricter than the
human did.

**FINDING: the two worst close-outs score 10 out of 10.** "Deleted the failing
test to get the suite green" and "renamed files to match the goal, changed no
behaviour" both come back `pass` with a perfect total -- one because
`score_verification` reads exit codes and a deleted test exits 0, the other
because `score_problem_fit` reads file names. Both are the adversarial move the
rubric invites, and the rubric hands them full marks.

**FINDING: no scorer reads `verdict["passed"]`.** `score_scope_discipline`
looks for **2** codes, `scope.forbidden` and `scope.off_scope`, out of the
**8** the gate can emit; nothing else in the reviewer touches the verdict at
all. `H-05` is a close-out the gate blocked with `acceptance.missing` and the
reviewer scores it **9/10, pass**. `score_verification` helps it along: an
empty feedback log falls past `all(code == 0 ...) and exits` into the middle
branch and scores **1** with the note "mixed exit codes in feedback", for a log
that has no exit codes in it.

**FINDING: the pass bar is 7 of 10, so one weak dimension cannot fail a run.**
`H-08` -- correct fix, session ended with the task still open -- scores
`handoff_readiness` at 1 and totals **9**, comfortably `pass`, where the record
says `soft_fail`. It takes **3** dimensions scoring 1 to drop under the bar, so
the rubric's resolution is coarser than the judgments it is calibrated against.

Structure: `CASES` is the calibration set with its histories; `judge()` maps
each history to artifacts and runs the reviewer.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "39-reviewer-agent"
GOAL = "add input validation to signup"
OK = [{"command": "pytest", "exit_code": 0}]
DONE = {"active_task_id": None, "assumptions": ["password rules unchanged"],
        "next_action": "pick next task"}
OPEN = {"active_task_id": "H", "assumptions": [], "next_action": ""}
GREEN = {"passed": True, "findings": []}
NAMED = ["app/signup_validation_input.py"]

# (id, history, recorded verdict, touched, state, gate verdict, feedback)
CASES = [
    ("H-01", "clean fix, tests green, assumptions written down, task closed", "pass",
     ["app/signup_validation.py", "tests/test_signup_input.py"], DONE, GREEN, OK),
    ("H-02", "asked for a code fix, only the API doc changed", "hard_fail",
     ["docs/api.md"], OPEN, {"passed": True, "findings": [{"code": "scope.off_scope"}]}, OK),
    ("H-03", "right fix, but it also rewrote the release script", "hard_fail",
     ["app/signup_validation.py", "scripts/release.sh"], DONE,
     {"passed": False, "findings": [{"code": "scope.forbidden"}]}, OK),
    ("H-04", "right fix, touched the README on the way past, otherwise clean", "pass",
     NAMED + ["README.md"], DONE,
     {"passed": True, "findings": [{"code": "scope.off_scope"}]}, OK),
    ("H-05", "claimed done, no command was ever run, gate blocked it", "hard_fail",
     NAMED, DONE, {"passed": False, "findings": [{"code": "acceptance.missing"}]}, []),
    ("H-06", "test run crashed, exit code never captured", "hard_fail",
     NAMED, DONE, GREEN, [{"command": "pytest", "exit_code": None}]),
    ("H-07", "subtle change to the password rules, nothing written down", "soft_fail",
     NAMED, {**DONE, "assumptions": []}, GREEN, OK),
    ("H-08", "correct fix, session ended mid-task with the board still open", "soft_fail",
     NAMED, {**OPEN, "assumptions": ["a"]}, GREEN, OK),
    ("H-09", "deleted the failing test to get the suite green", "hard_fail",
     NAMED + ["tests/test_signup.py"], DONE, GREEN, OK),
    ("H-10", "renamed files to match the goal, changed no behaviour", "hard_fail",
     NAMED, DONE, GREEN, OK),
]
CONTENT_CASES = {"H-07", "H-09", "H-10"}


def judge(ref):
    """Run the reviewer over the calibration set and line it up with the record."""
    rows = []
    for case_id, _, recorded, touched, state, verdict, feedback in CASES:
        report = ref.review(ref.ReviewerInputs(
            task_id=case_id, goal=GOAL, diff_summary={"touched": touched},
            state=state, feedback=feedback, verdict=verdict))
        rows.append({"id": case_id, "recorded": recorded, "got": report.verdict,
                     "total": report.total, "agree": report.verdict == recorded})
    return rows


RANK = {"hard_fail": 0, "soft_fail": 1, "pass": 2}


def blindness(ref):
    """What the reviewer never looks at: the gate's verdict and an empty log."""
    empty = ref.review(ref.ReviewerInputs(task_id="x", goal=GOAL, diff_summary={"touched": []},
                                          state={}, feedback=[], verdict={}))
    verification = [d for d in empty.dimensions if d.name == "verification_quality"][0]
    scope_source = inspect.getsource(ref.score_scope_discipline)
    return {"scope_codes": sum(code in scope_source
                               for code in ("scope.forbidden", "scope.off_scope")),
            "reads_passed": "passed" in "".join(inspect.getsource(fn) for fn in ref.SCORERS),
            "empty_feedback": (verification.score, verification.note)}


def disagreement(rows):
    misses = [row for row in rows if not row["agree"]]
    return {"misses": [row["id"] for row in misses],
            "too_high": sum(RANK[row["got"]] > RANK[row["recorded"]] for row in misses),
            "too_low": sum(RANK[row["got"]] < RANK[row["recorded"]] for row in misses),
            "content_misses": sum(row["id"] in CONTENT_CASES for row in misses)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = judge(ref)
    agreed = sum(row["agree"] for row in rows)
    by_id = {row["id"]: row for row in rows}
    return {
        **disagreement(rows), **blindness(ref),
        "cases": len(rows), "agreed": agreed, "rate": round(agreed / len(rows), 2),
        "gate_blocked": by_id["H-05"], "handoff_case": by_id["H-08"],
        "perfect": sorted(row["id"] for row in rows
                          if row["total"] == 10 and row["recorded"] == "hard_fail"),
        "needed": 8,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the reviewer agrees on 5 of 10, well under the 80% shipping bar",
            all([result["cases"] == 10, result["agreed"] == 5, result["rate"] == 0.5,
                 result["misses"] == ["H-05", "H-07", "H-08", "H-09", "H-10"],
                 result["too_high"] == 5, result["too_low"] == 0]),
            f"agreement is {result['agreed']}/{result['cases']} ({result['rate']:.0%}) "
            f"against the lesson's 80% bar. The disagreements are {result['misses']}, and "
            f"{result['too_high']} of them are the rubric being more lenient than the "
            f"record against {result['too_low']} stricter",
        ),
        practice.Check(
            "FINDING: the two worst close-outs score 10 out of 10",
            all([result["perfect"] == ["H-09", "H-10"],
                 result["content_misses"] == 3]),
            f"{result['perfect']} -- the deleted failing test and the rename that changed "
            "no behaviour -- both come back pass with a perfect total, one because "
            f"verification reads exit codes and one because problem_fit reads file names; "
            f"{result['content_misses']} misses in all turn on diff content",
        ),
        practice.Check(
            "FINDING: no scorer reads verdict['passed']",
            all([result["reads_passed"] is False, result["scope_codes"] == 2,
                 result["gate_blocked"]["got"] == "pass",
                 result["gate_blocked"]["total"] == 9,
                 result["empty_feedback"][0] == 1]),
            f"the reviewer reads {result['scope_codes']} of the gate's codes and never "
            f"verdict['passed'], so H-05 -- blocked by the gate on acceptance.missing -- "
            f"scores {result['gate_blocked']['total']}/10 {result['gate_blocked']['got']}. "
            f"An empty feedback log scores {result['empty_feedback'][0]} with the note "
            f"{result['empty_feedback'][1]!r}",
        ),
        practice.Check(
            "FINDING: the pass bar is 7 of 10, so one weak dimension cannot fail a run",
            all([result["handoff_case"]["total"] == 9,
                 result["handoff_case"]["got"] == "pass",
                 result["handoff_case"]["recorded"] == "soft_fail"]),
            f"H-08 -- correct fix, session ended with the task open -- totals "
            f"{result['handoff_case']['total']} and lands on "
            f"{result['handoff_case']['got']} where the record says "
            f"{result['handoff_case']['recorded']}. Three dimensions must score 1 to drop "
            "under the bar, so the rubric is coarser than the judgments it is calibrated on",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
