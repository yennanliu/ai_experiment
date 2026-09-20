"""Exercise 3 — a three-point rubric reaches the gate as one boolean.

    Add a rubric dimension that requires human judgment. Calibrate it on five
    examples before using it as a gate.

Reading of the exercise: calibration is a measurement between raters, so the
five examples are scored twice and the agreement is computed before anything
is gated. The disagreement that shows up is on the boundary example, which is
the only kind of example a gate ever has to decide, so calibration that
reports **4 of 5** is calibration that has not yet answered the question.

**ANSWER: score the dimension 0-2, calibrate on five, and gate only after the
boundary rule is written.** Two raters agree exactly on **4** of **5** and
differ on `partial-consequence`, the one example sitting on the threshold.
Adding one sentence -- a named owner and a named action, or it is a 1 --
takes agreement to **5** of **5**, and only then does the dimension become a
gate.

**FINDING: a three-point rubric reaches the gate as one boolean.**
`EvidenceCheck` has **3** fields: an id, a `passed` bool and an evidence
string. The 0-2 score and the two raters have to be collapsed to `True` or
`False` before `evaluate_evidence_checks` ever sees them, which moves the
threshold out of the gate and into whoever fills in the field.

**FINDING: the evidence string is required and never read.** A check whose
evidence is `"x"` validates identically to one carrying a transcript --
`evaluate_evidence_checks` tests that it is a non-empty string and stops.
The field that makes a judgment reviewable is the field nothing reviews.

**FINDING: the gate is `all(passed)`, so there is no room for a mean.** A
rubric averaging **1.0** and one averaging **1.6** arrive as **2** and **3**
passing booleans of five, and the mean itself never arrives. A dimension
that is genuinely continuous has to be discretized by the rater rather than
by policy.

Structure: `agreement()` is the calibration, run before `as_checks()` turns
the scores into the only shape the module accepts.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "27-skill-evals-packaging-and-portability"
DIMENSION = "names a consequence the reader can act on"
EXAMPLES = ("no-consequence", "vague-consequence", "partial-consequence",
            "named-consequence", "owned-consequence")
RATERS = ("rater-a", "rater-b")
RATER_A = {"no-consequence": 0, "vague-consequence": 0, "partial-consequence": 1,
           "named-consequence": 2, "owned-consequence": 2}
RATER_B = {"no-consequence": 0, "vague-consequence": 0, "partial-consequence": 2,
           "named-consequence": 2, "owned-consequence": 2}
RULE = "a 2 needs a named owner and a named action; anything short of both is a 1"
AFTER_RULE = {**RATER_B, "partial-consequence": 1}
PASS_AT = 2


def agreement(first, second):
    exact = [name for name in EXAMPLES if first[name] == second[name]]
    return {"exact": len(exact), "total": len(EXAMPLES),
            "disputed": sorted(set(EXAMPLES) - set(exact))}


def as_checks(ref, scores):
    """The only shape the gate accepts: one bool and one string per dimension."""
    return tuple(ref.EvidenceCheck(name, scores[name] >= PASS_AT,
                                   f"{DIMENSION}: scored {scores[name]}")
                 for name in EXAMPLES)


def mean(scores):
    return round(sum(scores.values()) / len(scores), 4)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    before = agreement(RATER_A, RATER_B)
    after = agreement(RATER_A, AFTER_RULE)
    checks = as_checks(ref, RATER_A)
    report = ref.evaluate_evidence_checks(checks, "rubric")

    thin = ref.evaluate_evidence_checks(
        (ref.EvidenceCheck("thin", True, "x"),), "rubric")
    try:
        ref.evaluate_evidence_checks((ref.EvidenceCheck("blank", True, "  "),), "rubric")
        empty_accepted = True
    except ValueError:
        empty_accepted = False

    generous = {name: min(2, RATER_A[name] + 1) for name in EXAMPLES}
    strict_pass = [case["passed"] for case in report["cases"]]
    generous_pass = [case["passed"] for case in
                     ref.evaluate_evidence_checks(as_checks(ref, generous),
                                                  "rubric")["cases"]]
    return {
        "before": before, "after": after, "rule": RULE,
        "scale": sorted(set(RATER_A.values()) | set(RATER_B.values())),
        "raters": len(RATERS),
        "check_fields": list(vars(ref.EvidenceCheck)["__dataclass_fields__"]),
        "gate_passed": report["passed"], "case_verdicts": strict_pass,
        "thin_passed": thin["passed"], "thin_evidence": thin["cases"][0]["evidence"],
        "empty_accepted": empty_accepted,
        "mean_strict": mean(RATER_A), "mean_generous": mean(generous),
        "generous_verdicts": generous_pass,
        "collapsed": sum(strict_pass), "collapsed_generous": sum(generous_pass),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: score 0-2, calibrate on five, and gate only after the boundary rule",
            all([result["before"]["exact"] == 4, result["before"]["total"] == 5,
                 result["before"]["disputed"] == ["partial-consequence"],
                 result["after"]["exact"] == 5, result["raters"] == 2,
                 result["scale"] == [0, 1, 2]]),
            f"two raters agree exactly on {result['before']['exact']} of "
            f"{result['before']['total']} and differ on {result['before']['disputed']} -- "
            f"the one example on the threshold, which is the only kind a gate ever has to "
            f"decide. Writing down '{result['rule']}' takes agreement to "
            f"{result['after']['exact']} of {result['after']['total']}",
        ),
        practice.Check(
            "FINDING: a three-point rubric reaches the gate as one boolean",
            all([len(result["check_fields"]) == 3,
                 result["check_fields"] == ["check_id", "passed", "evidence"],
                 result["case_verdicts"] == [False, False, False, True, True],
                 not result["gate_passed"]]),
            f"EvidenceCheck carries {result['check_fields']}, so a 0-2 score and two raters "
            f"collapse to {result['case_verdicts']} before evaluate_evidence_checks sees "
            "them. The threshold moves out of the gate and into whoever fills in the field, "
            "which is the person the calibration was supposed to constrain",
        ),
        practice.Check(
            "FINDING: the evidence string is required and never read",
            all([result["thin_passed"], result["thin_evidence"] == "x",
                 not result["empty_accepted"]]),
            f"a check whose evidence is {result['thin_evidence']!r} validates and passes, "
            f"while an all-whitespace one is rejected ({result['empty_accepted']}). "
            "evaluate_evidence_checks tests that the string is non-empty and stops -- the "
            "field that makes a judgment reviewable is the field nothing reviews",
        ),
        practice.Check(
            "FINDING: the gate is all(passed), so there is no room for a mean",
            all([result["mean_strict"] == 1.0, result["mean_generous"] == 1.6,
                 result["collapsed"] == 2, result["collapsed_generous"] == 3,
                 result["generous_verdicts"] != result["case_verdicts"]]),
            f"a rubric averaging {result['mean_strict']} and one averaging "
            f"{result['mean_generous']} reach the gate as {result['collapsed']} and "
            f"{result['collapsed_generous']} passing booleans out of 5, and the mean itself "
            "never arrives. A genuinely continuous dimension has to be discretized by the "
            "rater rather than by policy",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
