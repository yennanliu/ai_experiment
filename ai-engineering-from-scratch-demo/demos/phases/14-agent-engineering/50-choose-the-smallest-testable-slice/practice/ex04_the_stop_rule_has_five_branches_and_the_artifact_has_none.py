"""Exercise 4 — the stop rule has five branches and the artifact has none.

    Add a stop rule for a failed pilot.

Reading of the exercise: the lesson lists five things a failure can mean --
abandon the outcome, change the user or situation, test a different
mechanism, collect better evidence, narrow authority -- and says that if
every result leads to "keep building", the slice is not an experiment. So the
rule is a mapping from observed result to one of those five, written before
the pilot runs.

**ANSWER: five branches, none of which is "keep building", and the result
already in hand selects "collect better evidence".** The pilot's measurement
is the number-trace from Lesson 49: **87.5%** against a 90% threshold, with
both misses being lesson references rather than claims. Under the rule that
maps a near-miss caused by a metric defect to better evidence, the pilot
fails and the build does not continue on the same footing.

**FINDING: `Slice` has 7 fields and `decision` returns 3 keys, none of them a
stop rule.** The document records what was chosen and what was rejected, so
the one thing written before the pilot that is supposed to bind afterwards is
the one thing that does not travel with it.

**FINDING: the rule has to name the observation, not the feeling.** Each of
the five branches here is selected by a measured condition -- a rate below
threshold, a rate below half, a detector that never fired, a miss traced to
the metric, an error surviving more than one cycle -- so **5** of **5**
branches can be evaluated from a run rather than argued. A branch reading
"if it feels wrong" would make the rule decorative.

**FINDING: the same failure maps to different branches depending on one
number.** At **87.5%** the rule says collect better evidence; at **57.1%** --
the worst per-lesson rate the previous lesson measured -- it says test a
different mechanism. The stop rule is a function of the result, so writing it
before the pilot is what stops the result from choosing its own
interpretation.

Structure: `BRANCHES` is the rule; `apply()` evaluates it against a measured
pilot.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "50-choose-the-smallest-testable-slice"
THRESHOLD = 0.90
MEASURED = 0.875          # Lesson 49's traced-number rate for one finished lesson
WORST = 0.571             # the worst per-lesson rate in that same run
PILOT = ("publish one lesson, numbers only, planted error", 3, 5, 2, 3, True,
         ("numbers-trace", "wrong-number-detected"))

# (branch, the measured condition that selects it)
BRANCHES = [
    ("abandon the outcome", "the detector fires and readers still miss the error"),
    ("change the user or situation", "the rate clears the threshold and nobody reads it"),
    ("test a different mechanism", "the rate falls more than fifteen points below"),
    ("collect better evidence", "the misses trace to the metric rather than the generator"),
    ("narrow authority", "a planted error survives more than one publish cycle"),
]


def apply(rate, metric_defect, threshold=THRESHOLD):
    """The rule, written before the pilot and evaluated after it."""
    if rate >= threshold:
        return "continue"
    if rate < threshold - 0.15:
        return "test a different mechanism"
    if metric_defect:
        return "collect better evidence"
    return "narrow authority"


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pilot = ref.Slice(*PILOT)
    document = ref.decision([pilot], {"numbers-trace", "wrong-number-detected"})
    return {
        "branches": len(BRANCHES),
        "keep_building": sum(branch.startswith("keep") for branch, _ in BRANCHES),
        "measured": MEASURED, "threshold": THRESHOLD,
        "decision": apply(MEASURED, metric_defect=True),
        "passing": apply(0.95, metric_defect=False),
        "worst_decision": apply(WORST, metric_defect=True),
        "fields": len(ref.Slice.__dataclass_fields__),
        "keys": sorted(document),
        "stop_field": any(name in ref.Slice.__dataclass_fields__
                          for name in ("stop_rule", "on_failure", "halt")),
        "stop_key": any("stop" in key for key in document),
        "measurable": sum(any(word in condition for word in
                              ("rate", "fires", "survives", "trace"))
                          for _, condition in BRANCHES),
        "selected": document["selected"]["name"][:22],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: five branches, none of them keep building, and 87.5% selects one",
            all([result["branches"] == 5, result["keep_building"] == 0,
                 result["measured"] == 0.875,
                 result["decision"] == "collect better evidence",
                 result["passing"] == "continue"]),
            f"the rule has {result['branches']} branches and "
            f"{result['keep_building']} of them is 'keep building'; the pilot's measured "
            f"{result['measured']:.1%} against a {result['threshold']:.0%} threshold "
            f"selects {result['decision']!r}",
        ),
        practice.Check(
            "FINDING: Slice has 7 fields and decision returns 3 keys, none a stop rule",
            all([result["fields"] == 7, len(result["keys"]) == 3,
                 result["stop_field"] is False, result["stop_key"] is False]),
            f"the slice carries {result['fields']} fields and the document returns "
            f"{result['keys']}, so the one thing written before the pilot that is supposed "
            "to bind afterwards does not travel with it",
        ),
        practice.Check(
            "FINDING: the rule has to name the observation, not the feeling",
            all([result["measurable"] == 5, result["branches"] == 5]),
            f"{result['measurable']} of {result['branches']} branches are selected by a "
            "measured condition -- a rate, a detector firing, an error surviving -- so the "
            "rule can be evaluated from a run rather than argued",
        ),
        practice.Check(
            "FINDING: the same failure maps to different branches on one number",
            all([result["decision"] == "collect better evidence",
                 result["worst_decision"] == "test a different mechanism"]),
            f"at 87.5% the rule says {result['decision']!r} and at 57.1% -- the worst "
            f"per-lesson rate measured -- it says {result['worst_decision']!r}; writing the "
            "rule first is what stops the result choosing its own interpretation",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
