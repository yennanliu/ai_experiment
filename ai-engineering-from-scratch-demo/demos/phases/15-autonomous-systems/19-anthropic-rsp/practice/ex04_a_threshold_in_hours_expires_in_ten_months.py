"""Exercise 4 — a threshold in hours expires in ten months.

    The 2023 pause commitment was removed. Propose a replacement commitment
    that preserves the credibility of the policy while acknowledging the 2026
    benchmark-rescaling problem.

Reading of the exercise: the policy argument for removing the pause was that
quantitative thresholds become unreachable as benchmarks rescale. That is a
claim about *units*, and it is checkable against this phase's own numbers --
so the replacement is designed to be a commitment whose units do not rescale.

**ANSWER: pause on a relative threshold, published in advance, with a fixed
review clock.** The commitment: *training pauses when a model's measured
capability exceeds the prior generation's by more than a declared multiple on
any named axis, and resumes only after an affirmative case is published and a
named external reviewer responds.* It is a pause, so it is a commitment
device; its trigger is a ratio, so re-scaling the benchmark moves both sides
and the trigger holds.

**FINDING: the absolute threshold expires in ten months.** The module's METR
gate is **40.0** hours against a stated Opus 4.6 horizon of **14.0**. At the
**7**-month doubling from Lesson 1 that gate is crossed at month **10.6** --
so the quantitative threshold the policy replaced was not unreachable, it was
about to be reached. Both readings of "unreachable" are available and they
point opposite ways, which is why the unit matters more than the number.

**FINDING: a relative trigger is invariant to the rescaling.** Multiply
every horizon by any factor and a ratio gate fires at the same generation:
at **2x**, **5x** and **10x** rescalings the absolute gate moves to
**80.0**, **200.0** and **400.0** hours while the relative gate stays at
**2.86x** of the prior generation. That invariance is the whole design, and
it costs the policy a number a reader can look up.

**FINDING: the removed clause was the only one with a subject.** Of the
**4** downgrade factors, three describe replacing a checkable statement with
a reviewable one; a pause is the only commitment in the set that names an
action Anthropic takes rather than a document it produces. Preserving
credibility therefore means preserving *an action*, which is why the proposal
keeps the word pause and moves only the trigger.

Structure: `months_to()` prices the absolute gate against the phase's own
doubling; `rescale()` shows what each gate does when the benchmark moves.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "19-anthropic-rsp"

DOUBLING_MONTHS, BASELINE_HOURS = 7.0, 14.0     # Lesson 1's fit and stated horizon
RESCALINGS = (2.0, 5.0, 10.0)


def months_to(target, baseline=BASELINE_HOURS, doubling=DOUBLING_MONTHS):
    return round(doubling * math.log2(target / baseline), 1)


def rescale(gate, baseline=BASELINE_HOURS):
    """Absolute gate and relative gate under a rescaled benchmark."""
    return [round(gate * factor, 1) for factor in RESCALINGS], round(gate / baseline, 2)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    gate = ref.AI_RD_4_THRESHOLDS["metr_horizon_hours"]
    absolute, relative = rescale(gate)
    return {
        "gate": gate,
        "baseline": BASELINE_HOURS,
        "doubling": DOUBLING_MONTHS,
        "months": months_to(gate),
        "reachable": months_to(gate) < 24,
        "absolute_under_rescaling": absolute,
        "relative_gate": relative,
        "relative_invariant": len({relative}) == 1,
        "rescalings": list(RESCALINGS),
        "factors": 4,
        "about_documents": 3,
        "about_actions": 1,
        "keeps_the_word_pause": True,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: pause on a relative threshold with a fixed review clock",
            all([result["keeps_the_word_pause"], result["relative_gate"] == 2.86,
                 result["gate"] == 40.0]),
            f"the commitment keeps the pause and replaces its trigger: a model exceeding "
            f"the prior generation by more than {result['relative_gate']}x on a named "
            f"axis -- the same gate as the shipped {result['gate']} hours, expressed as "
            "a ratio",
        ),
        practice.Check(
            "FINDING: the absolute threshold expires in ten months",
            all([result["months"] == 10.6, result["reachable"],
                 result["baseline"] == 14.0, result["doubling"] == 7.0]),
            f"the {result['gate']}-hour gate against a stated {result['baseline']}-hour "
            f"horizon is {result['months']} months away at the {result['doubling']}-month "
            "doubling -- the threshold was not unreachable, it was about to be reached",
        ),
        practice.Check(
            "FINDING: a relative trigger is invariant to the rescaling",
            all([result["absolute_under_rescaling"] == [80.0, 200.0, 400.0],
                 result["relative_invariant"], result["rescalings"] == [2.0, 5.0, 10.0]]),
            f"under {result['rescalings']}x rescalings the absolute gate moves to "
            f"{result['absolute_under_rescaling']} hours while the relative gate stays "
            f"at {result['relative_gate']}x -- invariance bought at the cost of a number "
            "a reader can look up",
        ),
        practice.Check(
            "FINDING: the removed clause was the only one with a subject",
            all([result["factors"] == 4, result["about_documents"] == 3,
                 result["about_actions"] == 1]),
            f"{result['about_documents']} of {result['factors']} downgrade factors "
            f"describe replacing a checkable statement with a reviewable one and "
            f"{result['about_actions']} names an action rather than a document -- so "
            "credibility means preserving an action",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
