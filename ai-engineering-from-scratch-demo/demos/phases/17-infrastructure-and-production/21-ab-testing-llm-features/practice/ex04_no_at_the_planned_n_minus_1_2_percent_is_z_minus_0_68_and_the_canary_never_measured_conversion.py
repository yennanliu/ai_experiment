"""Exercise 4 — no: at the planned n, -1.2% is z = -0.68, and the canary never measured conversion.

    Your canary passes but A/B shows -1.2% conversion. Do you ship? Write the
    escalation criteria.

Reading of the exercise: the A/B is the one sized in exercise 1 -- 3%
baseline, 207,702 users per arm -- and "-1.2%" is read both ways: relative
(3% -> 2.964%) and absolute (3% -> 1.8%). "The canary" is Lesson 20's, the
prerequisite this lesson names, run on its own code.

**ANSWER: do not ship; hold and escalate.** Relative, the drop is z = -0.68
under the reference `z_statistic` and its 95% interval is -4.7% to +2.2%
relative: not proof of harm, but it cannot exclude a 4.7% loss, and an
unchanged variant reads -1.2% or worse 25% of the time. Under the lesson's
own skill rule -- "primary significant + all guardrails not
significant-negative -> ship" -- a non-significant primary is not a ship.
Absolute, z = -25.3: roll back. The criteria, as `decide()` runs them:
1. upper bound of the interval below 0 -> roll back now;
2. lower bound below the harm margin (-1% relative) -> hold, extend, and
   escalate to the metric owner with the n needed;
3. otherwise ship only if the primary is significantly positive or a
   non-inferiority test at that margin passes, with a written sign-off.

**FINDING: the lesson's canary cannot see conversion, so "passes" was
guaranteed.** Lesson 20's five gates are latency, cost, error rate, output
length and thumbs-down; none is conversion, and a clean `Regression()`
passes all 6 stages. The canary and the A/B answer different questions.

**FINDING: confirming a 1.2% drop takes 3,500,264 users per arm, 16.9x the
plan.** `fixed_sample_size(0.03, -0.012)`; and the sequential rule the lesson
demos needs |z| > 4.35 at the planned n, so it cannot fire on this effect.

Structure: `decide()` is the escalation rule; `interval()` uses the reference
`z_statistic` at the planned n, and the canary is Lesson 20's own code.
"""

from __future__ import annotations

import math
from statistics import NormalDist

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "21-ab-testing-llm-features"
CANARY = "20-shadow-canary-progressive"
BASE, MARGIN = 0.03, -0.01  # baseline conversion; relative harm margin


def interval(ref, n, p_treat):
    """(z, relative lift, 95% interval of the relative lift) at n per arm."""
    a, b = round(BASE * n), round(p_treat * n)
    z = ref.z_statistic(a, n, b, n)
    se = math.sqrt((a / n) * (1 - a / n) / n + (b / n) * (1 - b / n) / n)
    diff = b / n - a / n
    return z, diff / BASE, ((diff - 1.96 * se) / BASE, (diff + 1.96 * se) / BASE)


def decide(z, ci):
    low, high = ci
    if high < 0:
        return "rollback"
    if low < MARGIN:
        return "hold-and-escalate"
    return "ship" if z > 1.96 else "ship-with-signoff"


def canary_passes(canary):
    """Every stage of a regression-free rollout through Lesson 20's five gates."""
    reg = canary.Regression()
    return all(
        not canary.check_gates(canary.measure_stage(s, reg, canary.stage_seed(i)))
        for i, s in enumerate(canary.STAGES)
    )


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    canary = parity.load_reference(PHASE, CANARY, "main")
    n = ref.fixed_sample_size(BASE, 0.05)
    rel, absolute = interval(ref, n, BASE * 0.988), interval(ref, n, BASE - 0.012)
    return {
        "n": n,
        "rel": rel,
        "abs": absolute,
        "decisions": (decide(rel[0], rel[2]), decide(absolute[0], absolute[2])),
        "null_tail": NormalDist().cdf(rel[0]),
        "gates": sorted(canary.GATES),
        "canary_passes": canary_passes(canary),
        "n_detect": ref.fixed_sample_size(BASE, -0.012),
        "boundary": math.sqrt(2 * math.log(20) + math.log(2 * n)),
    }


def answer_holds(result):
    z, _, (low, high) = result["rel"]
    measured = (
        round(z, 2),
        round(low, 3),
        round(high, 3),
        round(result["null_tail"], 2),
    )
    return measured == (-0.68, -0.047, 0.022, 0.25) and all(
        [
            round(result["abs"][0], 1) == -25.3,
            result["decisions"] == ("hold-and-escalate", "rollback"),
        ]
    )


def verify(result):
    z, _, (low, high) = result["rel"]
    return [
        practice.Check(
            "ANSWER: do not ship; hold and escalate",
            answer_holds(result),
            f"relative: z = {z:.2f}, interval {low:+.1%} to {high:+.1%}, P(<= -1.2% | no "
            f"effect) = {result['null_tail']:.2f} -> {result['decisions'][0]}; absolute: "
            f"z = {result['abs'][0]:.1f} -> {result['decisions'][1]}",
        ),
        practice.Check(
            "FINDING: the lesson's canary cannot see conversion, so 'passes' was guaranteed",
            result["canary_passes"] and not any("conver" in g for g in result["gates"]),
            f"Lesson 20's gates are {result['gates']}; a clean rollout passes all stages",
        ),
        practice.Check(
            "FINDING: confirming a 1.2% drop takes 3,500,264 users per arm, 16.9x the plan",
            result["n_detect"] == 3500264
            and round(result["n_detect"] / result["n"], 1) == 16.9
            and abs(z) < result["boundary"],
            f"{result['n_detect']} per arm vs {result['n']}; the sequential boundary at "
            f"the planned n is {result['boundary']:.2f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
