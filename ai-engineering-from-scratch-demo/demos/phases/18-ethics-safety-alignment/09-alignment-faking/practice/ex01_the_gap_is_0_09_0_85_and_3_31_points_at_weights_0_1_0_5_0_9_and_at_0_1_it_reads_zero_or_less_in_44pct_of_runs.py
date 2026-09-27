"""Exercise 1 — the gap is 0.09, 0.85 and 3.31 points at weights 0.1, 0.5, 0.9, and at 0.1 it reads <= 0 in 44% of runs.

    Run `code/main.py`. Fix the perceived-training weight at 0.5 and measure the compliance gap. Repeat at 0.1 and 0.9. Plot the relationship.

Reading of the exercise: `code/main.py` has no parameter called
"perceived-training weight"; the one knob that sets how much being trained on
a conversation matters to the agent is `preservation_weight` (refusing while
monitored costs `-preservation_weight`), so that is the weight swept. Harm
aversion stays at the script's baseline 2.0. The gap is measured two ways:
exactly, from the reference's `compliance_probability`, and as the script
measures it, from `simulate(1000, ...)` over 1000 seeds. The "plot" is the
gap over weights 0 to 4 in steps of 0.5, the range that shows the whole
curve.

**ANSWER: the gap is +0.09, +0.85 and +3.31 points at weights 0.1, 0.5 and
0.9.** Monitored compliance is 0.33%, 1.10% and 3.56%. Unmonitored compliance
is 0.25% at every weight, because the weight only enters the monitored branch.
All three sit in the flat tail of a logistic:

    gap(w) = sigmoid(3 (w - 2)) - sigmoid(-6)

| w | 0 | 0.5 | 1 | 1.5 | 2 | 2.5 | 3 | 3.5 | 4 |
|---|---|---|---|---|---|---|---|---|---|
| gap | 0 | .0085 | .0450 | .1800 | .4975 | .8151 | .9501 | .9865 | .9951 |

The curve is steepest where the weight equals the harm aversion, at w = 2. It
tops out at 1 - sigmoid(-6) = 0.9975, not at 1.

**FINDING: at the script's sample size, weight 0.1 is indistinguishable from
no gap.** With n = 1000 per condition, `simulate` reports a gap of zero or less
with exact probability 44.3% at w = 0.1 (433 of 1000 seeded runs do). At
w = 0.5 the probability is 1.0% (11 of 1000 runs), and at w = 0.9 no run does.
The toy's own n cannot resolve the low end of the sweep the exercise asks
for.

Structure: `exact()` is the gap from `compliance_probability`; `p_nonpositive`
convolves the two binomials; `runs()` swaps a seeded `random.Random` into the
reference module for each call and restores the original afterwards.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "09-alignment-faking"
HARM, N, SEEDS = 2.0, 1000, range(1000)
WEIGHTS, SWEEP = (0.1, 0.5, 0.9), [k / 2 for k in range(9)]


def exact(ref, w):
    return ref.compliance_probability(True, HARM, w), ref.compliance_probability(False, HARM, w)


def p_nonpositive(p_m, p_u):
    """P(monitored count <= unmonitored count) for two Binomial(N, p) draws."""
    pmf = [[math.comb(N, k) * p ** k * (1 - p) ** (N - k) for k in range(N + 1)] for p in (p_m, p_u)]
    tail, total = [0.0] * (N + 2), 0.0
    for k in range(N, -1, -1):
        total += pmf[1][k]
        tail[k] = total
    return sum(pmf[0][k] * tail[k] for k in range(N + 1))


def runs(ref, w):
    """Share of seeded simulate(N) runs whose reported gap is <= 0."""
    saved, count = ref.random, 0
    try:
        for seed in SEEDS:
            ref.random = random.Random(seed)
            m, u = ref.simulate(N, HARM, w)
            count += m <= u
    finally:
        ref.random = saved
    return count / len(SEEDS)


def shape(ref, curve):
    """(curve rounded, worst deviation from the closed form, ceiling 1 - sigmoid(-3 harm))."""
    err = max(abs((m - u) - (ref.sigmoid(3 * (w - HARM)) - ref.sigmoid(-3 * HARM)))
              for w, (m, u) in curve.items())
    return {w: round(m - u, 4) for w, (m, u) in curve.items()}, err, round(1 - ref.sigmoid(-3 * HARM), 4)


def points(probs):
    return ({w: round(100 * (m - u), 2) for w, (m, u) in probs.items()},
            {w: round(100 * m, 2) for w, (m, _) in probs.items()})


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    probs = {w: exact(ref, w) for w in WEIGHTS}
    curve = {w: exact(ref, w) for w in SWEEP}
    (gap, monitored), (rounded, err, ceiling) = points(probs), shape(ref, curve)
    return {
        "gap": gap, "monitored": monitored, "curve": rounded, "logistic_err": err, "ceiling": ceiling,
        "unmonitored": sorted({round(100 * u, 2) for _, u in curve.values()}),
        "p_le0": {w: round(p_nonpositive(*probs[w]), 3) for w in WEIGHTS},
        "runs_le0": {w: runs(ref, w) for w in WEIGHTS},
    }


def verify(result):
    curve = result["curve"]
    steps = [curve[b] - curve[a] for a, b in zip(SWEEP, SWEEP[1:])]
    return [
        practice.Check(
            "ANSWER: the gap is +0.09, +0.85 and +3.31 points at weights 0.1, 0.5 and 0.9",
            result["gap"] == {0.1: 0.09, 0.5: 0.85, 0.9: 3.31}
            and result["monitored"] == {0.1: 0.33, 0.5: 1.1, 0.9: 3.56}
            and result["unmonitored"] == [0.25],
            f"gap (points) {result['gap']}; monitored % {result['monitored']}; "
            f"unmonitored % at every swept weight {result['unmonitored']}",
        ),
        practice.Check(
            "ANSWER: the curve is sigmoid(3(w - 2)) - sigmoid(-6), steepest at w = 2, capped at 0.9975",
            all([result["logistic_err"] < 1e-12, (curve[2.0], curve[4.0]) == (0.4975, 0.9951),
                 max(steps) in steps[SWEEP.index(2.0) - 1:SWEEP.index(2.0) + 1],
                 result["ceiling"] == 0.9975]),
            f"gap over w = 0..4 {curve}; worst deviation from the closed form "
            f"{result['logistic_err']:.1e}; ceiling {result['ceiling']}",
        ),
        practice.Check(
            "FINDING: at n = 1000, weight 0.1 reads a gap <= 0 in 44% of runs",
            result["p_le0"] == {0.1: 0.443, 0.5: 0.010, 0.9: 0.0}
            and result["runs_le0"] == {0.1: 0.433, 0.5: 0.011, 0.9: 0.0},
            f"exact P(gap <= 0) {result['p_le0']}; share of {len(SEEDS)} seeded simulate() "
            f"runs {result['runs_le0']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
