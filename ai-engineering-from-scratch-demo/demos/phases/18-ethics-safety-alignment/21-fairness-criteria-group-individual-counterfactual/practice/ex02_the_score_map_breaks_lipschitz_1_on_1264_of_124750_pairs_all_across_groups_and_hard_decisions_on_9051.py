"""Exercise 2 — the score map breaks Lipschitz-1 on 1264 of 124,750 pairs, all across groups; hard decisions on 9051.

    Implement the Dwork et al. 2012 individual-fairness metric using L2 on
    non-sensitive features. Report how many pairs violate Lipschitz with
    constant L=1.

Reading of the exercise: the individuals are `main()`'s 500 test rows on its
shipped seed, every unordered pair (124,750). d is L2 on the two non-sensitive
features (x0, x1). Dwork's map sends a person to a distribution over outcomes
and compares outcomes by total variation, which for a yes/no decision is
|p(x) - p(x')| with p the classifier's probability -- that is the faithful
metric. The 0/1 decision `predict()` returns is reported alongside, because it
is what the lesson's code actually outputs.

**ANSWER: the baseline violates L = 1 on 1264 of 124,750 pairs (1.01%) with
probabilities, and on 9051 (7.26%) with hard decisions.** After the DP
reweighting: 2 pairs with probabilities, 4075 with hard decisions.

**FINDING: every probability violation is a cross-group pair, and the
sensitive-attribute weight is the whole cause.** The sigmoid's slope is at
most 1/4, so on non-sensitive features the map is Lipschitz with constant
0.25 * ||(w0, w1)|| = 0.235 (baseline) and 0.187 (reweighted): no two people
in the same group can violate L = 1. The only way to break it is the +1.35
weight on A, which moves two people with identical features apart. Zeroing that
one weight leaves 0 violations. Individual fairness here fails for exactly the
reason group fairness does, just measured pair by pair.

**FINDING: a hard-threshold decision is not Lipschitz for any useful L.**
Any two people on either side of the boundary, however close, differ by 1.
Of the baseline's 9051 hard violations, 1908 are between people in the same
group, and the reweighting leaves 4075 (2075 same-group) though its A-weight
is near zero. Dwork's condition only makes sense on a randomized or scored
decision.

Structure: `test_rows()` rebuilds the seeded test set and both models through
the reference; `violations()` counts pairs, split by whether the two people
share a group; `certificate()` is the sigmoid Lipschitz bound.
"""

from __future__ import annotations

import itertools
import math
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "21-fairness-criteria-group-individual-counterfactual"
L = 1.0


def test_rows(ref, seed=53):
    """main()'s test set and its two models, via the reference, on its shipped seed."""
    saved = ref.random
    ref.random = random.Random(seed)
    try:
        train, test = ref.gen(1000), ref.gen(500)
        weights = [{(0, 1): 2.0, (1, 1): 0.5}.get((a, y), 1.0) for _, y, a in train]
        return test, ref.train(train), ref.train(train, sample_weights=weights)
    finally:
        ref.random = saved


def outputs(model, test, hard):
    w, b = model[:3], model[3]
    z = [b + sum(wi * xi for wi, xi in zip(w, x)) for x, _, _ in test]
    return [float(v > 0) if hard else 1.0 / (1.0 + math.exp(-v)) for v in z]


def violations(model, test, hard=False):
    """(pairs violating |f(x) - f(x')| <= L * d(x, x'), of which same-group, pairs checked)."""
    f, bad, same, n = outputs(model, test, hard), 0, 0, 0
    for i, j in itertools.combinations(range(len(test)), 2):
        n += 1
        if abs(f[i] - f[j]) > L * math.dist(test[i][0][:2], test[j][0][:2]):
            bad += 1
            same += test[i][2] == test[j][2]
    return bad, same, n


def certificate(model):
    """Lipschitz constant of sigmoid(w . x + b) in the non-sensitive features."""
    return round(0.25 * math.hypot(model[0], model[1]), 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    test, base, rew = test_rows(ref)
    blind = base[:2] + [0.0] + base[3:]
    return {
        "soft": (violations(base, test), violations(rew, test)),
        "hard": (violations(base, test, True), violations(rew, test, True)),
        "blind": violations(blind, test), "cert": (certificate(base), certificate(rew)),
        "w_a": round(base[2], 2),
    }


def verify(result):
    (sb, sr), (hb, hr), cert = result["soft"], result["hard"], result["cert"]
    return [
        practice.Check(
            "ANSWER: 1264 of 124,750 pairs with probabilities, 9051 with hard decisions",
            sb[::2] == (1264, 124750) and hb[0] == 9051 and (sr[0], hr[0]) == (2, 4075)
            and (round(sb[0] / sb[2], 4), round(hb[0] / hb[2], 4)) == (0.0101, 0.0726),
            f"baseline (violations, same-group, pairs): soft {sb}, hard {hb}; "
            f"reweighted soft {sr}, hard {hr}",
        ),
        practice.Check(
            "FINDING: every probability violation is a cross-group pair; the A weight is the cause",
            sb[1] == sr[1] == 0 and cert == (0.235, 0.187) and result["w_a"] == 1.35
            and result["blind"][0] == 0,
            f"same-group soft violations {sb[1]}/{sr[1]}; Lipschitz bound on (x0, x1) {cert}; "
            f"weight on A {result['w_a']}; with it zeroed {result['blind'][0]} violations",
        ),
        practice.Check(
            "FINDING: a hard-threshold decision is not Lipschitz for any useful L",
            (hb[1], hr[1]) == (1908, 2075),
            f"same-group hard violations: baseline {hb[1]} of {hb[0]}, reweighted {hr[1]} of {hr[0]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
