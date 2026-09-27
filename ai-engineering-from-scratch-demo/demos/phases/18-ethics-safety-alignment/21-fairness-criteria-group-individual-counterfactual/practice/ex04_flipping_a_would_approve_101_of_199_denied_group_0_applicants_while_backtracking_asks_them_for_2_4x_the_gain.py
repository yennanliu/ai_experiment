"""Exercise 4 — flipping A would approve 101 of 199 denied group-0 applicants, while backtracking asks them for 2.4x the gain.

    The 2024 backtracking-counterfactuals paper avoids intervention on
    protected attributes. Describe a scenario where this matters for legal
    compliance.

Reading of the exercise: the scenario is a credit or hiring decision that must
come with an adverse-action notice -- in US consumer credit, Regulation B
requires the principal reasons for a denial, and ECOA forbids basing the
decision on a protected characteristic. The lesson's classifier stands in for
the lender, its 500 seeded test rows for the applicants, and the two kinds of
counterfactual the lesson contrasts are computed for every denied applicant:
the interventional one sets A to the other group (the proxy x1 shifts by 0.5
with it, the recorded work-sample score x0 stays); the backtracking one keeps
A as it is and asks for the smallest change to the applicant's own background
(the noise behind x0 and x1, in standard-deviation units) that would have led
to approval.

**ANSWER: the notice.** The interventional counterfactual turns the notice
into an admission: for the lesson's baseline, 101 of the 199 denied group-0
applicants (50.8%) would be approved "had you been in group 1", and 0 of the
104 denied group-1 applicants would be approved as group 0. That sentence
cannot be sent, and cannot be acted on. The backtracking counterfactual gives
a reason in the applicant's own features: x0 carries 89.7% of the minimal
shift, i.e. "a higher work-sample score".

**FINDING: backtracking does not hide the disparity; it prices it.** The
median shift a denied group-0 applicant needs is 1.25 SD, against 0.53 SD for
group 1 -- 2.4x -- because the +1.35 weight on A has to be made up out of
features. After the DP reweighting the medians are 0.86 and 0.92 SD, and
do(A) approves 0 of 170 denied group-0 and 10 of 181 denied group-1
applicants. An auditor can read discrimination off the recourse gap without
ever intervening on the protected attribute.

Structure: `lesson_models()` rebuilds main()'s seeded test set and models;
`audit()` computes both counterfactuals per denied applicant, per group. For a
linear score the minimal L2 background shift is exactly -z / ||(w0, w1)||.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "21-fairness-criteria-group-individual-counterfactual"


def lesson_models(ref, seed=53):
    """main()'s test set, baseline and DP-reweighted models, on its shipped seed."""
    saved, ref.random = ref.random, random.Random(seed)
    try:
        train, test = ref.gen(1000), ref.gen(500)
        weights = [{(0, 1): 2.0, (1, 1): 0.5}.get((a, y), 1.0) for _, y, a in train]
        return test, ref.train(train), ref.train(train, sample_weights=weights)
    finally:
        ref.random = saved


def score(model, x):
    return model[3] + sum(w * v for w, v in zip(model[:3], x))


def do_a(x, a):
    """Intervene on A: the proxy x1 moves with it; the recorded x0 does not."""
    return [x[0], x[1] + 0.5 * (a - x[2]), float(a)]


def audit(ref, model, test):
    """Per group: (denied, approved under do(A = other group), median backtracking shift)."""
    denied = [(x, a) for (p, _, a), (x, _, _) in zip(ref.predict(model, test), test) if p == 0]
    norm, out = math.hypot(model[0], model[1]), {}
    for g in (0, 1):
        xs = [x for x, a in denied if a == g]
        flipped = sum(score(model, do_a(x, 1 - g)) > 0 for x in xs)
        out[g] = (len(xs), flipped, round(statistics.median(-score(model, x) / norm for x in xs), 3))
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    test, base, rew = lesson_models(ref)
    b, r = audit(ref, base, test), audit(ref, rew, test)
    return {
        "base": b, "rew": r, "x0_share": round(base[0] ** 2 / (base[0] ** 2 + base[1] ** 2), 3),
        "ratio": round(b[0][2] / b[1][2], 1), "w_a": round(base[2], 2),
    }


def verify(result):
    b, r = result["base"], result["rew"]
    return [
        practice.Check(
            "ANSWER: do(A) approves 101 of 199 denied group-0 applicants; backtracking names x0",
            b[0][:2] == (199, 101) and b[1][:2] == (104, 0) and result["x0_share"] == 0.897,
            f"baseline (denied, approved under do(A), median shift) by group {b}; "
            f"x0 carries {result['x0_share']:.1%} of the backtracking shift",
        ),
        practice.Check(
            "FINDING: backtracking does not hide the disparity; it prices it",
            (b[0][2], b[1][2], result["ratio"]) == (1.254, 0.532, 2.4) and result["w_a"] == 1.35
            and r == {0: (170, 0, 0.859), 1: (181, 10, 0.916)},
            f"median shift group 0 {b[0][2]} vs group 1 {b[1][2]} SD ({result['ratio']}x), "
            f"weight on A {result['w_a']}; reweighted {r}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
