"""Exercise 3 — the mean-gap condition holds in the toy, but amplification only appears above beta 3 and peaks at 0.3347.

    Read Shapira et al. (arXiv:2602.01002) Section 3. Identify the key theorem
    and restate it in plain English in two sentences.

Reading of the exercise: Section 3 of the paper (read 2026-09-27) states a
covariance theorem for KL-regularized optimization -- the shift in any
behaviour g under the optimal policy equals Cov(g, e^{r/beta}) / Z under the
base policy (the lesson's beta is the KL coefficient, the paper's inverse) --
and its weak-optimization corollary, the mean-gap condition, plus a
closed-form minimal agreement penalty. The restatement is then tested on the
lesson's toy, whose `ppo_train` optimizes exactly that KL-regularized
objective, so every clause becomes a number.

**ANSWER: the theorem in two sentences.** Under light optimization,
RLHF makes a model agree with users' stated beliefs more often exactly when,
on the base model's own answers, agreeing answers get a higher average
learned reward than correcting ones -- whatever the labelers intended. At
any strength, the change in agreement is its covariance with the
exponentiated reward, so the verdict belongs to whichever answers own the
top of the reward distribution, and the smallest fix is a closed-form
penalty on agreement that cancels that covariance.

**FINDING: the toy satisfies the mean-gap condition, and still de-amplifies
at every beta it runs.** The seed-7 reward model gives agreement +0.051
against -0.026 for the rest, a gap of +0.077, so the theorem predicts
amplification for weak optimization, and it occurs: exact P(S) is above 1/3
for beta above 3.17 and peaks at 0.3347 at beta = 6.4; the reference's own
PPO gives 0.3346 at beta = 8 and matches the exact tilt at beta = 1 (0.3001
vs 0.3007). Below 3.17 the correct answer owns the top of the reward and
P(S) falls; main() runs only beta <= 1. The lesson's "any method that
upweights by exp(r) therefore upweights" sycophancy drops the theorem's
"weak optimization" clause.

**FINDING: the minimal penalty is 0 at the beta the lesson corrects.** The
paper's lambda* = max(0, beta * log(m1 / m0)), m = mean e^{r/beta} over
agreeing / other answers, is 0.0 at beta = 0.1, 0.2 and 1 -- the shipped
alpha sweep penalizes an amplification that is not there -- and 0.0461 at
beta = 8, where applying it through `agreement_penalty_correction` brings
PPO back to P(S) = 0.3333.

Structure: `tilt()` is Theorem 1's closed form over the reference RM;
`crossover()` bisects it; `ppo()` runs the reference PPO under a seeded
`random`.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "04-sycophancy-rlhf-amplification"
BASE = 1 / 3


def seeded(ref, seed, fn, *args, **kw):
    saved, ref.random = ref.random, random.Random(seed)
    try:
        return fn(*args, **kw)
    finally:
        ref.random = saved


def tilt(rm, beta):
    """Theorem 1's optimum: pi_beta(y) proportional to pi_0(y) e^{r(y)/beta}, pi_0 uniform."""
    w = {a: math.exp(r / beta) for a, r in rm.items()}
    return {a: v / sum(w.values()) for a, v in w.items()}


def crossover(rm, lo=1.0, hi=100.0):
    """The beta at which the exact P(S) crosses back through its base value 1/3."""
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if tilt(rm, mid)["S"] > BASE else (mid, hi)
    return (lo + hi) / 2


def minimal_penalty(rm, beta):
    """Theorem 6 in the lesson's units: lambda* = max(0, beta log(m1 / m0))."""
    m1 = math.exp(rm["S"] / beta)
    m0 = (math.exp(rm["A"] / beta) + math.exp(rm["W"] / beta)) / 2
    return max(0.0, beta * math.log(m1 / m0))


def ppo(ref, reward, beta):
    return ref.sycophancy(ref.softmax(seeded(ref, 0, ref.ppo_train, [0.0] * 3, reward, beta)))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rm = seeded(ref, 7, ref.train_rm)
    grid = [1 + 0.1 * i for i in range(200)]
    peak = max(grid, key=lambda b: tilt(rm, b)["S"])
    lam8 = minimal_penalty(rm, 8.0)
    doc = parity.doc_text(PHASE, LESSON, "en")
    return {
        "means": (round(rm["S"], 3), round((rm["A"] + rm["W"]) / 2, 3)),
        "gap": round(rm["S"] - (rm["A"] + rm["W"]) / 2, 3),
        "crossover": round(crossover(rm), 2),
        "peak": (round(peak, 1), round(tilt(rm, peak)["S"], 4)),
        "ppo": {b: round(ppo(ref, rm, b), 4) for b in (1.0, 8.0)},
        "exact_1": round(tilt(rm, 1.0)["S"], 4),
        "lambda": {b: round(minimal_penalty(rm, b), 4) for b in (0.1, 0.2, 1.0, 8.0)},
        "corrected_8": round(ppo(ref, ref.agreement_penalty_correction(rm, lam8), 8.0), 4),
        "doc_overclaims": "therefore upweights the marginal probability" in doc,
    }


def verify(result):
    lam, pp = result["lambda"], result["ppo"]
    return [
        practice.Check(
            "ANSWER: under weak optimization agreement drifts the way the mean gap points",
            result["gap"] > 0 and pp[8.0] > BASE > pp[1.0],
            f"mean gap {result['gap']:+}; reference PPO P(S) at beta 8 {pp[8.0]} > 1/3; at "
            f"beta 1 (strong optimization) {pp[1.0]} < 1/3",
        ),
        practice.Check(
            "FINDING: the toy satisfies the mean-gap condition, and still de-amplifies at "
            "every beta it runs",
            [result[k] for k in ("means", "gap", "crossover", "peak", "exact_1", "doc_overclaims")]
            == [(0.051, -0.026), 0.077, 3.17, (6.4, 0.3347), 0.3007, True]
            and pp == {1.0: 0.3001, 8.0: 0.3346},
            f"E[r | agree], E[r | other] {result['means']}, gap {result['gap']}; exact P(S) > 1/3 "
            f"above beta {result['crossover']}, peak {result['peak']}; PPO {pp} vs exact "
            f"{result['exact_1']} at beta 1",
        ),
        practice.Check(
            "FINDING: the minimal penalty is 0 at the beta the lesson corrects",
            lam == {0.1: 0.0, 0.2: 0.0, 1.0: 0.0, 8.0: 0.0461}
            and result["corrected_8"] == 0.3333,
            f"lambda* by beta {lam}; PPO at beta 8 with lambda* applied: "
            f"P(S) {result['corrected_8']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
