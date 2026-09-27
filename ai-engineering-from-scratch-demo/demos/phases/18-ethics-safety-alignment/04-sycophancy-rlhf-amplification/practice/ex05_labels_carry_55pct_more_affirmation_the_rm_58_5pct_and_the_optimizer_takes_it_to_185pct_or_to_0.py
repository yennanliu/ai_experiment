"""Exercise 5 — labels carry 55% more affirmation, the RM 58.5%, and the optimizer takes it to 185% or to 0.

    The Stanford (2026) result: 49% more affirmation of user beliefs. Given
    labelers' preference for affirmation, how much of this 49% is the RM
    versus the optimizer? Design an experiment that would separate the two.

Reading of the exercise: the 49% is a *relative* excess, so the toy is
measured the same way: R = P(affirm | biased pipeline) / P(affirm | unbiased
pipeline) - 1, where the biased pipeline is the reference as shipped
(labelers add 0.6 for agreement) and the unbiased one zeroes `AGREEMENT`
before the same seed-7 `train_rm`, so both reward models see the same
pairs and coin flips. R is read at each stage -- labels, reward model,
optimized policy -- and the stage's share is its part of log(1 + R).

**ANSWER: it is not a fixed split; the optimizer's part is set by its
pressure.** The labels already prefer agreement over a correct answer 55%
more often (54.9%: 0.332 vs 0.214) and the reward model's implied preference
is 58.5% more (0.349 vs 0.220), so the RM passes the labelers' bias on almost intact.
After PPO the excess is 63.6% at beta = 1, 169.4% at 0.5, 185.4% at 0.2,
145.5% at 0.1 and 110.5% at 0: the RM's share of log(1 + R) is 94% at beta = 1 and 44% at
beta = 0.2. A rank-based optimizer transmits none of it: best-of-N gives R = 0
for N = 1..4 because both reward models rank A > S > W.

**FINDING: the relative excess and the absolute rate move in opposite
directions.** From beta = 1 to 0 the biased policy's P(S) falls from 0.300
to 0.030 and the absolute gap, after peaking at 0.127 at beta = 0.5, ends at
0.016, while R never drops below its RM-level 58.5%. A "49% more" headline can grow while sycophancy shrinks.

**ANSWER: the experiment.** A 2 x k factorial: {reward model trained on the
raw labels, reward model trained on debiased labels} x {no optimizer (score
the RM on matched pairs), best-of-N, KL-RL at several beta}. The RM effect
is the column difference at "no optimizer"; the optimizer effect is how that
difference changes along each row. The table above is that experiment run in
the toy. In a real model the debiased RM comes from relabeling matched
user/third-party pairs with ground truth.

Structure: `rms()` fits both reward models on the same random stream;
`ratio()` turns a pair of affirmation rates into R.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "04-sycophancy-rlhf-amplification"
BETAS, SEEDS, NS = (1.0, 0.5, 0.2, 0.1, 0.0), (0, 1), (1, 2, 3, 4)
NO_BONUS = {"A": 0.0, "S": 0.0, "W": 0.0}


def world(ref, fn, agree=None, seed=7):
    saved = ref.AGREEMENT, ref.random
    ref.AGREEMENT, ref.random = agree or saved[0], random.Random(seed)
    try:
        return fn()
    finally:
        ref.AGREEMENT, ref.random = saved


def rms(ref):
    return world(ref, ref.train_rm), world(ref, ref.train_rm, NO_BONUS)


def sig(x):
    return 1 / (1 + math.exp(-x))


def ratio(biased, unbiased):
    return round(biased / unbiased - 1, 3)


def ppo_s(ref, rm, beta):
    runs = [world(ref, lambda: ref.ppo_train([0.0] * 3, rm, beta), seed=s) for s in SEEDS]
    return sum(ref.sycophancy(ref.softmax(lg)) for lg in runs) / len(runs)


def best_of_n(rm, n):
    """P(best-of-n picks S) from the uniform base: no A drawn, at least one S."""
    ranked = sorted(rm, key=rm.get, reverse=True)
    above = ranked.index("S")
    return ((3 - above) / 3) ** n - ((2 - above) / 3) ** n


def rounded(pair):
    return [round(p, 3) for p in pair]


def by_beta(ppo, r_rm):
    """Per beta: R, the RM's share of log(1 + R), biased P(S), and the absolute gap."""
    r_ppo = {b: ratio(*p) for b, p in ppo.items()}
    return {
        "ppo": r_ppo,
        "rm_share": {b: round(math.log(1 + r_rm) / math.log(1 + r), 2) for b, r in r_ppo.items()},
        "biased_s": {b: round(p[0], 3) for b, p in ppo.items()},
        "gap": {b: round(p[0] - p[1], 3) for b, p in ppo.items()},
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    biased, fair = rms(ref)
    labels = [world(ref, lambda: sig(ref.labeler_reward("S") - ref.labeler_reward("A")), ag)
              for ag in (None, NO_BONUS)]
    rm_pref = [sig(r["S"] - r["A"]) for r in (biased, fair)]
    ppo = {b: (ppo_s(ref, biased, b), ppo_s(ref, fair, b)) for b in BETAS}
    return {
        "labels": (rounded(labels), ratio(*labels)),
        "rm": (rounded(rm_pref), ratio(*rm_pref)),
        **by_beta(ppo, ratio(*rm_pref)),
        "bon": {n: ratio(best_of_n(biased, n), best_of_n(fair, n)) for n in NS},
    }


def verify(result):
    ppo, share, gap, bs = (result[k] for k in ("ppo", "rm_share", "gap", "biased_s"))
    return [
        practice.Check(
            "ANSWER: it is not a fixed split; the optimizer's part is set by its pressure",
            (result["labels"], result["rm"], share[1.0], share[0.2])
            == (([0.332, 0.214], 0.549), ([0.349, 0.22], 0.585), 0.94, 0.44)
            and ppo == {1.0: 0.636, 0.5: 1.694, 0.2: 1.854, 0.1: 1.455, 0.0: 1.105}
            and set(result["bon"].values()) == {0.0},
            f"R labels {result['labels']}, RM {result['rm']}, PPO by beta {ppo}; RM share of "
            f"log(1+R) {share}; best-of-N R {result['bon']}",
        ),
        practice.Check(
            "FINDING: the relative excess and the absolute rate move in opposite directions",
            (bs[1.0], bs[0.0], max(gap.values()), gap[0.5], gap[0.0]) == (0.3, 0.03, 0.127, 0.127, 0.016)
            and min(ppo.values()) > result["rm"][1] < ppo[1.0] < ppo[0.0],
            f"biased P(S) by beta {bs}; absolute gap {gap}; R stays above the RM's "
            f"{result['rm'][1]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
