"""Exercise 2 — all three hack, and the penalty arrives after it is over.

    **Medium.** Run the toy PPO-RLHF loop with `β ∈ {0.0, 0.1, 1.0}`. For each,
    plot RM score vs KL-to-reference over updates. Which runs reward-hack?

Reading of the exercise: "which runs reward-hack" presumes the sweep separates them,
so reward hacking is given a threshold rather than an impression -- the reference
policy is uniform over the 12-token vocabulary, so KL is bounded above by
`ln 12 = 2.4849` and a run is hacking when it spends most of that budget. `β` is then
swept far past the stated range to find where the penalty starts to bind. 5 seeds per
`β`; the plot ships as a table.

**ANSWER: all three, including `β = 1.0`.** Final KL reaches 91%, 89% and 78% of the
maximum, with the top token's probability at 0.958, 0.947 and 0.872 against a uniform
0.083. Every arm collapses onto the single highest-scoring token.

**FINDING: `β` has to reach 100 before the KL is meaningfully held.** A 1000x
increase over the lesson's default takes KL to 20% of maximum -- and costs 35% of the
RM score. Nothing inside the stated sweep binds at all.

**MECHANISM: the penalty is removed by the advantage normalisation.** The reward is
`rm_score − β·KL`, but `KL` is computed from the whole distribution, so it depends on
the prompt and not on the sampled token: a 16-rollout batch holds at most **3**
distinct values of it against 12 for the RM score. The batch is then standardised by
`(r − mean)/sd`, which deletes what is common.

**FINDING: and the penalty only becomes visible once it is too late.** Early, when
the policy can still be steered, `sd(rm)` is **1662x** `sd(β·KL)` at `β = 0.1`. By
update 150 the policy has collapsed so hard that every rollout draws the same token
and `sd(rm)` is **0.0000** -- only then does the KL term dominate the advantage, with
nothing left to move.

Structure: `arm` runs the lesson's own `rlhf_loop`; `decompose` re-samples one batch
from a trained policy and splits the reward into its two parts.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "09-reward-modeling-rlhf"
UPDATES, SEEDS, RM_PAIRS = 150, 5, 600
STATED = (0.0, 0.1, 1.0)
WIDE = (0.0, 0.1, 1.0, 10.0, 100.0)
HACKING = 0.5


def arm(ref, model, beta, seed):
    """One `rlhf_loop` run: final RM score, final KL, and the top token probability."""
    theta, history = ref.rlhf_loop(model, updates=UPDATES, beta=beta, rng=random.Random(seed))
    tops = [max(ref.policy_probs(theta, p)) for p in range(len(ref.PROMPTS))]
    return {"rm": history[-1][1], "kl": history[-1][2], "top": statistics.fmean(tops)}


def cell(ref, model, beta):
    """One `β` averaged over the seeds."""
    rows = [arm(ref, model, beta, s) for s in range(SEEDS)]
    return {k: statistics.fmean(r[k] for r in rows) for k in ("rm", "kl", "top")}


def decompose(ref, model, beta, updates):
    """Re-sample one batch from a trained policy; split the reward into its two parts."""
    theta, _ = ref.rlhf_loop(model, updates=updates, beta=beta, rng=random.Random(0))
    reference = [[0.0 for _ in ref.VOCAB] for _ in ref.PROMPTS]
    rng = random.Random(123)
    scores, penalties = [], []
    for _ in range(16):
        prompt = rng.randrange(len(ref.PROMPTS))
        probs = ref.policy_probs(theta, prompt)
        scores.append(model.get(ref.VOCAB[ref.sample_token(probs, rng)], 0.0))
        penalties.append(beta * ref.kl(probs, ref.policy_probs(reference, prompt)))
    return (statistics.pstdev(scores), statistics.pstdev(penalties),
            len({round(p, 9) for p in penalties}))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    model = ref.train_rm(n_pairs=RM_PAIRS, rng=random.Random(42))
    return {"cells": {b: cell(ref, model, b) for b in WIDE},
            "early": decompose(ref, model, 0.1, 10),
            "late": decompose(ref, model, 0.1, UPDATES),
            "maxkl": math.log(len(ref.VOCAB)),
            "uniform": 1.0 / len(ref.VOCAB),
            "ceiling": max(model.values())}


def reads(result):
    """(the stated arms, their KL shares, and the two predicates the checks need)."""
    cells, maxkl = result["cells"], result["maxkl"]
    stated = [cells[b] for b in STATED]
    shares = [c["kl"] / maxkl for c in stated]
    early, late = result["early"], result["late"]
    return (stated, shares,
            min(shares) > HACKING and min(c["top"] for c in stated) > 0.8,
            late[0] < early[0] / 100 and late[1] > early[1])


def verify(result):
    cells, maxkl = result["cells"], result["maxkl"]
    early, late = result["early"], result["late"]
    stated, shares, hacks, too_late = reads(result)
    return [
        practice.Check(
            "ANSWER: all three, including beta = 1.0",
            hacks,
            f"the reference is uniform over 12 tokens, so KL is bounded by ln 12 = {maxkl:.4f}. "
            f"Final KL at beta = {STATED} is "
            + ", ".join(f"{c['kl']:.4f}" for c in stated) + " -- "
            + ", ".join(f"{100 * s:.0f}%" for s in shares)
            + " of that budget -- with the top token at "
            + ", ".join(f"{c['top']:.3f}" for c in stated)
            + f" against a uniform {result['uniform']:.3f}. Every arm collapses onto one token",
        ),
        practice.Check(
            "FINDING: beta has to reach 100 before the KL is meaningfully held",
            cells[100.0]["kl"] / maxkl < 0.25 < cells[1.0]["kl"] / maxkl,
            "KL as a share of its maximum across a wider sweep -- "
            + ", ".join(f"beta={b}: {100 * cells[b]['kl'] / maxkl:.0f}%" for b in WIDE)
            + f". A 1000x increase over the lesson's default takes it to "
            f"{100 * cells[100.0]['kl'] / maxkl:.0f}% and costs "
            f"{100 * (1 - cells[100.0]['rm'] / cells[0.0]['rm']):.0f}% of the RM score. Nothing "
            "inside the range the exercise states binds at all",
        ),
        practice.Check(
            "MECHANISM: the penalty is removed by the advantage normalisation",
            early[2] <= len(STATED) and early[0] > 100 * early[1],
            f"the reward is rm_score - beta*KL, but KL is computed from the whole distribution, "
            f"so it depends on the prompt and not on the sampled token: a 16-rollout batch holds "
            f"{early[2]} distinct values of it against 12 for the RM score. The batch is then "
            f"standardised by (r - mean)/sd, which deletes what is common -- and early in "
            f"training sd(rm) = {early[0]:.4f} against sd(beta*KL) = {early[1]:.6f}, a factor of "
            f"{early[0] / early[1]:.0f}",
        ),
        practice.Check(
            "FINDING: the penalty only becomes visible once it is too late",
            too_late,
            f"by update {UPDATES} the policy has collapsed so hard that every rollout draws the "
            f"same token: sd(rm) falls from {early[0]:.4f} at update 10 to {late[0]:.4f}, while "
            f"sd(beta*KL) rises from {early[1]:.6f} to {late[1]:.6f}. Only then does the penalty "
            "dominate the normalised advantage, and by then there is nothing left to steer -- "
            "the term meant to prevent the collapse becomes audible after it has finished",
        ),
        practice.Check(
            "FINDING: the RM score the hack reaches is the ceiling of the reward model",
            cells[0.0]["rm"] > 0.95 * result["ceiling"],
            f"the highest single-token weight the reward model learned is "
            f"{result['ceiling']:.4f}, and the beta=0 arm reaches {cells[0.0]['rm']:.4f}, "
            f"{100 * cells[0.0]['rm'] / result['ceiling']:.1f}% of it. That is what reward "
            "hacking looks like when the reward model is a bag of 12 weights: there is one best "
            "token and the policy finds it. Nothing here is subtle enough for beta to arbitrate",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
