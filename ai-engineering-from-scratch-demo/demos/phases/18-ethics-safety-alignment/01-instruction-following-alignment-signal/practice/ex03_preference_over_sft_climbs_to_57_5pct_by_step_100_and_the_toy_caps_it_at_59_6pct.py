"""Exercise 3 — preference over SFT climbs 50.2 / 51.1 / 53.6 / 57.5% and cannot pass 59.6%.

    Read Ouyang et al. (arXiv:2203.02155) Figure 1. Reproduce the
    labeler-preference curve by running PPO for 1, 5, 20, 100 steps and
    measuring preference against the SFT model.

Reading of the exercise: "preference" is the paper's win rate -- how often a
labeler prefers the policy's output to the SFT model's on the same prompt,
ties split evenly. In the toy the labeler is the Bradley-Terry rule
`stage2_reward_model` uses to label pairs, P(a beats b) = sigmoid(u_a - u_b)
on `labeler_true_utility()`, so the win rate is computed exactly over both
policies' action distributions rather than sampled. PPO is the shipped
beta = 0.1 run on seed 0. Figure 1's x-axis is model size, not PPO steps;
this is the step curve the exercise asks for.

**ANSWER: 50.2%, 51.1%, 53.6%, 57.5% at 1, 5, 20 and 100 steps.** The
curve rises monotonically and bends over: 300 steps gives 58.8%.

**FINDING: the toy caps preference over SFT at 59.6%.** Even a policy that
always plays B only wins 59.6% against SFT. SFT already plays B 63% of the
time, and those matchups are ties. The paper's large margins over SFT cannot
come out of a three-action bandit whose SFT already imitates the labeler.

**FINDING: against the untrained base the same policy wins 65.1%, near the
lesson's ~70% headline.** The base here is the reference's default
`Policy()`, uniform over A/B/C. The 100-step policy beats it 65.1% of the
time, against a ceiling of 67.2%. SFT alone beats it 57.6%, so SFT does most
of the work and PPO adds 7.5 points.

Structure: `win()` is the exact labeler win rate; `policy_after()` runs the
seeded pipeline with a `random.Random` swapped into the reference (restored
after).
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "01-instruction-following-alignment-signal"
STEPS, SEED, BETA = (1, 5, 20, 100, 300), 0, 0.1


def win(ref, p, q):
    """P(labeler prefers a ~ p over b ~ q) under the Bradley-Terry rule that labels RM pairs."""
    u = ref.labeler_true_utility()
    return round(sum(pa * qb / (1 + math.exp(u[b] - u[a]))
                     for a, pa in enumerate(p) for b, qb in enumerate(q)), 3)


def policy_after(ref, steps):
    """(SFT probs, PPO probs after `steps`) for the shipped beta = 0.1 run on SEED."""
    saved, ref.random = ref.random, random.Random(SEED)
    try:
        sft = ref.stage1_sft()
        pi = ref.stage3_ppo(sft, ref.stage2_reward_model(), beta=BETA, steps=steps)[0]
        return sft.probs(), pi.probs()
    finally:
        ref.random = saved


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = {n: policy_after(ref, n) for n in STEPS}
    sft, base, best = runs[1][0], ref.Policy().probs(), [0.0, 1.0, 0.0]
    return {
        "curve": {n: win(ref, pi, sft) for n, (_, pi) in runs.items()},
        "ceiling": win(ref, best, sft), "sft_b": round(sft[1], 2),
        "vs_base": win(ref, runs[100][1], base), "base_ceiling": win(ref, best, base),
        "sft_vs_base": win(ref, sft, base), "base": [round(p, 3) for p in base],
    }


def verify(result):
    curve = result["curve"]
    return [
        practice.Check(
            "ANSWER: 50.2%, 51.1%, 53.6%, 57.5% at 1, 5, 20 and 100 steps",
            curve == {1: 0.502, 5: 0.511, 20: 0.536, 100: 0.575, 300: 0.588}
            and list(curve.values()) == sorted(curve.values()),
            f"win rate over SFT by PPO steps: {curve}",
        ),
        practice.Check(
            "FINDING: the toy caps preference over SFT at 59.6%",
            result["ceiling"] == 0.596 and result["sft_b"] == 0.63
            and max(curve.values()) < result["ceiling"],
            f"always-B policy vs SFT: {result['ceiling']}; SFT plays B {result['sft_b']}",
        ),
        practice.Check(
            "FINDING: against the untrained base the same policy wins 65.1%",
            result["vs_base"] == 0.651 and result["base_ceiling"] == 0.672
            and result["sft_vs_base"] == 0.576 and result["base"] == [0.333] * 3
            and round(result["vs_base"] - result["sft_vs_base"], 3) == 0.075,
            f"100-step policy vs Policy() {result['base']}: {result['vs_base']}; ceiling "
            f"{result['base_ceiling']}; SFT vs base {result['sft_vs_base']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
