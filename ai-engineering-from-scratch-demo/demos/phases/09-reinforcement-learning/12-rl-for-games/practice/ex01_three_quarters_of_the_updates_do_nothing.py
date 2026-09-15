"""Exercise 1 — it converges, and three quarters of the updates do nothing.

    **Easy.** Implement the GRPO bandit in `code/main.py`. Train on 2 prompts × 4
    answer tokens each. Converge in < 1,000 updates with `G=8`.

Reading of the exercise: the bandit is already implemented, so "implement" is read as
*run it and report what convergence costs*. Accuracy is computed from the policy
itself rather than from the lesson's 200-episode `evaluate`, since a softmax over four
answers gives the probability of being right exactly. And the exercise says two
prompts where `QUESTIONS` ships three; the shipped three are used and the discrepancy
recorded.

**ANSWER: yes -- 0.954 by 500 updates and 0.969 by 1,000, on every seed.** The
minimum across ten seeds at 1,000 updates is 0.966, so the bar is cleared with room.

**FINDING: two thirds of those updates produce exactly zero gradient.** GRPO standardises
rewards inside the group, so when all `G` samples earn the same reward the variance is
zero, every advantage is `0/1e-8 = 0`, and the update is a no-op.

**FINDING: and the rate climbs as the policy improves.** Zero-gradient updates run
39.8%, 75.2%, 75.6% and 78.2% across successive blocks of 500. The better the policy
gets, the more often all eight samples are correct -- GRPO's signal is group
disagreement, and success destroys it.

**FINDING: which is why the curve flattens rather than finishes.** Accuracy moves
0.954 -> 0.969 between updates 500 and 1,000, and the remaining 0.03 is being bought
by the quarter of updates that still see a mixed group.

**FINDING: the exercise's "2 prompts" is not the lesson's 3.** `QUESTIONS` carries
three entries, so the shipped bandit is 3 x 4 rather than the 2 x 4 described.

Structure: `accuracy` reads the probability of the correct answer straight off the
policy; `groups` replays the sampling to classify each update as mixed or degenerate.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "12-rl-for-games"
GROUP, SEEDS, BLOCK = 8, 10, 500
CHECKPOINTS = (100, 250, 500, 1_000)


def accuracy(ref, theta):
    """Mean probability the policy assigns to the correct answer, over the prompts."""
    return statistics.fmean(ref.policy_probs(theta, p)[ref.QUESTIONS[p]["correct"]]
                            for p in range(ref.N_PROMPTS))


def groups(ref, updates, seed=0):
    """Share of updates whose group is all-same-reward, by block of `BLOCK`."""
    rng = random.Random(seed)
    theta = [[0.0] * ref.N_ANSWERS for _ in range(ref.N_PROMPTS)]
    reference = [row[:] for row in theta]
    flat, blocks = [], []
    for step in range(updates):
        prompt = rng.randrange(ref.N_PROMPTS)
        probs = ref.policy_probs(theta, prompt)
        rewards = {ref.verify(prompt, ref.sample(probs, rng)) for _ in range(GROUP)}
        flat.append(len(rewards) == 1)
        ref.grpo_step(theta, reference, random.Random(step))
        if (step + 1) % BLOCK == 0:
            blocks.append(statistics.fmean(flat[-BLOCK:]))
    return statistics.fmean(flat), blocks


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    curve = {n: [accuracy(ref, ref.train_grpo(n, rng=random.Random(s))[0]) for s in range(SEEDS)]
             for n in CHECKPOINTS}
    overall, blocks = groups(ref, 2_000)
    return {"curve": {n: statistics.fmean(v) for n, v in curve.items()},
            "worst": {n: min(v) for n, v in curve.items()},
            "degenerate": overall, "blocks": blocks,
            "prompts": ref.N_PROMPTS, "answers": ref.N_ANSWERS}


def verify(result):
    curve, worst, blocks = result["curve"], result["worst"], result["blocks"]
    return [
        practice.Check(
            "ANSWER: 0.954 by 500 updates and 0.969 by 1,000, on every seed",
            worst[1_000] > 0.9 and curve[500] > 0.9,
            "mean probability of the correct answer, over "
            + f"{SEEDS} seeds, at {CHECKPOINTS} updates: "
            + ", ".join(f"{curve[n]:.4f}" for n in CHECKPOINTS)
            + f". The worst seed at 1,000 updates is {worst[1_000]:.4f}, so the exercise's "
            "'converge in < 1,000 updates with G=8' is met with room to spare",
        ),
        practice.Check(
            "FINDING: two thirds of those updates produce exactly zero gradient",
            0.4 < result["degenerate"] < 0.8,
            f"GRPO standardises rewards inside the group, so when all {GROUP} samples earn the "
            f"same reward the variance is zero, every advantage is 0/1e-8 = 0, and the update is "
            f"a no-op. Over 2,000 updates that happens {100 * result['degenerate']:.1f}% of the "
            "time -- the group is the only baseline GRPO has, and an agreeing group has none",
        ),
        practice.Check(
            "FINDING: and the rate climbs as the policy improves",
            blocks[-1] > blocks[0] + 0.25,
            "share of zero-gradient updates by block of "
            + f"{BLOCK}: " + ", ".join(f"{100 * b:.1f}%" for b in blocks)
            + ". The better the policy gets the more often all eight samples are correct, so "
            "GRPO's signal is group disagreement and success is what destroys it. At the start "
            "the degenerate groups are all-wrong; by the end they are all-right",
        ),
        practice.Check(
            "FINDING: which is why the curve flattens rather than finishes",
            curve[1_000] - curve[500] < curve[500] - curve[250],
            f"accuracy moves {curve[250]:.4f} -> {curve[500]:.4f} over the second 250 updates "
            f"and {curve[500]:.4f} -> {curve[1_000]:.4f} over the next 500 -- half the rate for "
            f"twice the budget. The remaining {1 - curve[1_000]:.3f} is being bought by the "
            "quarter of updates that still see a mixed group",
        ),
        practice.Check(
            "FINDING: the exercise's '2 prompts' is not the lesson's 3",
            result["prompts"] == 3,
            f"`QUESTIONS` carries {result['prompts']} entries -- 'what is 1+2', 'what is 3*3' and "
            f"'capital of France' -- so the shipped bandit is {result['prompts']} x "
            f"{result['answers']} rather than the 2 x {result['answers']} the exercise describes. "
            "Every number above is for the bandit the lesson actually ships",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
