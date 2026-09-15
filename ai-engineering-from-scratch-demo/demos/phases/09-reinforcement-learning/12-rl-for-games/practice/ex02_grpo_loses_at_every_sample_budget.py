"""Exercise 2 — GRPO loses at every sample budget, once you count samples.

    **Medium.** Plug in PPO (clipped) and vanilla REINFORCE. Compare sample
    efficiency and reward variance to GRPO on the same bandit.

Reading of the exercise: "sample efficiency" is read as accuracy per *sample drawn
from the policy*, not per update, because GRPO draws `G=8` samples per update and
REINFORCE draws one -- comparing per update hands GRPO eight times the data and calls
the result efficiency. Both are also compared per update, since that is the reading
the lesson's own `main` invites, and the two orderings disagree. PPO is added as the
clipped ratio on the same group, so all three differ only in how the advantage is
formed.

**ANSWER: at matched samples REINFORCE wins at every budget.** 0.893 against 0.460 at
400 samples, 0.983 against 0.875 at 1,600, 0.994 against 0.954 at 4,000 and 0.999
against 0.970 at 16,000.

**FINDING: per update the ordering flips, and that is the comparison the lesson
prints.** At 500 updates GRPO leads 0.954 to 0.923; at 2,000 REINFORCE has overtaken,
0.987 to 0.970. One axis has GRPO ahead early and behind late; the other has it behind
throughout.

**FINDING: PPO's clip never fires here.** On one pass over a fresh group the ratio is
exactly 1, and a step of `lr/G = 0.0125` cannot move it out of `[0.8, 1.2]`; even
reusing the group for four passes leaves the clip rate near zero. PPO on this bandit
is GRPO with extra arithmetic.

**FINDING: both learners go quiet, for opposite reasons.** GRPO's per-update reward
variance falls from 0.174 to 0.022 -- and exercise 1 measured what that costs. REINFORCE's
holds near 0.069, but `verify` returns 0 or 1 and `reinforce_step` uses the raw reward as
the advantage, so its gradient is exactly zero on every update that answers wrong. It can
only reward, never punish, mirroring lesson 06 where every reward was negative and the
same update could only punish.

Structure: `matched` runs each learner to a sample budget rather than an update count;
`ppo_step` is the clipped variant sharing GRPO's group and advantage.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "12-rl-for-games"
GROUP, SEEDS, LR, EPS = 8, 10, 0.1, 0.2
BUDGETS, UPDATES = (400, 1_600, 4_000, 16_000), (500, 2_000)


def accuracy(ref, theta):
    """Mean probability the policy assigns to the correct answer, over the prompts."""
    return statistics.fmean(ref.policy_probs(theta, p)[ref.QUESTIONS[p]["correct"]]
                            for p in range(ref.N_PROMPTS))


def ppo_sample(ref, theta, prompt, old, action, adv):
    """One clipped update; returns whether the clip fired."""
    probs = ref.policy_probs(theta, prompt)
    ratio = probs[action] / max(old[action], 1e-12)
    if (adv > 0 and ratio > 1 + EPS) or (adv < 0 and ratio < 1 - EPS):
        return True
    for i, p in enumerate(probs):
        theta[prompt][i] += (LR / GROUP) * ratio * adv * ((1.0 if i == action else 0.0) - p)
    return False


def ppo_step(ref, theta, rng, epochs=1):
    """GRPO's group and advantage, with the clipped ratio in place of the plain one."""
    prompt = rng.randrange(ref.N_PROMPTS)
    old = ref.policy_probs(theta, prompt)
    picks = [ref.sample(old, rng) for _ in range(GROUP)]
    rewards = [ref.verify(prompt, a) for a in picks]
    mean, sd = statistics.fmean(rewards), statistics.pstdev(rewards) + 1e-8
    advs = [(r - mean) / sd for r in rewards]
    fired = [ppo_sample(ref, theta, prompt, old, a, adv)
             for _ in range(epochs) for a, adv in zip(picks, advs)]
    return statistics.pvariance(rewards), statistics.fmean(fired)


def one_update(ref, kind, theta, reference, rng):
    """One update of `kind`; returns (reward spread or reward, clip rate or None)."""
    if kind == "grpo":
        mean, _kl = ref.grpo_step(theta, reference, rng)
        return mean * (1 - mean), None
    if kind == "ppo":
        return ppo_step(ref, theta, rng)
    return ref.reinforce_step(theta, rng), None


def run(ref, kind, updates, seed):
    """One learner for `updates` updates; returns the policy, its spreads and clip rates."""
    rng = random.Random(seed)
    theta = [[0.0] * ref.N_ANSWERS for _ in range(ref.N_PROMPTS)]
    reference = [row[:] for row in theta]
    rows = [one_update(ref, kind, theta, reference, rng) for _ in range(updates)]
    return theta, [r[0] for r in rows], [r[1] for r in rows if r[1] is not None]


def matched(ref, kind, samples, seed):
    """The same learner run to a sample budget: GRPO and PPO draw `GROUP` per update."""
    drawn = GROUP if kind in ("grpo", "ppo") else 1
    return accuracy(ref, run(ref, kind, samples // drawn, seed)[0])


def multi_clip(ref, seed, epochs=4, updates=500):
    """Clip rate when the same group is reused for `epochs` passes instead of one."""
    theta, rng = [[0.0] * ref.N_ANSWERS for _ in range(ref.N_PROMPTS)], random.Random(seed)
    return statistics.fmean(ppo_step(ref, theta, rng, epochs=epochs)[1] for _ in range(updates))


def by_samples(ref, kinds):
    """Accuracy at each sample budget, for each learner."""
    return {k: {b: statistics.fmean(matched(ref, k, b, s) for s in range(SEEDS))
                for b in BUDGETS} for k in kinds}


def by_updates(ref, kinds):
    """Accuracy at each update count, for each learner."""
    return {k: {u: statistics.fmean(accuracy(ref, run(ref, k, u, s)[0]) for s in range(SEEDS))
                for u in UPDATES} for k in kinds}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    kinds = ("grpo", "ppo", "reinforce")
    return {"samples": by_samples(ref, kinds), "updates": by_updates(ref, kinds),
            "spread": {k: run(ref, k, 2_000, 0)[1] for k in ("grpo", "reinforce")},
            "clips": run(ref, "ppo", 2_000, 0)[2],
            "multi": statistics.fmean(multi_clip(ref, s) for s in range(SEEDS))}


def verify(result):
    by_sample, by_update = result["samples"], result["updates"]
    grpo, rein = by_sample["grpo"], by_sample["reinforce"]
    spread, rewards = result["spread"]["grpo"], result["spread"]["reinforce"]
    early, late = statistics.fmean(spread[:200]), statistics.fmean(spread[-200:])
    flat = statistics.fmean(statistics.pvariance(rewards[i:i + 200])
                            for i in range(0, len(rewards) - 199, 200))
    single, silent = statistics.fmean(result["clips"]), 1 - statistics.fmean(rewards)
    return [
        practice.Check(
            "ANSWER: at matched samples REINFORCE wins at every budget",
            all(rein[b] > grpo[b] for b in BUDGETS),
            "accuracy by policy samples drawn -- "
            + ", ".join(f"{b:,}: REINFORCE {rein[b]:.3f} vs GRPO {grpo[b]:.3f}" for b in BUDGETS)
            + f". GRPO draws {GROUP} samples per update and REINFORCE one, so comparing per "
            "update hands GRPO 8x the data and calls it efficiency",
        ),
        practice.Check(
            "FINDING: per update the ordering flips, and that is what the lesson prints",
            by_update["grpo"][500] > by_update["reinforce"][500]
            and by_update["reinforce"][2_000] > by_update["grpo"][2_000],
            f"at 500 updates GRPO leads {by_update['grpo'][500]:.3f} to "
            f"{by_update['reinforce'][500]:.3f}; at 2,000 REINFORCE has overtaken, "
            f"{by_update['reinforce'][2_000]:.3f} to {by_update['grpo'][2_000]:.3f}. One axis has "
            "GRPO ahead early and behind late, the other has it behind throughout",
        ),
        practice.Check(
            "FINDING: PPO's clip never fires, so PPO is GRPO with extra arithmetic",
            single < 0.01 and result["multi"] < 0.05,
            f"PPO here is GRPO's group and advantage with the clipped ratio substituted. On one "
            f"pass over a fresh group the clip fires on {100 * single:.2f}% of samples: the ratio "
            f"starts at exactly 1 and the only thing moving it is a step of lr/G = {LR / GROUP}; "
            f"four passes over the same group takes it to {100 * result['multi']:.2f}%. At "
            f"{BUDGETS[-1]:,} samples the two reach {by_sample['ppo'][BUDGETS[-1]]:.3f} and "
            f"{grpo[BUDGETS[-1]]:.3f}, a difference of ordering rather than clipping",
        ),
        practice.Check(
            "FINDING: both learners go quiet, for opposite reasons",
            late < early / 2 and 0 < silent < 1,
            f"GRPO's per-update reward variance falls from {early:.3f} over the first 200 "
            f"updates to {late:.3f} over the last 200, and exercise 1 measured what that costs: "
            f"a zero-variance group is a zero gradient. REINFORCE's variance holds at "
            f"{flat:.3f}, but `verify` returns 0 or 1 and `reinforce_step` uses the raw reward "
            f"as the advantage, so its gradient is exactly zero on the {100 * silent:.1f}% of "
            "updates that answer wrong -- it can only reward, never punish, which mirrors "
            "lesson 06 where every reward was negative and the same update could only punish",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
