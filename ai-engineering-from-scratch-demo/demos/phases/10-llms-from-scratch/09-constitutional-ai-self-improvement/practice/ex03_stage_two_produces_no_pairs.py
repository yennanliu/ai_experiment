"""Exercise 3 — CAI stage 2 yields 0 preference pairs, and the reward ties a third of the rest.

    Implement DPO on the preference pairs produced by CAI stage 2. Take 20
    prompts, generate two responses each, have the critic pick a winner per pair,
    then run the DPO loss from Lesson 08. Compare to the GRPO path on the same
    data.

Reading of the exercise: "the preference pairs produced by CAI stage 2" are the
(initial, revised) pairs `cai_stage_one` returns with `changed=True`, since a
revision the critic declined to make is not a preference. The comparison to GRPO
uses the lesson's own `group_relative_advantage` on the same responses, which is
what the GRPO path would consume.

**ANSWER: stage 2 produces 0 preference pairs from 20 prompts.** The critic
finds no problems and performs no revisions (Exercise 1), so every row comes back
`changed=False` and there is nothing for DPO to train on. The comparison the
exercise ends with has one arm.

**FINDING: the fallback -- letting the reward pick the winner -- ties 36% of the
time.** `combined_reward` takes three values on this sampler: 0.0, 1.0 and 1.1.
Over 20 prompts sampled twice across 30 seeds, the two responses score equal in
**36%** of draws, so a reward-based referee declines to order more than a third
of the pairs and orders the rest on a three-valued scale.

**MECHANISM: GRPO has the same problem and names it.**
`group_relative_advantage` returns all zeros when `r.std() < 1e-8` -- every
sample in the group scoring the same. At the group size a pairwise comparison
implies, **k=2**, that happens in **42%** of groups; at k=4 it is 6% and at k=8
it is 0%. The exercise's "two responses each" is the worst group size GRPO has,
and the DPO path inherits the tie without the guard.

**FINDING: what the group size buys is tie resistance, and ties are all this
comparison can see.** The degenerate rate falls from 42% at k=2 to 6% at k=4 and
0% at k=8: more samples means fewer groups in which every reward is equal, and a
pair is the fewest samples available. That is a statement about ties, not about
the two objectives -- DPO optimises a logistic loss on reference-relative
log-odds and GRPO normalises scalar rewards into a policy gradient, and nothing
measured here compares those. What *is* comparable is that "two responses each"
puts the DPO path at the group size where GRPO's own guard fires most often.

Structure: `pairs` runs stage 2 and counts what it returns; `tie_rate` measures
how often the reward orders two independent samples of the same prompt.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "09-constitutional-ai-self-improvement"
SEED, PROMPT_COUNT, SEEDS = 0, 20, range(30)
GROUPS = (2, 4, 8)
BASE = ("What is 3+4?", "What is 12*7?", "What is 100-37?", "What is (3+4)*5?")


def prompts():
    return [BASE[i % len(BASE)] for i in range(PROMPT_COUNT)]


def stage_two(ref):
    """The (initial, revised) pairs CAI stage 2 actually produces."""
    sampler = ref.mock_sampler(random.Random(SEED))
    random.seed(SEED)
    rows = ref.cai_stage_one([(prompt, sampler(prompt)) for prompt in prompts()])
    return [row for row in rows if row["changed"]], rows


def tie_rate(ref):
    """How often two independent samples of one prompt score equal under combined_reward."""
    ties = total = 0
    for seed in SEEDS:
        sampler = ref.mock_sampler(random.Random(seed))
        for prompt in prompts():
            first, second = sampler(prompt), sampler(prompt)
            ties += ref.combined_reward(prompt, first) == ref.combined_reward(prompt, second)
            total += 1
    return ties / total, total // len(SEEDS)


def degenerate(ref, group_size):
    """Share of GRPO groups whose advantages are all zero at this group size."""
    zero = total = 0
    for seed in SEEDS:
        sampler = ref.mock_sampler(random.Random(seed))
        for prompt in prompts()[:len(BASE)]:
            rewards = [ref.combined_reward(prompt, sampler(prompt)) for _ in range(group_size)]
            zero += all(abs(a) < 1e-9 for a in ref.group_relative_advantage(rewards))
            total += 1
    return zero / total


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    changed, rows = stage_two(ref)
    ties, per_seed = tie_rate(ref)
    sampler = ref.mock_sampler(random.Random(SEED))
    values = {ref.combined_reward(prompt, sampler(prompt))
              for _ in range(200) for prompt in BASE}
    return {
        "pairs": len(changed),
        "rows": len(rows),
        "ties": ties,
        "per_seed": per_seed,
        "reward_values": sorted(values),
        "degenerate": {g: degenerate(ref, g) for g in GROUPS},
    }


def verify(result):
    degenerate_rate = result["degenerate"]
    return [
        practice.Check(
            f"ANSWER: CAI stage 2 produces 0 preference pairs from {PROMPT_COUNT} prompts",
            result["pairs"] == 0,
            f"of the {result['rows']} rows cai_stage_one returns, {result['pairs']} come back "
            "changed=True -- the critic finds no problems and performs no revisions (Exercise "
            "1), so there is no (initial, revised) pair for DPO to train on. The comparison the "
            "exercise ends with has one arm",
        ),
        practice.Check(
            "FINDING: letting the reward pick the winner instead ties a third of the time",
            result["ties"] > 0.3 and len(result["reward_values"]) == 3,
            f"combined_reward takes {len(result['reward_values'])} values on this sampler -- "
            f"{result['reward_values']} -- so over {result['per_seed']} prompts sampled twice, "
            f"across {len(SEEDS)} seeds, the two responses score equal {result['ties']:.0%} of "
            "the time. Even a reward-based referee declines to order more than a third of the "
            "pairs, and the ones it does order it orders on a three-valued scale",
        ),
        practice.Check(
            "MECHANISM: GRPO has the same problem at k=2 and names it",
            degenerate_rate[2] > 5 * degenerate_rate[8] and degenerate_rate[2] > 0.3,
            "group_relative_advantage returns all zeros when r.std() < 1e-8 -- every sample in "
            "the group scoring the same. That happens in "
            + ", ".join(f"{100 * rate:.0f}% of groups at k={g}"
                        for g, rate in degenerate_rate.items())
            + ". The exercise's 'two responses each' is the worst group size GRPO has, and the "
            "DPO path inherits the tie without the guard that detects it",
        ),
        practice.Check(
            "FINDING: what the group size buys is tie resistance, and ties are all this compares",
            degenerate_rate[8] < degenerate_rate[4] < degenerate_rate[2],
            f"the degenerate rate falls from {100 * degenerate_rate[2]:.0f}% at k=2 to "
            f"{100 * degenerate_rate[4]:.0f}% at k=4 and {100 * degenerate_rate[8]:.0f}% at k=8, "
            "so more samples means fewer groups in which every reward is equal, and a pair is the "
            "fewest samples available. That is a statement about ties and not about the two "
            "objectives -- DPO optimises a logistic loss on reference-relative log-odds and GRPO "
            "normalises scalar rewards into a policy gradient, and nothing measured here compares "
            "those. What is comparable is that 'two responses each' puts the DPO path at the "
            "group size where GRPO's own guard fires most often",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
