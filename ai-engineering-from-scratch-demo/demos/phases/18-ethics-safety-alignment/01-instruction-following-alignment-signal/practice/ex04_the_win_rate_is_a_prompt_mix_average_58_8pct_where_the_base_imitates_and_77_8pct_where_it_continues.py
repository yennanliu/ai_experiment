"""Exercise 4 — the win rate is a prompt-mix average: 58.8% where the base imitates, 77.8% where it continues.

    The paper's Section 4.3 reports a 1.3B InstructGPT beats 175B GPT-3 about
    70% of the time. Why would the ratio be higher on hidden production
    prompts than on the labeler's own prompts?

Reading of the exercise: a win rate is an average over prompts of a
per-prompt win rate, so two prompt sets can only differ through their mix.
The lesson's own Problem section names the prompt kind where a base model
fails -- asked an instruction, it "continues" with another prompt -- so the
model has two kinds. On completion-style prompts (the few-shot kind labelers
were asked to write) the base does as well as SFT. On instruction prompts
it continues the text, i.e. plays the labeler's least-liked action. The
aligned policy is the shipped beta = 0.1, 300-step PPO run on seed 0, and
"beats" is the labeler's Bradley-Terry preference on `labeler_true_utility()`.

**ANSWER: because production prompts are more often plain instructions,
which is where the base model fails.** The aligned policy wins 58.8% of
completion-style prompts and 77.8% of instruction prompts, and the mix is
linear between the two:

| instruction share | 0% | 25% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|---:|
| win rate | 58.8% | 63.6% | 68.3% | 73.1% | 77.8% |

The aggregate crosses 70% at a 59% instruction share. Every prompt set with
more instructions than the labelers' set gets a higher ratio. Customers
wrote their API prompts for an instruction-following model, so production
traffic carries a higher instruction share.

**FINDING: the lesson's toy cannot show this at all.** No stage of the
reference takes a prompt: `labeler_true_utility()` has no parameters and the
three stage functions take only sample counts, a policy, a reward and
hyperparameters. The "200 prompts" and "500 pairwise rankings" are draws of
one context, so any two prompt sets give the same ratio in the reference.

Structure: `per_kind()` computes the two win rates with the exact labeler
rule; `mix()` is the linear average. The pipeline runs on a seeded
`random.Random` swapped into the reference (restored after).
"""

from __future__ import annotations

import inspect
import math
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "01-instruction-following-alignment-signal"
SHARES, TARGET = (0.0, 0.25, 0.5, 0.75, 1.0), 0.70
STAGES = ("labeler_true_utility", "stage1_sft", "stage2_reward_model", "stage3_ppo")


def win(u, p, q):
    return sum(pa * qb / (1 + math.exp(u[b] - u[a])) for a, pa in enumerate(p) for b, qb in enumerate(q))


def per_kind(ref):
    """(win rate where base = SFT, win rate where base continues) for the shipped policy."""
    saved, ref.random = ref.random, random.Random(0)
    try:
        sft = ref.stage1_sft()
        pi = ref.stage3_ppo(sft, ref.stage2_reward_model(), beta=0.1)[0].probs()
    finally:
        ref.random = saved
    u = ref.labeler_true_utility()
    continues = [float(a == u.index(min(u))) for a in range(3)]
    return win(u, pi, sft.probs()), win(u, pi, continues)


def mix(imitate, cont, share):
    return round((1 - share) * imitate + share * cont, 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    imitate, cont = per_kind(ref)
    params = {s: list(inspect.signature(getattr(ref, s)).parameters) for s in STAGES}
    return {
        "kinds": (round(imitate, 3), round(cont, 3)),
        "table": {s: mix(imitate, cont, s) for s in SHARES},
        "crossing": round((TARGET - imitate) / (cont - imitate), 2), "params": params,
    }


def verify(result):
    params = result["params"]
    return [
        practice.Check(
            "ANSWER: production prompts are more often instructions, where the base fails",
            result["kinds"] == (0.588, 0.778) and result["crossing"] == 0.59
            and result["table"] == {0.0: 0.588, 0.25: 0.636, 0.5: 0.683, 0.75: 0.731, 1.0: 0.778},
            f"win rate (base imitates, base continues): {result['kinds']}; by instruction "
            f"share {result['table']}; 70% at share {result['crossing']}",
        ),
        practice.Check(
            "FINDING: the lesson's toy cannot show this at all",
            params == {"labeler_true_utility": [], "stage1_sft": ["n_demos"],
                       "stage2_reward_model": ["n_pairs", "bias"],
                       "stage3_ppo": ["sft", "reward", "beta", "steps", "batch", "lr"]},
            f"parameters of every stage: {params}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
