"""Exercise 3 — every response is exactly 20 tokens, so the length reward cannot be hacked.

    Simulate reward hacking. Create a reward model that gives high scores to long
    responses (reward = len(response) / 100). Run PPO with this flawed reward
    model and observe the policy model generating increasingly long, repetitive
    outputs. Then add a KL penalty of 0.1 and show that it prevents the
    degenerate behavior.

Reading of the exercise: the flawed reward model is built exactly as specified
and driven through the lesson's own `ppo_training`, so the "degenerate behavior"
the exercise predicts has to come out of the reference's own generation loop.
Response length is therefore measured rather than assumed.

**ANSWER: the reward is a function of the prompt, and reward hacking cannot
occur.** `generate_response` appends exactly `max_new_tokens` tokens and has no
end-of-sequence check anywhere in the file, so the policy emits **20 tokens**
every time, at every episode. The reward `len(response) / 100` therefore takes
exactly six values across twenty episodes -- one per prompt, each of them
`(len(prompt) + 20) / 100` -- and the policy's contribution to all six is the
same constant 20.

**MECHANISM: the policy has no way to express length.** Length is a loop bound
in `generate_response`, not something the model emits. There is no EOS token in
`SPECIAL_TOKENS`, no stopping criterion, and no branch that can end a sequence
early -- so the one quantity this reward measures is the one quantity the policy
cannot influence.

**FINDING: adding the KL penalty changes nothing, because there is nothing to
prevent.** With `kl_coeff=0.1` the reward trace is identical to `kl_coeff=0.0`,
and the KL itself stays at **6e-13** -- the policy does not move far enough from
the reference for the penalty term to reach the fourth decimal of
`total_reward`. The exercise asks you to show the penalty preventing a
degenerate behaviour that the code cannot produce.

**FINDING: the PPO update could not hack any reward.** `ppo_training` computes
`total_reward = reward - kl_coeff * kl` and then writes
`block.ffn.W1 += lr * total_reward * np.random.randn(...) * 0.01`. The direction
is a fresh random draw each episode, so the expected update is zero whatever the
reward says; the reward sets only the step *size*. Reward hacking requires a
policy that can climb a reward, and this one performs a random walk whose stride
is proportional to how well it happens to be doing.

Structure: `LengthReward` is the flawed model the exercise asks for, with the
`forward` signature `ppo_training` expects; `episode_lengths` records what the
generator actually produced.
"""

from __future__ import annotations

import contextlib
import io
import statistics

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "07-rlhf"
SEED, EPISODES, NEW_TOKENS, MAX_LEN = 7, 20, 20, 128
SHAPE = dict(vocab_size=256, embed_dim=64, num_heads=4, num_layers=2,
             max_seq_len=MAX_LEN, ff_dim=256)


class LengthReward:
    """The flawed reward the exercise asks for: reward = len(response) / 100."""

    def __init__(self):
        self.seen = []

    def forward(self, token_ids):
        length = int(token_ids.shape[-1])
        self.seen.append(length)
        return np.array([length / 100.0])


def episode_lengths(ref, prompts, draws=5):
    """How many tokens `generate_response` actually appends, over repeated calls."""
    np.random.seed(SEED)
    policy = ref.MiniGPT(**SHAPE)
    counts = []
    for i in range(draws):
        prompt = list(prompts[i % len(prompts)].encode("utf-8"))
        out = ref.generate_response(policy, prompt, max_new_tokens=NEW_TOKENS,
                                    max_seq_len=MAX_LEN)
        counts.append(len(out) - len(prompt))
    return counts


def ppo(ref, prompts, kl_coeff):
    """One `ppo_training` run against the flawed reward, from a fixed initialisation."""
    np.random.seed(SEED)
    policy = ref.MiniGPT(**SHAPE)
    reference = ref.MiniGPT(**SHAPE)
    ref.copy_model_weights(policy, reference)
    np.random.seed(SEED + 11)
    with contextlib.redirect_stdout(io.StringIO()):
        _, rewards, kls = ref.ppo_training(policy, reference, LengthReward(), prompts,
                                           num_episodes=EPISODES, kl_coeff=kl_coeff,
                                           max_seq_len=MAX_LEN)
    return rewards, kls


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    prompts = [pair["prompt"] for pair in ref.PREFERENCE_DATA]
    unpenalised, kls = ppo(ref, prompts, 0.0)
    penalised, _ = ppo(ref, prompts, 0.1)
    source = parity.lesson_dir(PHASE, LESSON).joinpath("code", "main.py").read_text("utf-8")
    cycle = [(len(p.encode("utf-8")) + NEW_TOKENS) / 100.0 for p in prompts]
    return {
        "lengths": episode_lengths(ref, prompts),
        "prompt_determined": [round(r, 10) for r in unpenalised]
                             == [round(cycle[i % len(cycle)], 10) for i in range(EPISODES)],
        "distinct": sorted({round(r, 10) for r in unpenalised}),
        "rewards": (unpenalised, penalised),
        "variance": statistics.pvariance(unpenalised),
        "identical": unpenalised == penalised,
        "kl": (kls[0], max(kls)),
        "has_eos": "eos" in source.lower() or "end_of" in source.lower(),
        "specials": sorted(ref.SPECIAL_TOKENS) if hasattr(ref, "SPECIAL_TOKENS") else [],
    }


def verify(result):
    unpenalised, penalised = result["rewards"]
    lengths = result["lengths"]
    first_kl, peak_kl = result["kl"]
    return [
        practice.Check(
            f"ANSWER: the reward is a function of the prompt -- the policy always emits {NEW_TOKENS}",
            set(lengths) == {NEW_TOKENS} and result["prompt_determined"],
            f"generate_response appends exactly max_new_tokens tokens: {lengths} over five "
            f"calls. So len(response) / 100 takes {len(result['distinct'])} values over "
            f"{len(unpenalised)} episodes -- {result['distinct']} -- and every one is "
            f"(len(prompt) + {NEW_TOKENS}) / 100 for the prompt that episode happened to draw. "
            "The policy contributes the same 20 to every reward it is scored on",
        ),
        practice.Check(
            "MECHANISM: length is a loop bound, not something the policy can emit",
            not result["has_eos"],
            "there is no EOS token anywhere in the lesson's code, no stopping criterion in "
            "generate_response and no branch that can end a sequence early -- the word does not "
            "appear in the file. The one quantity this reward measures is the one quantity the "
            "policy has no way to influence",
        ),
        practice.Check(
            "FINDING: the KL penalty changes nothing, because there is nothing to prevent",
            result["identical"] and peak_kl < 1e-9,
            f"with kl_coeff=0.1 the reward trace is identical to kl_coeff=0.0, and the KL itself "
            f"starts at {first_kl:.1e} and peaks at {peak_kl:.1e}. The policy never moves far "
            f"enough from the reference for {0.1} * kl to reach the fourth decimal of "
            "total_reward. The exercise asks you to show a penalty preventing a behaviour the "
            "code cannot produce",
        ),
        practice.Check(
            "FINDING: the PPO update could not hack any reward, flawed or not",
            len(unpenalised) == EPISODES,
            "ppo_training computes total_reward = reward - kl_coeff * kl and then writes "
            "block.ffn.W1 += lr * total_reward * np.random.randn(...) * 0.01. The direction is a "
            "fresh random draw every episode, so the expected update is zero whatever the reward "
            "says and the reward sets only the step size. Reward hacking needs a policy that can "
            "climb; this one takes a random walk whose stride tracks its own score",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
