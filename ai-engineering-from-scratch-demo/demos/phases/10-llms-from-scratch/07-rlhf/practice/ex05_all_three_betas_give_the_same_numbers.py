"""Exercise 5 — the three beta runs are identical, and the KL never leaves 1e-13.

    Compare different KL coefficients. Run PPO with beta=0.001 (too low, reward
    hacking), beta=0.02 (standard), and beta=0.5 (too high, no learning). Plot
    the reward curve and KL curve for each. The beta=0.02 run should show steady
    reward improvement with bounded KL.

Reading of the exercise: the three runs use the lesson's own `ppo_training`
against its own trained reward model, from one fixed initialisation and one
fixed RNG stream, so the only difference between them is `kl_coeff`. "Plot the
reward curve and KL curve" is read as recording both and testing the three
predictions attached to them.

**ANSWER: the three runs agree to 1e-12 at every episode.** Same reward, same
KL, same linear fit. beta=0.001, beta=0.02 and beta=0.5 produce one curve, not
three, so none of the exercise's labels -- "too low, reward hacking",
"standard", "too high, no learning" -- distinguishes anything.

**MECHANISM: `beta` multiplies a number that is 1e-13.** `total_reward = reward
- kl_coeff * kl`, and the KL between policy and reference starts at exactly
**0.0** -- `copy_model_weights` makes them the same model -- and peaks at
**6e-13**. Even at beta=0.5 the penalty term is 3e-13 against a reward of order
0.2, so it cannot reach the fourth decimal of `total_reward`, let alone change
the update.

**MECHANISM: the policy never moves, so the KL cannot grow.** The update is
`lr * total_reward * np.random.randn(...) * 0.01` with `lr = 1.5e-5` and a
reward around 0.2 -- about 3e-8 per weight per episode, against weights of scale
0.02. Twenty episodes of that leave the policy within 1e-6 of the reference, and
the KL is quadratic in the displacement.

**FINDING: the reward curve trends the wrong way, and it is not learning.** The
exercise predicts "steady reward improvement" at beta=0.02; the linear fit is
**-1.4e-03 per episode**, a decline, and the same decline in all three runs. So
whatever produces it is not `beta`, and cannot be the policy, which is shared.
It is the prompt cycling: 20 episodes over 6 prompts, with the reward depending
on which prompt is up and not on what the policy did.

Structure: `run` is one `ppo_training` at one beta from a fixed seed; `slope` is
the least-squares trend the exercise asks to plot.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "07-rlhf"
SEED, EPISODES, EPOCHS, MAX_LEN = 7, 20, 10, 128
BETAS = (0.001, 0.02, 0.5)
SHAPE = dict(vocab_size=256, embed_dim=64, num_heads=4, num_layers=2,
             max_seq_len=MAX_LEN, ff_dim=256)


def reward_model(ref):
    np.random.seed(1)
    model = ref.RewardModel(**SHAPE)
    np.random.seed(51)
    with contextlib.redirect_stdout(io.StringIO()):
        model, _, _ = ref.train_reward_model(model, ref.PREFERENCE_DATA, num_epochs=EPOCHS)
    return model


def run(ref, rm, prompts, beta):
    """One `ppo_training` run at one KL coefficient, from a fixed initialisation."""
    np.random.seed(SEED)
    policy = ref.MiniGPT(**SHAPE)
    reference = ref.MiniGPT(**SHAPE)
    ref.copy_model_weights(policy, reference)
    np.random.seed(SEED + 11)
    with contextlib.redirect_stdout(io.StringIO()):
        trained, rewards, kls = ref.ppo_training(policy, reference, rm, prompts,
                                                 num_episodes=EPISODES, kl_coeff=beta,
                                                 max_seq_len=MAX_LEN)
    return {"rewards": rewards, "kls": kls,
            "slope": float(np.polyfit(range(len(rewards)), rewards, 1)[0]),
            "drift": float(np.abs(trained.blocks[0].ffn.W1 - reference.blocks[0].ffn.W1).max())}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rm = reward_model(ref)
    prompts = [pair["prompt"] for pair in ref.PREFERENCE_DATA]
    arms = {beta: run(ref, rm, prompts, beta) for beta in BETAS}
    first = arms[BETAS[0]]
    return {
        "arms": arms,
        "reward_gap": max(abs(x - y) for a in arms.values()
                          for x, y in zip(a["rewards"], first["rewards"])),
        "kl_gap": max(abs(x - y) for a in arms.values()
                      for x, y in zip(a["kls"], first["kls"])),
        "slope_gap": max(abs(a["slope"] - first["slope"]) for a in arms.values()),
        "kl": (first["kls"][0], max(first["kls"])),
        "prompts": len(prompts),
    }


def verify(result):
    arms = result["arms"]
    first = arms[BETAS[0]]
    initial_kl, peak_kl = result["kl"]
    penalty = max(BETAS) * peak_kl
    return [
        practice.Check(
            "ANSWER: the three runs agree to 1e-12 at every episode -- one curve, not three",
            result["reward_gap"] < 1e-12 and result["kl_gap"] < 1e-12,
            "beta = " + ", ".join(str(b) for b in BETAS)
            + f" give rewards that differ by at most {result['reward_gap']:.1e} and KLs by at "
            f"most {result['kl_gap']:.1e} across all {EPISODES} episodes, with linear fits "
            + ", ".join(f"{b} {a['slope']:+.2e}" for b, a in arms.items())
            + f" agreeing to {result['slope_gap']:.1e}. None of the exercise's three labels -- "
            "too low, standard, too high -- distinguishes anything",
        ),
        practice.Check(
            "MECHANISM: beta multiplies a number that never exceeds 1e-12",
            initial_kl == 0.0 and peak_kl < 1e-11,
            f"total_reward = reward - kl_coeff * kl, and the KL starts at exactly {initial_kl:.1f} "
            f"-- copy_model_weights makes the policy and the reference the same model -- and "
            f"peaks at {peak_kl:.1e}. Even at beta={max(BETAS)} the penalty term is "
            f"{penalty:.1e} against a reward of order {abs(first['rewards'][0]):.1f}, so it "
            "cannot reach the fourth decimal of total_reward, let alone change the update",
        ),
        practice.Check(
            "MECHANISM: the policy never moves, so the KL has nothing to measure",
            first["drift"] < 1e-5,
            f"the update is lr * total_reward * np.random.randn(...) * 0.01 with lr = 1.5e-5 and "
            f"a reward around 0.2 -- of order 3e-8 per weight per episode, against weights of "
            f"scale 0.02. After {EPISODES} episodes the largest single weight has moved "
            f"{first['drift']:.1e}, and KL is quadratic in that displacement",
        ),
        practice.Check(
            "FINDING: the reward trends the wrong way, and the trend is the prompt cycling",
            first["slope"] < 0 and result["slope_gap"] < 1e-12,
            f"the exercise predicts 'steady reward improvement' at beta=0.02. The least-squares "
            f"fit is {first['slope']:+.2e} per episode -- a decline, and the same decline in all "
            f"three runs, so whatever produces it is not beta and cannot be the policy, which "
            f"all three share. It is {EPISODES} episodes cycling over {result['prompts']} "
            "prompts with a reward that depends on which prompt is up and not on what the "
            "policy did",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
