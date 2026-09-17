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

**FINDING: the reward curve trends the wrong way, and the trend is the prompt
order.** The exercise predicts "steady reward improvement" at beta=0.02; the
linear fit is **-1.35e-03 per episode**, a decline, and the same decline in all
three runs. Freezing the policy entirely -- same prompts, same sampling stream,
the update computed and thrown away -- reproduces the reward sequence **bit for
bit**, so none of the decline comes from learning. Reversing the prompt order on
that same frozen policy moves the fit to **-2.25e-04**, six times smaller. The
per-prompt mean rewards run -0.12 to +0.18, and 20 episodes over 6 prompts ends
on the two lowest: the "reward curve" is the prompt schedule sampled 20 times.

Structure: `run` is one `ppo_training` at one beta from a fixed seed; `replay`
is the same loop through the lesson's own `generate_response`,
`compute_kl_divergence` and reward model, with a switch for whether the update
is applied, which is the control `ppo_training` cannot be asked for.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "07-rlhf"
SEED, EPISODES, EPOCHS, MAX_LEN = 7, 20, 10, 128
LR = 1.5e-5
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


def replay(ref, rm, prompts, freeze):
    """`ppo_training`'s loop through the lesson's own calls, optionally not applying the
    update -- the same randn draws are taken either way, so the sampling stream matches."""
    np.random.seed(SEED)
    policy, reference = ref.MiniGPT(**SHAPE), ref.MiniGPT(**SHAPE)
    ref.copy_model_weights(policy, reference)
    np.random.seed(SEED + 11)
    rewards = []
    for episode in range(EPISODES):
        prompt = [min(t, 252) for t in prompts[episode % len(prompts)].encode("utf-8")]
        response = ref.generate_response(policy, prompt, max_new_tokens=20,
                                         temperature=0.8, max_seq_len=MAX_LEN)
        ids = np.array(response[:MAX_LEN]).reshape(1, -1)
        rewards.append(float(rm.forward(ids)[0]))
        total = rewards[-1] - BETAS[1] * ref.compute_kl_divergence(policy.forward(ids),
                                                                  reference.forward(ids))
        for block in policy.blocks:
            steps = [LR * total * np.random.randn(*w.shape) * 0.01
                     for w in (block.ffn.W1, block.ffn.W2)]
            if not freeze:
                block.ffn.W1 += steps[0]
                block.ffn.W2 += steps[1]
    return rewards


def slope(rewards):
    return float(np.polyfit(range(len(rewards)), rewards, 1)[0])


def gaps(arms, first):
    """How far the three beta arms diverge from the first, on each recorded series."""
    return {"reward_gap": max(abs(x - y) for a in arms.values()
                              for x, y in zip(a["rewards"], first["rewards"])),
            "kl_gap": max(abs(x - y) for a in arms.values()
                          for x, y in zip(a["kls"], first["kls"])),
            "slope_gap": max(abs(a["slope"] - first["slope"]) for a in arms.values())}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rm = reward_model(ref)
    prompts = [pair["prompt"] for pair in ref.PREFERENCE_DATA]
    arms = {beta: run(ref, rm, prompts, beta) for beta in BETAS}
    first = arms[BETAS[0]]
    frozen = replay(ref, rm, prompts, freeze=True)
    return dict(gaps(arms, first),
                arms=arms,
                frozen_gap=max(abs(x - y) for x, y in zip(frozen, first["rewards"])),
                shuffled_slope=slope(replay(ref, rm, list(reversed(prompts)), freeze=True)),
                kl=(first["kls"][0], max(first["kls"])),
                prompts=len(prompts))


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
            "FINDING: the trend is the prompt order -- a frozen policy draws the same curve",
            first["slope"] < 0 and result["frozen_gap"] == 0.0
            and abs(result["shuffled_slope"]) < 0.5 * abs(first["slope"]),
            f"the exercise predicts 'steady reward improvement' at beta=0.02; the least-squares "
            f"fit is {first['slope']:+.2e} per episode, a decline, and the same decline in all "
            f"three runs. Freezing the policy entirely -- same prompts, same sampling stream, the "
            f"update computed and thrown away -- reproduces the reward sequence to "
            f"{result['frozen_gap']:.1e}, so none of the decline comes from learning. Reversing "
            f"the prompt order on that same frozen policy moves the fit to "
            f"{result['shuffled_slope']:+.2e}, {abs(first['slope'] / result['shuffled_slope']):.0f} "
            f"times smaller. The reward curve is {EPISODES} episodes cycling over "
            f"{result['prompts']} prompts, sampled in the order the loop happens to visit them",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
