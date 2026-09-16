"""Exercise 5 — the reference that wins both of the exercise's questions wins them on noise.

    Compare DPO with different reference models. Instead of using the SFT
    checkpoint as the reference, try: (a) the base model (pre-SFT), (b) a
    checkpoint from epoch 1 of DPO, (c) an exponential moving average of the
    policy model. Report which reference produces the highest preference accuracy
    and the most stable training curve.

Reading of the exercise: each strategy gets its own DPO run, from identical
initial weights and the same RNG stream, so the reference is the only thing that
differs. The runs are driven one epoch at a time through the lesson's own
`dpo_train` because (b) and (c) change the reference *during* training: (b)
snapshots the policy after epoch 1, and (c) keeps a genuine exponential moving
average, updated after every epoch. "Training curve" is the per-example loss
trace `dpo_train` returns, and "stable" is its range across that trace.

**ANSWER: the epoch-1 checkpoint wins both questions at once.**

    reference            curve range   margin spread   accuracy
    SFT checkpoint         3.8e-06        2.7e-06        0.000
    base model             0.3164         0.2027         0.167
    epoch-1 checkpoint     3.1e-06        9.2e-07        0.833
    EMA of the policy      3.4e-06        2.2e-06        0.000

Highest accuracy **and** flattest curve, which is the combination the exercise
asks for -- on margins of order 1e-06 produced by a policy that moved 9e-07.

**FINDING: two references that agree to 1e-06 score 0.000 and 0.833.** The
epoch-1 checkpoint is the policy snapshotted after one epoch, and the policy
drifts **9.0e-07** in total, so it and the SFT checkpoint are the same weights to
six decimal places. Every margin against either is ~1e-06, so
`preferred_reward > rejected_reward` is a sign test on noise; five of six land
one way against one reference and none against the other.

**FINDING: only the base model produces a curve at all.** Its loss trace moves
over **0.3164**; the other three move 3e-06, because each of those references is
a copy of the policy and `pi_logprob - ref_logprob` is 0 for every response. The
exercise asks for the most stable curve and three of the four candidates are flat
lines at `log(2) = 0.6931`.

**FINDING: "most stable" and "most informative" point in opposite directions.**
The only arm whose margins distinguish the six pairs is the least stable one, and
it is least stable because its reference is a *different random initialisation*
-- the spread measures the gap between two random models, not anything DPO
learned.

Structure: `run_arm` is one DPO run under one reference strategy, epoch by
epoch; `blend` is the EMA update; `clone` makes the fresh copies each strategy
needs.
"""

from __future__ import annotations

import contextlib
import io
import statistics

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "08-dpo"
SEED, BETA, EPOCHS, DECAY = 3, 0.1, 3, 0.9
STRATEGIES = ("sft", "base", "epoch1", "ema")
SHAPE = dict(vocab_size=256, embed_dim=64, num_heads=4, num_layers=2,
             max_seq_len=128, ff_dim=256)


def models(ref):
    np.random.seed(SEED)
    policy = ref.MiniGPT(**SHAPE)
    np.random.seed(SEED + 96)
    reference = ref.MiniGPT(**SHAPE)
    ref.copy_model_weights(policy, reference)
    return policy, reference


def clone(ref, model, seed):
    np.random.seed(seed)
    copy = ref.MiniGPT(**SHAPE)
    ref.copy_model_weights(model, copy)
    return copy


def blend(shadow, policy, decay=DECAY):
    """The EMA update, over the only weights `dpo_train` moves."""
    for old, new in zip(shadow.blocks, policy.blocks):
        old.ffn.W1 = decay * old.ffn.W1 + (1 - decay) * new.ffn.W1
        old.ffn.W2 = decay * old.ffn.W2 + (1 - decay) * new.ffn.W2


def logprob(ref, model, pair, key):
    return float(ref.compute_sequence_log_prob(
        model, ref.tokenize_sequence(pair["prompt"]), ref.tokenize_sequence(pair[key])))


def margin(ref, policy, reference, pair):
    """The implicit reward margin `dpo_loss` reports for one pair against this reference."""
    _, metrics = ref.dpo_loss(logprob(ref, policy, pair, "preferred"),
                              logprob(ref, policy, pair, "rejected"),
                              logprob(ref, reference, pair, "preferred"),
                              logprob(ref, reference, pair, "rejected"), BETA)
    return metrics["reward_margin"]


def run_arm(ref, strategy):
    """One DPO run under one reference strategy, driven an epoch at a time."""
    policy, sft = models(ref)
    reference = clone(ref, policy, SEED + 39) if strategy != "base" else ref.MiniGPT(**SHAPE)
    shadow = clone(ref, policy, SEED + 400)
    curve = []
    np.random.seed(SEED + 2)
    for epoch in range(EPOCHS):
        with contextlib.redirect_stdout(io.StringIO()):
            policy, losses, _ = ref.dpo_train(policy, reference, ref.PREFERENCE_DATA,
                                              num_epochs=1, beta=BETA)
        curve.extend(losses)
        if strategy == "ema":
            blend(shadow, policy)
            reference = shadow
        elif strategy == "epoch1" and epoch == 0:
            reference = clone(ref, policy, SEED + 77)
    return {"curve_range": max(curve) - min(curve), "final_loss": curve[-1],
            "margin_spread": statistics.pstdev(
                margin(ref, policy, reference, p) for p in ref.PREFERENCE_DATA),
            "accuracy": ref.evaluate_preference_accuracy(policy, reference,
                                                         ref.PREFERENCE_DATA, BETA),
            "drift": float(np.abs(policy.blocks[0].ffn.W1 - sft.blocks[0].ffn.W1).max())}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {"arms": {name: run_arm(ref, name) for name in STRATEGIES},
            "pairs": len(ref.PREFERENCE_DATA)}


def row(arms, field, fmt):
    return ", ".join(f"{name} {format(arm[field], fmt)}" for name, arm in arms.items())


def peaks(arms, copies):
    """The aggregates the checks compare, so verify() reads as four claims."""
    return {"best_accuracy": max(a["accuracy"] for a in arms.values()),
            "flattest": min(a["curve_range"] for a in arms.values()),
            "copy_drift": max(a["drift"] for a in copies),
            "copy_range": max(a["curve_range"] for a in copies),
            "copy_spread": max(a["margin_spread"] for a in copies),
            "spreads": ", ".join(f"{a['margin_spread']:.1e}" for a in copies)}


def verify(result):
    arms = result["arms"]
    base, sft, epoch1 = arms["base"], arms["sft"], arms["epoch1"]
    copies = [arms[name] for name in ("sft", "epoch1", "ema")]
    top = peaks(arms, copies)
    return [
        practice.Check(
            "ANSWER: the epoch-1 checkpoint wins both questions at once, on 1e-06 margins",
            epoch1["accuracy"] == top["best_accuracy"]
            and epoch1["curve_range"] == top["flattest"],
            "the accuracies are " + row(arms, "accuracy", ".3f")
            + " and the loss-trace ranges " + row(arms, "curve_range", ".1e")
            + f". The epoch-1 reference has the highest accuracy and the flattest curve, which is "
            f"the combination the exercise asks for -- on margins whose spread is "
            f"{epoch1['margin_spread']:.1e}, produced by a policy that moved "
            f"{epoch1['drift']:.1e} in total",
        ),
        practice.Check(
            "FINDING: two references that agree to 1e-06 score 0.000 and 0.833",
            abs(sft["accuracy"] - epoch1["accuracy"]) > 0.5 and top["copy_drift"] < 1e-5,
            f"the epoch-1 checkpoint is the policy snapshotted after one epoch, and the policy "
            f"drifts {top['copy_drift']:.1e} in total, so it and the SFT checkpoint are the same "
            f"weights to six decimal places. Every margin against either is of order 1e-06, so "
            f"preferred_reward > rejected_reward is a sign test on noise: five of "
            f"{result['pairs']} land one way ({epoch1['accuracy']:.3f}) and none the other "
            f"({sft['accuracy']:.3f})",
        ),
        practice.Check(
            "FINDING: only the base model produces a curve at all",
            base["curve_range"] > 0.01 and top["copy_range"] < 1e-4,
            "across the per-example loss trace the range is " + row(arms, "curve_range", ".1e")
            + f". Three of the four references are copies of the policy, so the loss is "
            f"log(2) = {sft['final_loss']:.4f} at every step. The exercise asks for the most "
            "stable curve and three of the candidates are flat lines",
        ),
        practice.Check(
            "FINDING: most stable and most informative point in opposite directions",
            base["margin_spread"] > 1000 * top["copy_spread"],
            f"the only arm whose margins distinguish the six pairs is the least stable one: "
            f"{base['margin_spread']:.4f} against " + top["spreads"]
            + ". And it is least stable because its reference is a different random "
            "initialisation, so that spread is the gap between two random models",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
