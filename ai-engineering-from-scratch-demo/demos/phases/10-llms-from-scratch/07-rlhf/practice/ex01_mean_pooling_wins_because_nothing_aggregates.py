"""Exercise 1 — mean pooling wins 4-2, because there is no attention to rely on.

    Modify the reward model to use the mean of all hidden states instead of just
    the last position. Compare accuracy. The mean pooling approach gives every
    token equal weight, while the last-position approach relies on the causal
    attention to aggregate information. Test on the 6 preference pairs and
    report which approach scores higher accuracy.

Reading of the exercise: both poolings are scored on the *same* trained reward
model -- `train_reward_model` only updates `reward_head`, so swapping the
pooling changes the feature the head reads and nothing else, and a single
training run supports both arms. Accuracy is `r(preferred) > r(rejected)` over
the lesson's own 6 pairs, as the exercise says.

**ANSWER: mean pooling, 4 of 6 against 2 of 6.** Last-position scores below
chance, and not by accident: its mean margin is **-0.0383**, negative, so it
prefers the rejected response on average rather than merely guessing.

**MECHANISM: the exercise names the reason last-position loses.** "The
last-position approach relies on the causal attention to aggregate information"
-- and the attention here is at its random initialisation and stays there.
`train_reward_model` updates `rm.reward_head` and nothing else, so there is no
aggregation to rely on. Mean pooling wins because averaging 128 random
projections is a lower-variance summary than reading one of them: its margin
spread is **0.0081** against last-position's **0.0503**, six times tighter.

**FINDING: the head is trained against a feature the model does not use.** The
update multiplies the Bradley-Terry gradient by
`rm.ln_f.forward(rm.embedding.forward(ids))[:, -1, :]` -- the embedding passed
straight into the final LayerNorm, **skipping every transformer block** --
while `forward` scores `ln_f(blocks(embedding(x)))[-1]`. The two vectors have a
cosine of **0.50**. The head is being fitted to one representation and evaluated
on another.

**FINDING: the gradient also drops half of the Bradley-Terry term.** The
derivative is `(sigmoid(d) - 1) * (h_preferred - h_rejected)`; the code uses only
`h_preferred`, so every step pushes the head along the preferred feature and
raises the rejected response's score with it whenever the two are similar --
which, sharing a prompt, they are.

Structure: `pooled_reward` is the reward model's own forward pass with the
pooling swapped; `accuracy` runs it over the six pairs.
"""

from __future__ import annotations

import contextlib
import io
import statistics

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "07-rlhf"
SEED, EPOCHS, MAX_LEN = 1, 10, 128
SHAPE = dict(vocab_size=256, embed_dim=64, num_heads=4, num_layers=2,
             max_seq_len=MAX_LEN, ff_dim=256)


def trained(ref):
    """The lesson's own reward model after its own training run."""
    np.random.seed(SEED)
    model = ref.RewardModel(**SHAPE)
    np.random.seed(SEED + 50)
    with contextlib.redirect_stdout(io.StringIO()):
        model, _, accuracies = ref.train_reward_model(model, ref.PREFERENCE_DATA,
                                                      num_epochs=EPOCHS)
    return model, accuracies


def hidden(model, ids):
    """`RewardModel.forward` up to the final LayerNorm, all positions kept."""
    mask = np.triu(np.full((ids.shape[-1],) * 2, -1e9), k=1)
    x = model.embedding.forward(ids)
    for block in model.blocks:
        x = block.forward(x, mask)
    return model.ln_f.forward(x)


def pooled_reward(model, ref, prompt, text, pooling):
    ids = np.array(ref.tokenize_for_reward(prompt, text)[:MAX_LEN]).reshape(1, -1)
    states = hidden(model, ids)
    summary = states[:, -1, :] if pooling == "last" else states.mean(axis=1)
    return float((summary @ model.reward_head)[0])


def accuracy(model, ref, pooling):
    """Correct preferences and the margin distribution over the lesson's six pairs."""
    margins = [pooled_reward(model, ref, pair["prompt"], pair["preferred"], pooling)
               - pooled_reward(model, ref, pair["prompt"], pair["rejected"], pooling)
               for pair in ref.PREFERENCE_DATA]
    return sum(margin > 0 for margin in margins), margins


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    model, accuracies = trained(ref)
    arms = {p: accuracy(model, ref, p) for p in ("last", "mean")}
    ids = np.array(ref.tokenize_for_reward(ref.PREFERENCE_DATA[0]["prompt"],
                                           ref.PREFERENCE_DATA[0]["preferred"])).reshape(1, -1)
    shortcut = model.ln_f.forward(model.embedding.forward(ids))[:, -1, :].flatten()
    real = hidden(model, ids)[:, -1, :].flatten()
    np.random.seed(SEED)
    start = ref.RewardModel(**SHAPE)
    return {
        "pairs": len(ref.PREFERENCE_DATA),
        "correct": {p: c for p, (c, _) in arms.items()},
        "margin": {p: statistics.fmean(ms) for p, (_, ms) in arms.items()},
        "spread": {p: statistics.pstdev(ms) for p, (_, ms) in arms.items()},
        "trained_accuracy": accuracies[-1],
        "cosine": float(shortcut @ real / (np.linalg.norm(shortcut) * np.linalg.norm(real))),
        "blocks_frozen": all(np.array_equal(a.attn.W_q, b.attn.W_q)
                             for a, b in zip(start.blocks, model.blocks)),
        "head_moved": not np.array_equal(start.reward_head, model.reward_head),
    }


def verify(result):
    correct, margin, spread = result["correct"], result["margin"], result["spread"]
    pairs = result["pairs"]
    return [
        practice.Check(
            f"ANSWER: mean pooling, {correct['mean']} of {pairs} against {correct['last']}",
            correct["mean"] > correct["last"] and margin["last"] < 0,
            f"over the lesson's {pairs} preference pairs, mean pooling gets "
            f"{correct['mean']} right and last-position {correct['last']}. Last-position scores "
            f"below chance and not by accident: its mean margin is {margin['last']:+.4f}, "
            "negative, so it prefers the rejected response on average rather than guessing",
        ),
        practice.Check(
            "MECHANISM: there is no attention to aggregate anything -- it is never trained",
            result["blocks_frozen"] and result["head_moved"],
            "the exercise names the reason itself: last-position 'relies on the causal attention "
            "to aggregate information'. train_reward_model updates rm.reward_head and nothing "
            "else, so every attention matrix is bit-for-bit at its random initialisation. Mean "
            f"pooling wins because averaging {MAX_LEN} random projections is a lower-variance "
            f"summary than reading one: margin spread {spread['mean']:.4f} against "
            f"{spread['last']:.4f}, {spread['last'] / spread['mean']:.0f}x tighter",
        ),
        practice.Check(
            "FINDING: the head is fitted to a feature the forward pass does not compute",
            0.3 < result["cosine"] < 0.7,
            "the update multiplies the Bradley-Terry gradient by "
            "ln_f(embedding(ids))[:, -1, :] -- the embedding passed straight into the final "
            "LayerNorm, skipping every transformer block -- while forward() scores "
            f"ln_f(blocks(embedding(x)))[-1]. The two vectors have a cosine of "
            f"{result['cosine']:.2f}, so the head is fitted to one representation and evaluated "
            "on another",
        ),
        practice.Check(
            "FINDING: the gradient drops half of the Bradley-Terry term",
            result["trained_accuracy"] <= 0.5,
            "the derivative is (sigmoid(d) - 1) * (h_preferred - h_rejected) and the code uses "
            "only h_preferred, so every step pushes the head along the preferred feature and "
            "raises the rejected response's score with it whenever the two are similar -- "
            f"sharing a prompt, they are. Training ends at {result['trained_accuracy']:.0%} "
            f"accuracy on its own {pairs} pairs, which a coin beats",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
