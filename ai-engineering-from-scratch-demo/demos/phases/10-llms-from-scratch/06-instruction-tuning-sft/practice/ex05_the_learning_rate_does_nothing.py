"""Exercise 5 — the loss is 5.5011 at every learning rate, because the update is noise.

    Compare learning rates. Train the same model three times with lr=1e-4,
    lr=2e-5, and lr=1e-6. Plot the loss curves. The 1e-4 run should show rapid
    initial descent but higher final loss (overfitting). The 1e-6 run should
    barely move. The 2e-5 run should be the sweet spot.

Reading of the exercise: the three runs are the lesson's own `sft_train` on its
own `INSTRUCTION_DATA`, from the same initialisation, with only `lr` changed --
and a fourth run at 1e-3 is added, a further order of magnitude out, because
three points that agree tell you less than four. "Plot the loss curves" is read
as *record them and test the three predictions the exercise makes about them*,
which a plot cannot do.

**ANSWER: all three runs give 5.5011.** First step and last step, identical to
four decimal places, at 1e-6, 2e-5, 1e-4 and 1e-3 alike. The loss does not
respond to the learning rate across four orders of magnitude, so none of the
three predictions occurs: no rapid initial descent, no overfitting, no sweet
spot, and "the 1e-6 run should barely move" is right for a reason that applies
equally to the other three.

**MECHANISM: `sft_train` throws its gradient away.** The loop computes
`dlogits`, masks it, divides by the response-token count -- and then updates
with

    block.ffn.W1 -= lr * np.random.randn(*block.ffn.W1.shape) * 0.01

`dlogits` appears nowhere on the right-hand side. The update is Gaussian noise
scaled by the learning rate, so `lr` sets the size of a random walk and nothing
else.

**PROOF: training on a different dataset produces bit-identical weights.** Run
`sft_train` on `INSTRUCTION_DATA` and on eight examples of `"zzzz"` and
`"qqqq"`, from the same seed, and every FFN weight matches exactly. A trainer
whose output does not depend on its input is not training.

**FINDING: even the random walk touches two arrays.** `ffn.W1` and `ffn.W2`
move; `W_q`, `W_out`, `ln1`, `ln2`, `token_embed` and `pos_embed` are unchanged
-- so the attention stack and the embedding table are frozen exactly as they are
in lesson 04, and here nothing replaces them.

Structure: `run` is one `sft_train` at one learning rate from a fixed
initialisation; `weights` pulls the named arrays out of a model for an identity
comparison.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "06-instruction-tuning-sft"
SEED, EPOCHS = 1, 3
RATES = (1e-6, 2e-5, 1e-4, 1e-3)
SHAPE = dict(vocab_size=256, embed_dim=64, num_heads=4, num_layers=2,
             max_seq_len=64, ff_dim=256)
DECOY = [{"instruction": "zzzz " * 4, "response": "qqqq " * 6} for _ in range(8)]
NAMES = ("ffn.W1", "ffn.W2", "W_q", "W_out", "ln1", "ln2", "token_embed", "pos_embed")


def weights(model):
    """The named arrays, for a bit-for-bit comparison between two models."""
    block = model.blocks[0]
    return {"ffn.W1": block.ffn.W1, "ffn.W2": block.ffn.W2, "W_q": block.attn.W_q,
            "W_out": block.attn.W_out, "ln1": block.ln1.gamma, "ln2": block.ln2.gamma,
            "token_embed": model.embedding.token_embed,
            "pos_embed": model.embedding.pos_embed}


def run(ref, dataset, lr):
    """One `sft_train` from a fixed initialisation: the model and its loss trace."""
    np.random.seed(SEED)
    model = ref.MiniGPT(**SHAPE)
    np.random.seed(SEED + 100)
    with contextlib.redirect_stdout(io.StringIO()):
        trained, losses = ref.sft_train(model, dataset, num_epochs=EPOCHS, lr=lr)
    return trained, losses


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sweep = {lr: run(ref, ref.INSTRUCTION_DATA, lr)[1] for lr in RATES}
    real, _ = run(ref, ref.INSTRUCTION_DATA, RATES[1])
    decoy, _ = run(ref, DECOY, RATES[1])
    np.random.seed(SEED)
    start = weights(ref.MiniGPT(**SHAPE))
    trained = weights(real)
    return {
        "first": {lr: losses[0] for lr, losses in sweep.items()},
        "last": {lr: losses[-1] for lr, losses in sweep.items()},
        "spread": max(l[-1] for l in sweep.values()) - min(l[-1] for l in sweep.values()),
        "data_independent": all(np.array_equal(weights(real)[n], weights(decoy)[n])
                                for n in NAMES),
        "moved": [n for n in NAMES if not np.array_equal(start[n], trained[n])],
    }


def verify(result):
    first, last = result["first"], result["last"]
    moved = result["moved"]
    frozen = [n for n in NAMES if n not in moved]
    return [
        practice.Check(
            "ANSWER: 5.5011 at every learning rate, across four orders of magnitude",
            result["spread"] < 1e-3 and abs(last[RATES[0]] - last[RATES[-1]]) < 1e-3,
            "final loss by learning rate: "
            + ", ".join(f"{lr:.0e} {last[lr]:.4f}" for lr in RATES)
            + f", a spread of {result['spread']:.1e} over a 1000x range of lr. None of the "
            "exercise's three predictions occurs: no rapid initial descent, no overfitting at "
            "1e-4, no sweet spot at 2e-5 -- and 'the 1e-6 run should barely move' is right for "
            "a reason that applies just as well to the other three",
        ),
        practice.Check(
            "MECHANISM: sft_train computes the gradient and then updates with noise",
            all(abs(first[lr] - last[lr]) < 1e-3 for lr in RATES),
            "the loop builds dlogits, masks it and divides by the response-token count, then "
            "writes `block.ffn.W1 -= lr * np.random.randn(*block.ffn.W1.shape) * 0.01`. dlogits "
            "appears nowhere on the right-hand side, so lr scales a random walk and nothing "
            "else. First step to last moves the loss by "
            + ", ".join(f"{abs(first[lr] - last[lr]):.1e}" for lr in RATES)
            + " at the four rates",
        ),
        practice.Check(
            "PROOF: training on a different dataset produces bit-identical weights",
            result["data_independent"],
            "running sft_train on INSTRUCTION_DATA and on eight examples of 'zzzz'/'qqqq' from "
            f"the same seed leaves all {len(NAMES)} named arrays equal bit for bit. A trainer "
            "whose output does not depend on its input is not training, and every exercise in "
            "this lesson that ends 'compare the final loss' is comparing two of these",
        ),
        practice.Check(
            "FINDING: even the random walk reaches only the two FFN matrices",
            moved == ["ffn.W1", "ffn.W2"],
            f"after {EPOCHS} epochs {moved} have moved and {frozen} have not. The attention "
            "stack and the embedding table are frozen exactly as they are in lesson 04's "
            "pre-training loop -- and here there is no gradient anywhere to replace them with",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
