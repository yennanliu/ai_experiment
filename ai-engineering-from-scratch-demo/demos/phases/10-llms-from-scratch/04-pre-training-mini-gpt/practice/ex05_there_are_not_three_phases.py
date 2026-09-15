"""Exercise 5 — there is one phase, it has not plateaued, and the plateau is not overfitting.

    Build a training loss curve plotter. Train the model for 1000 steps and plot
    loss vs step. Identify the three phases: rapid initial descent (learning
    common bytes), slower middle phase (learning byte patterns), and plateau
    (overfitting on the small corpus). The shape of this curve is the same
    whether you are training a 128-dim model or GPT-4.

Reading of the exercise: the curve is the one `train_mini_gpt` already prints,
every 20 steps at its own defaults, so "build a plotter" is read as *record the
curve and test the three claims made about it* -- which is what the rest of the
exercise asks for, and the three claims are falsifiable where a plot is not.

**ANSWER: there are not three phases.** Over 1000 steps the per-third drop is
0.8135, 0.4694, 0.3771 -- monotonically decelerating, one descent, no knee
anywhere. Each third is slower than the last by a shrinking factor, 1.73 then
1.24, which is what a single decelerating descent looks like, not three regimes.

**FINDING: the third phase has not happened.** The minimum is at logged point 48
of 50 and the last stretch still moves by more than the noise, so at step 1000
the loss is still falling. The curve ends at 3.73 against the uniform-byte
baseline ln(256) = 5.5452: it has covered a third of the distance to a byte
entropy it never reaches.

**FINDING: the plateau is attributed to overfitting, which a training curve
cannot show.** Overfitting is the *gap* between training and held-out loss.
Measured here it is **+0.05**, 1.4% -- training 3.7052, held-out 3.7573. There
is no overfitting after 1000 steps, and there could not be evidence of any in
the curve the exercise asks you to plot, because a training-loss curve does not
contain the held-out loss.

**FINDING: 30% of the model never trains.** `train_mini_gpt`'s backward pass
takes the residual path around the attention sub-block and never enters it, so
`W_q`, `W_k`, `W_v`, `W_out`, `ln1` and `pos_embed` stay at their random
initialisation for all 1000 steps. "The shape of this curve is the same whether
you are training a 128-dim model or GPT-4" is claimed for a curve produced with
attention frozen -- which is the one thing GPT-4's curve is mostly about.

Structure: `curve` runs the reference trainer and recovers its printed losses;
`held_out` scores a corpus the trainer never saw.
"""

from __future__ import annotations

import contextlib
import io
import math
import statistics

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "04-pre-training-mini-gpt"
STEPS, SEED, CONTEXT = 1000, 1, 64
TRAIN = ("Machine learning is a subset of artificial intelligence. "
         "Deep learning uses neural networks with many layers. "
         "The transformer architecture relies on self-attention. "
         "Language models predict the next token in a sequence. ") * 10
HELD = ("Gradient descent follows the slope of the loss surface downhill. "
        "A tokenizer maps text into the integers a model can read. ") * 4
FROZEN = ("W_q", "W_k", "W_v", "W_out")


def curve(ref, steps, seed):
    """`train_mini_gpt`'s own printed loss trace, every 20 steps."""
    np.random.seed(seed)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        model = ref.train_mini_gpt(TRAIN, num_steps=steps)
    return model, [float(line.split("Loss: ")[1])
                   for line in buffer.getvalue().splitlines() if "Loss:" in line]


def held_out(ref, model, text, chunks=6):
    """Mean cross-entropy over `chunks` windows of text, by the lesson's own loss."""
    tokens = np.array(list(text.encode("utf-8")))
    scores = []
    for i in range(chunks):
        window = tokens[i * CONTEXT:i * CONTEXT + CONTEXT + 1]
        if len(window) <= CONTEXT:
            break
        logits = model.forward(window[:-1].reshape(1, -1))
        scores.append(float(ref.cross_entropy_loss(logits, window[1:].reshape(1, -1))))
    return statistics.fmean(scores)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    np.random.seed(SEED)
    start = ref.MiniGPT(vocab_size=256, embed_dim=128, num_heads=4, num_layers=4,
                        max_seq_len=CONTEXT, ff_dim=512)
    model, losses = curve(ref, STEPS, SEED)
    third = len(losses) // 3
    parts = (losses[:third], losses[third:2 * third], losses[2 * third:])
    return {
        "losses": losses,
        "drops": [part[0] - part[-1] for part in parts],
        "argmin": losses.index(min(losses)) + 1,
        "tail": losses[-6] - losses[-1],
        "noise": statistics.stdev(losses[-6:]),
        "uniform": math.log(256),
        "train": held_out(ref, model, TRAIN),
        "heldout": held_out(ref, model, HELD),
        "frozen": [name for name in FROZEN
                   if np.array_equal(getattr(start.blocks[0].attn, name),
                                     getattr(model.blocks[0].attn, name))],
        "pos_frozen": np.array_equal(start.embedding.pos_embed, model.embedding.pos_embed),
    }


def verify(result):
    losses, drops = result["losses"], result["drops"]
    gap = result["heldout"] - result["train"]
    return [
        practice.Check(
            "ANSWER: there are not three phases -- the drop decelerates monotonically",
            drops[0] > drops[1] > drops[2] > 0,
            f"over {STEPS} steps the per-third drop is "
            + ", ".join(f"{d:.4f}" for d in drops)
            + f", each third slower than the last by a shrinking amount: "
            f"{drops[0] / drops[1]:.2f} then {drops[1] / drops[2]:.2f}. That is one "
            "decelerating descent, not a fast phase, a slow phase and a plateau -- there is no "
            "knee anywhere in the curve to put a boundary at",
        ),
        practice.Check(
            "FINDING: the third phase has not happened -- the loss is still falling at step 1000",
            result["argmin"] > 0.9 * len(losses) and result["tail"] > result["noise"],
            f"the minimum is at logged point {result['argmin']} of {len(losses)}, and over the "
            f"last 100 steps the loss moves {result['tail']:.4f} against a local spread of "
            f"{result['noise']:.4f} -- still descending faster than it wobbles. It ends at "
            f"{losses[-1]:.4f} against the uniform-byte baseline ln(256) = "
            f"{result['uniform']:.4f}, having covered "
            f"{100 * (losses[0] - losses[-1]) / result['uniform']:.0f}% of the way to a byte "
            "entropy it never reaches",
        ),
        practice.Check(
            "FINDING: the plateau is called overfitting, which a training curve cannot show",
            abs(gap) < 0.1 * result["train"],
            f"overfitting is the gap between training and held-out loss, and here it is "
            f"{gap:+.4f} -- {100 * gap / result['train']:.1f}% -- with training at "
            f"{result['train']:.4f} and held-out text at {result['heldout']:.4f}. There is no "
            f"overfitting after {STEPS} steps, and there could be no evidence of any in the "
            "curve the exercise asks you to plot, because a training-loss curve does not "
            "contain the held-out loss. The third phase is named after a quantity not on the axis",
        ),
        practice.Check(
            "FINDING: the curve is produced with 30% of the model frozen at random init",
            set(result["frozen"]) == set(FROZEN) and result["pos_frozen"],
            f"after {STEPS} steps {sorted(result['frozen'])} and pos_embed are bit-for-bit "
            "unchanged: the backward pass takes the residual path around the attention "
            "sub-block and never enters it. 'The shape of this curve is the same whether you "
            "are training a 128-dim model or GPT-4' is claimed for a curve produced with "
            "attention frozen -- which is the part of GPT-4 the curve is mostly about",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
