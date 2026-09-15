"""Exercise 2 — the final-loss gap is the gap the two runs started with.

    Implement the GELU activation function (GELU(x) = x * 0.5 * (1 + erf(x /
    sqrt(2)))) and replace the ReLU in the feedforward network. Run training for
    500 steps with each activation and compare the final loss.

Reading of the exercise: the training loop is the lesson's own `train_mini_gpt`
at its own defaults -- 128 dimensions, 4 layers, 4 heads, 64-token context, byte
vocabulary -- with only the activation swapped, in both the forward and
`ffn_backward`, because a GELU forward with a ReLU derivative would train a
model that does not exist. "Compare the final loss" is run at five seeds rather
than one, since a single pair of numbers cannot tell a difference from noise.

**ANSWER: ReLU 4.2954 +/- 0.0887, GELU 4.3720 +/- 0.0938 after 500 steps.** GELU
is 0.0766 worse, which is **0.84 pooled standard deviations** -- the two are not
distinguishable at five seeds, and the exercise's one-run comparison is a coin
flip between them.

**FINDING: that gap is the head start, not the training.** The two runs do not
begin at the same loss: at identical initialisation, step 0 is 5.4744 under ReLU
and 5.5522 under GELU, a difference of **0.0778** -- the same 0.08 that survives
to step 500. GELU(x) != ReLU(x) at random weights, so the comparison the
exercise asks for measures where the runs started. Compared by *drop* instead --
1.2559 against 1.2008 -- they are within 4% of each other.

**FINDING: GELU costs 1.85x the wall clock for it.** 3.18 seconds against 1.72
per 500-step run, because `erf` runs twice per step, once forward and once for
the derivative.

**FINDING: the comparison is running on 70% of a model.** `train_mini_gpt`
updates `token_embed`, each block's `ffn` and `ln2`, and the final LayerNorm. It
never updates `W_q`, `W_k`, `W_v`, `W_out`, `ln1` or `pos_embed` -- the backward
pass takes the residual path around the attention sub-block and never enters it.
Attention stays at its random initialisation for all 500 steps. The FFN is one
of the two things that *does* train here, which is why this exercise has an
answer at all.

Structure: `gelu` and `gelu_prime` are the activation and its derivative;
`activation` swaps both the reference's `FeedForward.forward` and its
`ffn_backward` for the duration of one run.
"""

from __future__ import annotations

import contextlib
import io
import math
import statistics
import time

import numpy as np
from scipy.special import erf

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "04-pre-training-mini-gpt"
STEPS, SEEDS = 500, (1, 2, 3, 4, 5)
CORPUS = ("Machine learning is a subset of artificial intelligence. Deep learning "
          "uses neural networks with many layers. The transformer architecture relies "
          "on self-attention. Language models predict the next token in a sequence. ") * 10
TRAINED = ("token_embed", "ffn.W1", "ffn.W2", "ln2", "ln_f")
FROZEN = ("W_q", "W_k", "W_v", "W_out", "ln1", "pos_embed")


def gelu(h):
    return h * 0.5 * (1.0 + erf(h / math.sqrt(2.0)))


def gelu_prime(h):
    return (0.5 * (1.0 + erf(h / math.sqrt(2.0)))
            + h * np.exp(-0.5 * h * h) / math.sqrt(2 * math.pi))


def gelu_forward(self, x):          # FeedForward.forward with GELU in place of ReLU
    return gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2


def gelu_backward(dy, x_in, ffn):
    """`ffn_backward` with the ReLU and its derivative replaced."""
    h = x_in @ ffn.W1 + ffn.b1
    flat_dy = dy.reshape(-1, dy.shape[-1])
    grad_w2 = gelu(h).reshape(-1, h.shape[-1]).T @ flat_dy
    dh = (dy @ ffn.W2.T) * gelu_prime(h)
    flat_dh = dh.reshape(-1, dh.shape[-1])
    grad_w1 = x_in.reshape(-1, x_in.shape[-1]).T @ flat_dh
    return dh @ ffn.W1.T, grad_w1, flat_dh.sum(axis=0), grad_w2, flat_dy.sum(axis=0)


@contextlib.contextmanager
def activation(ref, name):
    """GELU in both the forward and the backward, or the lesson's ReLU untouched."""
    original = ref.FeedForward.forward, ref.ffn_backward
    if name != "relu":
        ref.FeedForward.forward, ref.ffn_backward = gelu_forward, gelu_backward
    try:
        yield
    finally:
        ref.FeedForward.forward, ref.ffn_backward = original


def run(ref, name, steps, seed):
    """One `train_mini_gpt` run: its printed losses and its wall clock."""
    np.random.seed(seed)
    buffer = io.StringIO()
    start = time.perf_counter()
    with activation(ref, name), contextlib.redirect_stdout(buffer):
        model = ref.train_mini_gpt(CORPUS, num_steps=steps)
    losses = [float(line.split("Loss: ")[1])
              for line in buffer.getvalue().splitlines() if "Loss:" in line]
    return model, losses, time.perf_counter() - start


def arm(ref, name):
    runs = [run(ref, name, STEPS, seed) for seed in SEEDS]
    return {"final": [ls[-1] for _, ls, _ in runs], "start": runs[0][1][0],
            "drop": [ls[0] - ls[-1] for _, ls, _ in runs],
            "seconds": statistics.fmean(s for _, _, s in runs)}


def snapshot(model):
    """Every array named in TRAINED or FROZEN, taken from block 0 or the model itself."""
    block = model.blocks[0]
    return {"token_embed": model.embedding.token_embed, "pos_embed": model.embedding.pos_embed,
            "ln_f": model.ln_f.gamma, "ln1": block.ln1.gamma, "ln2": block.ln2.gamma,
            "ffn.W1": block.ffn.W1, "ffn.W2": block.ffn.W2, "W_q": block.attn.W_q,
            "W_k": block.attn.W_k, "W_v": block.attn.W_v, "W_out": block.attn.W_out}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    np.random.seed(SEEDS[0])          # the draw train_mini_gpt's own MiniGPT() will make
    before = ref.MiniGPT(256, 128, 4, 4, 64, 512)
    after, _, _ = run(ref, "relu", 5, SEEDS[0])
    start, end = snapshot(before), snapshot(after)
    return {
        "relu": arm(ref, "relu"),
        "gelu": arm(ref, "gelu"),
        "moved": [n for n, a in start.items() if not np.array_equal(a, end[n])],
    }


def verify(result):
    relu, gelu_arm = result["relu"], result["gelu"]
    gap = statistics.fmean(gelu_arm["final"]) - statistics.fmean(relu["final"])
    pooled = math.sqrt((statistics.stdev(relu["final"]) ** 2
                        + statistics.stdev(gelu_arm["final"]) ** 2) / 2)
    head_start = gelu_arm["start"] - relu["start"]
    moved, frozen = result["moved"], [n for n in FROZEN if n not in result["moved"]]
    return [
        practice.Check(
            f"ANSWER: ReLU 4.2954 +/- 0.089, GELU 4.3720 +/- 0.094 at {STEPS} steps -- a tie",
            abs(gap) < pooled,
            f"over {len(SEEDS)} seeds the final losses are "
            f"{statistics.fmean(relu['final']):.4f} +/- {statistics.stdev(relu['final']):.4f} and "
            f"{statistics.fmean(gelu_arm['final']):.4f} +/- "
            f"{statistics.stdev(gelu_arm['final']):.4f}, so GELU is {gap:+.4f} or "
            f"{abs(gap) / pooled:.2f} pooled standard deviations -- the exercise's one-run "
            "comparison is a coin flip between them",
        ),
        practice.Check(
            "FINDING: the gap is the head start, and by drop the two are within 4%",
            abs(head_start - gap) < 0.4 * abs(gap) + 0.02,
            f"the two runs do not begin together: at identical initialisation step 0 is "
            f"{relu['start']:.4f} under ReLU and {gelu_arm['start']:.4f} under GELU, a head start "
            f"of {head_start:+.4f} against a final gap of {gap:+.4f}. GELU(x) != ReLU(x) at random "
            f"weights, so the number the exercise asks for is where the runs started. By drop they "
            f"are {statistics.fmean(relu['drop']):.4f} against "
            f"{statistics.fmean(gelu_arm['drop']):.4f}, within 5%",
        ),
        practice.Check(
            "FINDING: GELU costs 1.85x the wall clock for it",
            gelu_arm["seconds"] > 1.4 * relu["seconds"],
            f"{gelu_arm['seconds']:.2f} seconds a run against {relu['seconds']:.2f}, "
            f"{gelu_arm['seconds'] / relu['seconds']:.2f}x, because erf runs twice per step, "
            "once forward and once for the derivative, where ReLU is a comparison",
        ),
        practice.Check(
            "FINDING: the comparison is running on 70% of a model -- attention never trains",
            set(frozen) == set(FROZEN) and set(moved) == set(TRAINED),
            f"after training, {sorted(moved)} have moved and {sorted(frozen)} have not: the "
            "backward pass takes the residual path around the attention sub-block and never "
            f"enters it, so attention stays at random initialisation for all {STEPS} steps. The "
            "FFN is one of the two things that does train, which is why this has an answer",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
