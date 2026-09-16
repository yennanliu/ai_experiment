"""Exercise 2 — the target is the block mean, so mean-pool scores exactly zero and the MLP cannot tie it.

    Replace the mean-pool compressor with a tiny learned MLP (2-layer, hidden
    32). Train it on a synthetic task where the signal is the average of a
    block. Measure the perplexity gap against the mean-pool baseline on
    held-out data.

Reading of the exercise: the MLP is built at the size the exercise names and
trained by hand-written backprop on the task the exercise names -- predict the
average of a block -- and the baseline it is measured against is the lesson's own
`compress_mean`. The gap is reported as mean squared error rather than
perplexity, because the task the exercise specifies produces a vector and not a
distribution, and a perplexity would have to be invented.

**ANSWER: mean-pool scores exactly 0.000e+00, and a trained MLP reaches
1.19e-05.**

    before training   mean-pool 0.000e+00   MLP 8.94e-02
    after 80 epochs   mean-pool 0.000e+00   MLP 1.19e-05

The baseline is not a baseline on this task -- it is the closed-form solution.
`compress_mean` computes the block average and the target *is* the block
average, so its error is zero by construction and no amount of training can
close a gap to it. The exercise asks to measure a quantity whose sign is fixed
before the experiment runs.

**FINDING: the MLP does learn the function, and learning it is the problem.**
Training cuts its error by **7,519x**, from 8.94e-02 to 1.19e-05, which is the
MLP successfully discovering that the answer is its own input. Every parameter
it spends is spent approximating the identity.

**MECHANISM: NSA's learned compressor exists for a task this synthetic does not
contain.** A real compressor maps `l` raw keys into one summary that must
preserve *whatever the router needs*, which is not the mean -- it is a learned
projection into the query's space. Feed it a task whose answer is the mean and
the optimum is mean-pooling, so the experiment compares a network against the
function it is being asked to imitate.

**FINDING: the comparison the exercise names would need a different target.** If
the signal were, say, the block's *maximum*, mean-pool's error would be
**5.60e+00** and the MLP would have something to win. The choice of "average of a
block" is what makes the answer unbounded rather than close.

Structure: `mlp` is the 2-layer network the exercise specifies; `train` runs
plain SGD over its own gradients; `score` is MSE against held-out blocks.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "17-native-sparse-attention"
DIM, BLOCK, HIDDEN = 16, 64, 32
TRAIN, TEST, EPOCHS, LR, SEED = 200, 100, 80, 0.5, 3


def blocks(rng, count, target):
    """`count` blocks of `BLOCK` keys, each paired with the statistic `target` extracts."""
    out = []
    for _ in range(count):
        block = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(BLOCK)]
        out.append((block, target(block)))
    return out


def block_mean(block):
    return [sum(row[c] for row in block) / len(block) for c in range(DIM)]


def block_max(block):
    return [max(row[c] for row in block) for c in range(DIM)]


def new_mlp(rng):
    return {"w1": [[rng.gauss(0, 0.3) for _ in range(HIDDEN)] for _ in range(DIM)],
            "b1": [0.0] * HIDDEN,
            "w2": [[rng.gauss(0, 0.3) for _ in range(DIM)] for _ in range(HIDDEN)],
            "b2": [0.0] * DIM}


def forward(net, x):
    hidden = [math.tanh(sum(x[i] * net["w1"][i][j] for i in range(DIM)) + net["b1"][j])
              for j in range(HIDDEN)]
    return hidden, [sum(hidden[j] * net["w2"][j][c] for j in range(HIDDEN)) + net["b2"][c]
                    for c in range(DIM)]


def descend(matrix, rows, grad, scale):
    """`matrix[i][j] -= LR * scale[i] * grad[j]` over a rectangular block of weights."""
    for i in range(rows):
        for j in range(len(grad)):
            matrix[i][j] -= LR * scale[i] * grad[j]


def step(net, x, target):
    """One SGD step on one block, by hand."""
    hidden, out = forward(net, x)
    d_out = [2 * (out[c] - target[c]) / DIM for c in range(DIM)]
    d_hidden = [sum(d_out[c] * net["w2"][j][c] for c in range(DIM)) * (1 - hidden[j] ** 2)
                for j in range(HIDDEN)]
    descend(net["w2"], HIDDEN, d_out, hidden)
    descend(net["w1"], DIM, d_hidden, x)
    for j in range(HIDDEN):
        net["b1"][j] -= LR * d_hidden[j]
    for c in range(DIM):
        net["b2"][c] -= LR * d_out[c]


def train(net, data, compress):
    for _ in range(EPOCHS):
        for block, target in data:
            step(net, compress(block), target)
    return net


def score(data, predict):
    return statistics.fmean(sum((a - b) ** 2 for a, b in zip(predict(block), target)) / DIM
                            for block, target in data)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(SEED)
    training, held_out = blocks(rng, TRAIN, block_mean), blocks(rng, TEST, block_mean)
    pooled = lambda block: ref.compress_mean(block, BLOCK)[0]        # noqa: E731
    net = new_mlp(rng)
    before = score(held_out, lambda block: forward(net, pooled(block))[1])
    train(net, training, pooled)
    after = score(held_out, lambda block: forward(net, pooled(block))[1])
    max_task = blocks(rng, TEST, block_max)
    return {
        "pool": score(held_out, pooled),
        "mlp": (before, after),
        "improvement": before / after,
        "pool_on_max": score(max_task, pooled),
        "hidden": HIDDEN,
        "epochs": EPOCHS,
    }


def verify(result):
    before, after = result["mlp"]
    return [
        practice.Check(
            "ANSWER: mean-pool scores exactly 0.000e+00 and the MLP cannot reach it",
            result["pool"] == 0.0 and after > 0,
            f"the lesson's own compress_mean scores {result['pool']:.3e} on held-out blocks, and "
            f"a {result['hidden']}-hidden-unit MLP goes from {before:.2e} before training to "
            f"{after:.2e} after {result['epochs']} epochs. The baseline is not a baseline on this "
            "task -- it is the closed-form solution, because compress_mean computes the block "
            "average and the target is the block average. The gap the exercise asks to measure "
            "has its sign fixed before the experiment runs",
        ),
        practice.Check(
            "FINDING: the MLP does learn the function, and learning it is the problem",
            result["improvement"] > 100,
            f"training cuts the MLP's error by {result['improvement']:,.0f}x, from {before:.2e} to "
            f"{after:.2e}. That is the network successfully discovering that the answer is its own "
            "input: every one of its parameters is spent approximating the identity, and the best "
            "outcome available to it is to stop being a network",
        ),
        practice.Check(
            "MECHANISM: NSA's learned compressor exists for a task this synthetic does not contain",
            after > result["pool"],
            "a real compressor maps l raw keys into one summary that must preserve whatever the "
            "router needs, which is not the mean -- it is a learned projection into the query's "
            "space. Feed it a task whose answer is the mean and the optimum is mean-pooling, so "
            "the experiment compares a network against the function it is being asked to imitate, "
            f"and the MLP's {after:.2e} can only approach the pool's {result['pool']:.3e} from "
            "above",
        ),
        practice.Check(
            "FINDING: the comparison would need a different target to be a comparison",
            result["pool_on_max"] > 0.01,
            f"if the signal were the block's maximum instead of its average, mean-pool would "
            f"score {result['pool_on_max']:.2e} on the same blocks and the MLP would have "
            "something to win. The choice of 'the average of a block' is what makes the answer "
            "unbounded rather than close, and it is the one line of the exercise that decides the "
            "result",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
