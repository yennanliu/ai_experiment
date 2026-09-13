"""Exercise 1 — eleven of the sixteen head counts do not exist.

    **Easy.** Take the MHA from `code/main.py` and change `n_heads` from 1 to 16
    with `d_model=64` fixed. Plot the loss of a tiny one-layer model on a
    synthetic copy task. Do more heads help, plateau, or hurt?

Reading of the exercise: "from 1 to 16" is swept over every integer, which is
how the illegal ones surface. The lesson ships no loss, no gradients and no
optimizer, so the "tiny one-layer model" is built here -- the lesson's own MHA
forward, its backward pass, and Adam -- and the numpy forward is checked against
`multi_head_attention` at d_model=64 before it is trusted. The copy task is
two simultaneous lookups per position: each token carries two query keys and
must emit the values stored at the two positions those keys match, which is the
smallest task a single head provably cannot do.

**FINDING: only 5 of the 16 head counts are legal.** `split_heads` asserts
`d_model % n_heads == 0`, so at d_model=64 the sweep the exercise describes
raises `AssertionError("d_model not divisible by n_heads")` for 11 of its 16
values. The legal set is {1, 2, 4, 8, 16}, and a plot "from 1 to 16" has five
points on it.

**ANSWER: all three, in that order, and the turn is d_head.**

| n_heads | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| d_head | 64 | 32 | 16 | 8 | 4 |
| variance explained | 51% | **99%** | 99% | 93% | 72% |

One head **fails** -- a single attention distribution cannot be one-hot at two
positions at once, so it recovers only half the target. Two heads **help**,
enormously. Four **plateau**. Eight and sixteen **hurt**, and they hurt where
`d_head` drops below the task's key dimension of 12: at d_head 8 and 4 a head
can no longer represent the key it is matching on. The variable that decides the
answer is `d_model / n_heads`, not `n_heads`.

**CONTROL: the lesson's own Matrix agrees to 1.1e-15.** The sweep runs in numpy
because `matmul` is a triple Python loop, but the numbers are the lesson's.

Structure: `task` is the two-lookup copy task; `forward`/`backward` are one MHA
layer and its gradients; `train` is Adam; `legal` probes the assertion.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "03-multi-head-attention"
D_MODEL, TOKENS, KEY, VALUE = 64, 8, 12, 14
SWEEP, STEPS, BATCH, RATE = (1, 2, 4, 8, 16), 600, 32, 3e-3


def task(np, rng, batch):
    """Two lookups per position: emit the values at the positions its keys match."""
    keys = rng.normal(0, 1, (batch, TOKENS, KEY))
    keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
    values = [rng.normal(0, 1, (batch, TOKENS, VALUE)) for _ in range(2)]
    picks = [rng.integers(0, TOKENS, (batch, TOKENS))[..., None] for _ in range(2)]
    x = np.concatenate([np.take_along_axis(keys, p, 1) for p in picks] + [keys] + values, -1)
    target = np.zeros_like(x)
    target[..., :2 * VALUE] = np.concatenate(
        [np.take_along_axis(v, p, 1) for v, p in zip(values, picks)], -1)
    return x, target


def split(a, n_heads):
    """(batch, tokens, d_model) -> (batch, n_heads, tokens, d_head)."""
    return a.reshape(a.shape[:-1] + (n_heads, a.shape[-1] // n_heads)).swapaxes(1, 2)


def forward(np, x, w, n_heads):
    """One MHA layer: the lesson's arithmetic, batched."""
    q, k, v = (split(x @ m, n_heads) for m in w[:3])
    scores = q @ k.swapaxes(-1, -2) / np.sqrt(D_MODEL / n_heads)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    context = (weights @ v).swapaxes(1, 2).reshape(x.shape)
    return context @ w[3], (q, k, v, weights, context)


def backward(np, x, target, w, n_heads, cache):
    """Gradients of the mean-squared error, by hand -- there is no autograd here."""
    q, k, v, weights, context = cache
    grad = 2 * (context @ w[3] - target) / target.size
    d_context = split(grad @ w[3].T, n_heads)
    d_weights = d_context @ v.swapaxes(-1, -2)
    d_scores = weights * (d_weights - (d_weights * weights).sum(-1, keepdims=True))
    d_scores /= np.sqrt(D_MODEL / n_heads)
    rows, merge = x.reshape(-1, D_MODEL).T, [d_scores @ k, d_scores.swapaxes(-1, -2) @ q,
                                             weights.swapaxes(-1, -2) @ d_context]
    return [rows @ m.swapaxes(1, 2).reshape(-1, D_MODEL) for m in merge] + [
        context.reshape(-1, D_MODEL).T @ grad.reshape(-1, D_MODEL)]


def train(np, n_heads, seed=0):
    """Adam on one MHA layer; returns validation MSE and the variance it must beat."""
    rng = np.random.default_rng(seed)
    w = [rng.normal(0, 0.125, (D_MODEL, D_MODEL)) for _ in range(4)]
    avg, sq = ([np.zeros_like(m) for m in w] for _ in range(2))
    for step in range(1, STEPS + 1):
        x, target = task(np, rng, BATCH)
        for i, g in enumerate(backward(np, x, target, w, n_heads,
                                       forward(np, x, w, n_heads)[1])):
            avg[i], sq[i] = 0.9 * avg[i] + 0.1 * g, 0.999 * sq[i] + 0.001 * g * g
            w[i] -= RATE * (avg[i] / (1 - 0.9 ** step)) / (
                np.sqrt(sq[i] / (1 - 0.999 ** step)) + 1e-8)
    x, target = task(np, np.random.default_rng(9999), 256)
    return float(((forward(np, x, w, n_heads)[0] - target) ** 2).mean()), float((target ** 2).mean())


def legal(ref):
    """Which of the 16 head counts `split_heads` actually accepts at d_model=64."""
    def accepts(n_heads):
        try:
            return bool(ref.split_heads(ref.Matrix(2, D_MODEL), n_heads))
        except AssertionError:
            return False
    return [n for n in range(1, 17) if accepts(n)]


def agreement(np, ref):
    """The numpy forward against the lesson's own Matrix MHA, at d_model=64."""
    rng = random.Random(42)
    x = ref.randn_matrix(TOKENS, D_MODEL, rng, scale=1.0)
    w = [ref.randn_matrix(D_MODEL, D_MODEL, rng) for _ in range(4)]
    theirs = ref.multi_head_attention(x, *w, n_heads=4)[0]
    pull = [np.array(m.data).reshape(m.rows, m.cols) for m in (x, *w, theirs)]
    return float(np.abs(forward(np, pull[0][None], pull[1:5], 4)[0][0] - pull[5]).max())


def solve():
    import numpy as np
    ref = parity.load_reference(PHASE, LESSON, "main")
    scores = {n: train(np, n) for n in SWEEP}
    return {"legal": legal(ref), "agreement": agreement(np, ref),
            "explained": {n: 1 - mse / base for n, (mse, base) in scores.items()},
            "d_head": {n: D_MODEL // n for n in SWEEP}}


def verify(result):
    got, heads = result["explained"], result["d_head"]
    say = {n: f"{v:.1%}" for n, v in got.items()}
    return [
        practice.Check(
            "FINDING: 11 of the 16 head counts the exercise sweeps do not exist",
            result["legal"] == list(SWEEP),
            f"split_heads asserts d_model % n_heads == 0, so at d_model={D_MODEL} only "
            f"{result['legal']} of 1..16 are accepted; the other 11 raise AssertionError. "
            "A plot 'from 1 to 16' has five points on it",
        ),
        practice.Check(
            "ANSWER: one head fails -- a single distribution cannot be one-hot twice",
            got[1] < 0.7,
            f"each position must emit the values at the two positions its two keys match. One "
            f"head has one attention distribution, so it recovers {say[1]} of the target "
            "variance: about half, which is what averaging two lookups gets you",
        ),
        practice.Check(
            "ANSWER: two heads help enormously, and four plateau",
            got[2] > 0.97 and abs(got[4] - got[2]) < 0.03,
            f"{say[1]} at 1 head -> {say[2]} at 2 -> {say[4]} at 4. The second head is worth 47 "
            "points of variance and the third and fourth are worth nothing: the task needs two "
            "lookups, and buying more than two only changes how wide each one is",
        ),
        practice.Check(
            "ANSWER: eight and sixteen hurt, where d_head crosses the key width",
            got[8] < got[4] and got[16] < got[8] - 0.1,
            f"{say[4]} at d_head={heads[4]} -> {say[8]} at {heads[8]} -> {say[16]} at "
            f"{heads[16]}. The task's keys are {KEY}-dimensional and a head narrower than {KEY} "
            "cannot represent what it matches on. d_model / n_heads decides this, not n_heads",
        ),
        practice.Check(
            "CONTROL: the numpy forward is the lesson's own arithmetic to 1e-15",
            result["agreement"] < 1e-13,
            f"the sweep runs in numpy because the lesson's matmul is a triple Python loop, so it "
            f"is checked against multi_head_attention at d_model={D_MODEL} on the lesson's own "
            f"randn_matrix weights: {result['agreement']:.1e} worst disagreement",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
