"""Exercise 3 — the two conditions the question asks to satisfy never overlap.

    **Hard.** Implement a tiny version of Multi-head Latent Attention: compress
    K,V to a rank-`r` latent, store the latent in the KV cache, decompress at
    attention time. At what `r` does cache memory cross below 1/8 of full MHA
    while quality stays within 1 bit of validation ppl?

Reading of the exercise: the lesson has no model, no dataset, no loss and no
perplexity, so "validation ppl" is read as the validation loss of the same
one-layer model and two-lookup copy task Exercise 1 has to build, and "1 bit" is
converted honestly: for a Gaussian residual, one bit per dimension is a factor of
**4 in mean-squared error**, since `0.5 * log2(mse_a / mse_b) = 1`. The MLA is
`c = X Wd` at width r, then `K = c Wuk`, `V = c Wuv`, with only `c` cached.

**ANSWER: no such r exists here, and the gap is 2.25x.**

| r | cache vs MHA | below 1/8 | bits worse |
|---:|---:|---|---:|
| 16 | 0.125 | no (equal) | 2.81 |
| 32 | 0.250 | no | 1.82 |
| **36** | 0.281 | no | **0.71** |
| 40 | 0.312 | no | -0.19 |
| 64 | 0.500 | no | **-1.04** |

The cache condition is pure arithmetic: MLA stores r numbers per position where
MHA stores `2 * d_model`, so `r / 128 < 1/8` means **r < 16**. The quality
condition first holds at **r = 36**, a cache ratio of 0.281. The two never meet,
and they miss by a factor of 2.25.

**FINDING: the floor is information, not optimisation.** Every position's K and V
must carry its key and both of its values -- `12 + 14 + 14 = 40` dimensions --
and the curve breaks exactly there: 1.82 bits worse at r=32, 0.71 at r=36, and
*better than full MHA* from r=40 up. No training schedule can push a rank-r
bottleneck below the rank of what it has to carry, so the exercise's 1/8 target
is under the task's own floor before any model is trained.

**FINDING: the free win is 2x, not 8x.** At `r = d_model = 64` the cache is
already halved -- one latent instead of a K and a V -- and quality is **1.04
bits better** than full MHA, because the shared low-rank factorisation ties K and
V together. The exercise asks for the ratio where MLA starts to hurt; the
measurement says it starts by helping.

Structure: `task`, `forward`, `backward`, `train` are Exercise 1's model with an
optional latent path; `bits` is the mse-to-bits conversion stated above.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "03-multi-head-attention"
D_MODEL, TOKENS, KEY, VALUE = 64, 8, 12, 14
HEADS, STEPS, BATCH, RATE = 4, 600, 32, 3e-3
RANKS, TARGET, BUDGET = (16, 32, 36, 40, 64), 0.125, 1.0


def task(np, rng, batch):
    """Exercise 1's task: two lookups per position, the smallest two-head problem."""
    keys = rng.normal(0, 1, (batch, TOKENS, KEY))
    keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
    values, picks = ([rng.normal(0, 1, (batch, TOKENS, VALUE)) for _ in range(2)],
                     [rng.integers(0, TOKENS, (batch, TOKENS))[..., None] for _ in range(2)])
    x = np.concatenate([np.take_along_axis(keys, p, 1) for p in picks] + [keys] + values, -1)
    target = np.zeros_like(x)
    target[..., :2 * VALUE] = np.concatenate(
        [np.take_along_axis(v, p, 1) for v, p in zip(values, picks)], -1)
    return x, target


def split(a):
    """(batch, tokens, d_model) -> (batch, HEADS, tokens, d_head)."""
    return a.reshape(a.shape[:-1] + (HEADS, a.shape[-1] // HEADS)).swapaxes(1, 2)


def forward(np, x, w, rank):
    """MHA, or MLA when `rank` is set: K and V then come from one cached latent."""
    latent = None if rank is None else x @ w[4]
    pair = (x @ w[1], x @ w[2]) if rank is None else (latent @ w[5], latent @ w[6])
    q, k, v = split(x @ w[0]), split(pair[0]), split(pair[1])
    scores = q @ k.swapaxes(-1, -2) / math.sqrt(D_MODEL / HEADS)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    context = (weights @ v).swapaxes(1, 2).reshape(x.shape)
    return context @ w[3], (q, k, v, weights, context, latent)


def backward(np, x, target, w, rank, cache):
    """Gradients by hand; the latent path routes dK and dV through Wd."""
    q, k, v, weights, context, latent = cache
    grad, rows = 2 * (context @ w[3] - target) / target.size, x.reshape(-1, D_MODEL)
    d_context = split(grad @ w[3].T)
    d_weights = d_context @ v.swapaxes(-1, -2)
    d_scores = weights * (d_weights - (d_weights * weights).sum(-1, keepdims=True))
    d_scores /= math.sqrt(D_MODEL / HEADS)
    merge = [d_scores @ k, d_scores.swapaxes(-1, -2) @ q, weights.swapaxes(-1, -2) @ d_context]
    dq, dk, dv = (m.swapaxes(1, 2).reshape(-1, D_MODEL) for m in merge)
    d_out = context.reshape(-1, D_MODEL).T @ grad.reshape(-1, D_MODEL)
    if rank is None:
        return [rows.T @ dq, rows.T @ dk, rows.T @ dv, d_out]
    flat = latent.reshape(-1, rank)
    return [rows.T @ dq, np.zeros_like(w[1]), np.zeros_like(w[2]), d_out,
            rows.T @ (dk @ w[5].T + dv @ w[6].T), flat.T @ dk, flat.T @ dv]


def train(np, rank=None, seed=0):
    """Adam; returns the validation MSE of one layer at this latent width."""
    rng = np.random.default_rng(seed)
    make = lambda a, b: rng.normal(0, math.sqrt(2 / (a + b)), (a, b))
    w = [make(D_MODEL, D_MODEL) for _ in range(4)] + ([] if rank is None else [
        make(D_MODEL, rank), make(rank, D_MODEL), make(rank, D_MODEL)])
    avg, sq = ([np.zeros_like(m) for m in w] for _ in range(2))
    for step in range(1, STEPS + 1):
        x, target = task(np, rng, BATCH)
        for i, g in enumerate(backward(np, x, target, w, rank, forward(np, x, w, rank)[1])):
            avg[i], sq[i] = 0.9 * avg[i] + 0.1 * g, 0.999 * sq[i] + 0.001 * g * g
            w[i] -= RATE * (avg[i] / (1 - 0.9 ** step)) / (
                np.sqrt(sq[i] / (1 - 0.999 ** step)) + 1e-8)
    x, target = task(np, np.random.default_rng(9999), 256)
    return float(((forward(np, x, w, rank)[0] - target) ** 2).mean())


def agreement(np, ref):
    """The MHA arm against the lesson's own Matrix implementation, before it is trusted."""
    rng = random.Random(42)
    x, w = ref.randn_matrix(TOKENS, D_MODEL, rng, scale=1.0), [
        ref.randn_matrix(D_MODEL, D_MODEL, rng) for _ in range(4)]
    pull = [np.array(m.data).reshape(m.rows, m.cols)
            for m in (x, *w, ref.multi_head_attention(x, *w, n_heads=HEADS)[0])]
    return float(np.abs(forward(np, pull[0][None], pull[1:5], None)[0][0] - pull[5]).max())


def solve():
    import numpy as np
    ref = parity.load_reference(PHASE, LESSON, "main")
    base, scored = train(np), {r: train(np, rank=r) for r in RANKS}
    bits = {r: 0.5 * math.log2(mse / base) for r, mse in scored.items()}
    return {
        "base": base, "mse": scored, "bits": bits, "agreement": agreement(np, ref),
        "ratio": {r: r / (2 * D_MODEL) for r in RANKS}, "floor": KEY + 2 * VALUE,
        "cache_ok": [r for r in RANKS if r / (2 * D_MODEL) < TARGET],
        "quality_ok": [r for r in RANKS if bits[r] <= BUDGET],
        "budget_r": int(TARGET * 2 * D_MODEL),
    }


def verify(result):
    bits, ratio, first = result["bits"], result["ratio"], min(result["quality_ok"])
    return [
        practice.Check(
            "ANSWER: the cache condition is arithmetic and it says r < 16",
            result["cache_ok"] == [] and result["budget_r"] == 16 and result["agreement"] < 1e-13,
            f"MLA caches r numbers per position where MHA caches 2 * d_model = {2 * D_MODEL}, so "
            f"r / {2 * D_MODEL} < 1/8 means r < {result['budget_r']}, and no rank meeting the "
            f"quality bar is below it: r={result['budget_r']} sits on 1/8, not under. Ratios "
            f"{[round(ratio[r], 3) for r in RANKS]} are linear in r and owe nothing to N or to "
            f"head count, and the MHA arm agrees with the lesson's own multi_head_attention to "
            f"{result['agreement']:.1e} -- so every uncertainty here is on the quality side",
        ),
        practice.Check(
            "ANSWER: quality first comes within 1 bit at r=36, a ratio of 0.281",
            first == 36 and bits[32] > BUDGET,
            f"one bit per dimension is a 4x mse ratio, so the bar is {4 * result['base']:.4f} "
            f"against a full-MHA {result['base']:.4f}: r=32 is {bits[32]:.2f} bits worse, r=36 is "
            f"{bits[36]:.2f}. The first rank that qualifies caches {ratio[first]:.3f} of MHA, "
            f"{ratio[first] / TARGET:.2f}x the target, so the two conditions never overlap",
        ),
        practice.Check(
            "FINDING: the floor is information, not optimisation",
            result["floor"] == 40 and bits[40] < 0 < bits[36],
            f"each position's K and V must carry its key and both values, {KEY} + 2*{VALUE} = "
            f"{result['floor']} dimensions, and the curve breaks there: {bits[32]:.2f} bits worse "
            f"at 32, {bits[36]:.2f} at 36, {bits[40]:.2f} at 40. A rank-r bottleneck cannot go "
            "below the rank of what it carries, at any learning rate",
        ),
        practice.Check(
            "FINDING: the free win is 2x and it is an improvement, not a cost",
            bits[D_MODEL] < -0.5 and ratio[D_MODEL] == 0.5,
            f"at r = d_model = {D_MODEL} the cache is halved -- one latent instead of a K and a "
            f"V -- and the model is {-bits[D_MODEL]:.2f} bits *better* than full MHA, because the "
            "shared factorisation ties K and V together. The question asks where MLA starts to "
            "hurt; the measurement says it starts by helping",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
