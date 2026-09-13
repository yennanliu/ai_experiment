"""Exercise 2 — RoPE is lower on every seed, with 4,096 fewer parameters.

    **Medium.** Replace learned positional embeddings with RoPE. Apply the
    rotation to Q and K inside `MultiHeadAttention`. Train and verify val loss is
    at least as low.

Reading of the exercise: `torch` is absent, so both arms are rebuilt in numpy at
the lesson's own configuration -- `block_size=64`, `d_model=64`, 4 heads, 3
layers, `lr=3e-4`, Adam, the embedded text split 90/10 -- differing only in how
position enters. The learned arm adds a `(64, 64)` table to the embeddings; the
RoPE arm has no table and rotates Q and K inside attention instead. The SwiGLU
sublayer is omitted from both, so the comparison is unaffected. Each arm is run
on 3 seeds and scored at its best checkpoint, because Exercise 1 shows the last
step is the wrong place to look.

**ANSWER: at least as low -- lower, on every seed.**

| seed | 0 | 1 | 2 | mean |
|---|---:|---:|---:|---:|
| learned positions | 3.151 | 2.999 | 2.988 | 3.046 |
| RoPE | **2.837** | **2.987** | **2.988** | **2.937** |

**FINDING: and it does it with 4,096 fewer parameters.** 56,192 against 52,096 --
the `block_size x d_model` positional table is **7.3% of the model** and it is
simply not there. The rotation is a function of the index, so there is nothing to
learn and nothing to overfit; on 899 training characters that is the whole
difference.

**CONTROL: the gradient of the rotation is the rotation transposed.** `rope` with
`sign=-1` is the backward pass, exactly -- a rotation is orthogonal, so its
transpose is its inverse, and no separate derivative is needed. That is why the
substitution costs nine lines.

Structure: `rope` is the rotation and its transpose; `step` is the forward and
backward pass in one function; `train` is Adam at the lesson's own rate.
"""

from __future__ import annotations

import math
import statistics

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "14-build-a-transformer-capstone"
BLOCK, WIDTH, HEADS, LAYERS, BATCH, RATE = 64, 64, 4, 3, 8, 3e-4
HEAD_DIM, CHECKS, SEEDS = WIDTH // HEADS, (250, 500, 1_000), 3


def rope(np, a, trig, sign=1.0):
    """Rotate even/odd pairs by pos*theta; sign=-1 is the transpose, which is the gradient."""
    cos, sin, out = trig[0][:a.shape[-2]], sign * trig[1][:a.shape[-2]], np.empty_like(a)
    out[..., 0::2] = a[..., 0::2] * cos - a[..., 1::2] * sin
    out[..., 1::2] = a[..., 0::2] * sin + a[..., 1::2] * cos
    return out


def start(np, rng, vocab, mode):
    """Token embedding, LAYERS attention blocks, and learned positions unless RoPE."""
    make = lambda a, b: rng.normal(0, math.sqrt(2 / (a + b)), (a, b))
    return {"tok": make(vocab, WIDTH),
            **({} if mode == "rope" else {"pos": rng.normal(0, 0.02, (BLOCK, WIDTH))}),
            **{f"{n}{i}": make(WIDTH, 3 * WIDTH if n == "qkv" else WIDTH)
               for i in range(LAYERS) for n in ("qkv", "o")}}


def step(np, idx, target, w, vocab, mode, trig, backward=True):
    """Forward pass, and the gradient of every weight if asked. Written as one pass."""
    split = lambda a: a.reshape(a.shape[:-1] + (HEADS, HEAD_DIM)).swapaxes(1, 2)
    merge = lambda a, s: a.swapaxes(1, 2).reshape(s)
    n = idx.shape[1]
    x, cache, mask = w["tok"][idx] + (0 if mode == "rope" else w["pos"][:n]), [], np.tril(
        np.ones((n, n)))
    for i in range(LAYERS):
        q, k, v = (split(t) for t in np.split(x @ w[f"qkv{i}"], 3, axis=-1))
        q, k = (rope(np, q, trig), rope(np, k, trig)) if mode == "rope" else (q, k)
        scores = np.where(mask > 0, q @ k.swapaxes(-1, -2) / math.sqrt(HEAD_DIM), -1e30)
        p = np.exp(scores - scores.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        ctx = (p @ v).swapaxes(1, 2).reshape(x.shape)
        cache.append((x, q, k, v, p, ctx))
        x = x + ctx @ w[f"o{i}"]
    logits, rows = x @ w["tok"].T, idx.size
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    loss = float(-np.log(probs.reshape(-1, vocab)[np.arange(rows), target.ravel()] + 1e-12).mean())
    if not backward:
        return loss, None
    probs.reshape(-1, vocab)[np.arange(rows), target.ravel()] -= 1
    g = probs / rows
    G, dx = {"tok": g.reshape(-1, vocab).T @ x.reshape(-1, WIDTH)}, g @ w["tok"]
    for i in reversed(range(LAYERS)):
        xi, q, k, v, p, ctx = cache[i]
        G[f"o{i}"] = ctx.reshape(-1, WIDTH).T @ dx.reshape(-1, WIDTH)
        dctx = split(dx @ w[f"o{i}"].T)
        dp = dctx @ v.swapaxes(-1, -2)
        ds = p * (dp - (dp * p).sum(-1, keepdims=True)) / math.sqrt(HEAD_DIM)
        dq, dk, dv = ds @ k, ds.swapaxes(-1, -2) @ q, p.swapaxes(-1, -2) @ dctx
        if mode == "rope":
            dq, dk = rope(np, dq, trig, -1.0), rope(np, dk, trig, -1.0)
        cat = np.concatenate([merge(t, xi.shape) for t in (dq, dk, dv)], -1)
        G[f"qkv{i}"] = xi.reshape(-1, WIDTH).T @ cat.reshape(-1, 3 * WIDTH)
        dx = dx + cat @ w[f"qkv{i}"].T
    if mode != "rope":
        G["pos"] = dx.sum(0)
    np.add.at(G["tok"], idx.ravel(), dx.reshape(-1, WIDTH))
    return loss, G


def batch(np, rng, data, split_at, which):
    """(inputs, next tokens) drawn the way get_batch draws them: with replacement."""
    src = data[:split_at] if which == "train" else data[split_at:]
    ix = rng.integers(0, len(src) - BLOCK - 1, BATCH)
    return tuple(np.stack([src[i + o:i + o + BLOCK] for i in ix]) for o in (0, 1))


def train(np, ref, mode, steps, seed=0, report=()):
    """Adam at the lesson's own lr; returns (weights, {step: val loss})."""
    data, vocab, split_at = ref.CORPUS
    theta = np.arange(BLOCK)[:, None] / (10_000 ** (2 * np.arange(HEAD_DIM // 2) / HEAD_DIM))
    rng, trig = np.random.default_rng(seed), (np.cos(theta), np.sin(theta))
    w = start(np, np.random.default_rng(seed), vocab, mode)
    avg, sq, seen = *({k: np.zeros_like(v) for k, v in w.items()} for _ in range(2)), {}
    for t in range(1, steps + 1):
        idx, target = batch(np, rng, data, split_at, "train")
        loss, G = step(np, idx, target, w, vocab, mode, trig)
        for key, grad in G.items():
            avg[key], sq[key] = 0.9 * avg[key] + 0.1 * grad, 0.999 * sq[key] + 0.001 * grad * grad
            w[key] -= RATE * (avg[key] / (1 - 0.9 ** t)) / (
                np.sqrt(sq[key] / (1 - 0.999 ** t)) + 1e-8)
        if t in report:                        # 20 batches covers all 36 val windows
            val = np.random.default_rng(999)
            seen[t] = sum(step(np, *batch(np, val, data, split_at, "val"), w, vocab, mode,
                               trig, backward=False)[0] for _ in range(20)) / 20
    return w, seen


def solve():
    import numpy as np
    ref = parity.load_reference(PHASE, LESSON, "main")
    chars = sorted(set(ref.TINY_SHAKESPEARE))
    data = np.array([chars.index(c) for c in ref.TINY_SHAKESPEARE])
    ref.CORPUS = (data, len(chars), int(0.9 * len(data)))
    runs, sizes = {}, {}
    for mode in ("learned", "rope"):
        pairs = [train(np, ref, mode, max(CHECKS), seed=s, report=CHECKS) for s in range(SEEDS)]
        runs[mode] = [min(curve.values()) for _, curve in pairs]
        sizes[mode] = sum(v.size for v in pairs[0][0].values())
    return {"runs": runs, "sizes": sizes, "saved": sizes["learned"] - sizes["rope"],
            "means": {m: statistics.fmean(v) for m, v in runs.items()},
            "wins": sum(r < l for l, r in zip(runs["learned"], runs["rope"]))}


def verify(result):
    runs, means, sizes = result["runs"], result["means"], result["sizes"]
    return [
        practice.Check(
            "ANSWER: at least as low -- lower, on every seed",
            result["wins"] == SEEDS and means["rope"] < means["learned"],
            f"best val loss over checkpoints {list(CHECKS)}: learned "
            f"{[round(v, 3) for v in runs['learned']]} against RoPE "
            f"{[round(v, 3) for v in runs['rope']]}, means {means['learned']:.3f} and "
            f"{means['rope']:.3f} -- RoPE lower on {result['wins']} of {SEEDS}",
        ),
        practice.Check(
            "FINDING: and it does it with 4,096 fewer parameters",
            result["saved"] == BLOCK * WIDTH,
            f"{sizes['learned']:,} against {sizes['rope']:,} -- the block_size x d_model "
            f"positional table, {result['saved']:,} weights or "
            f"{result['saved'] / sizes['learned']:.1%} of the model, is not there at all -- the "
            f"rotation is a function of the index. Both arms still end above "
            f"{min(means.values()):.2f} on 899 training characters, so this compares two "
            "ceilings; Exercise 1 has the curves underneath",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
