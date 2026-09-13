"""Exercise 5 — at matched active parameters the MoE is a wash, for 1.5x the memory.

    **Hard.** Replace the single FFN per block with a 4-expert MoE. Router +
    top-2 routing. See how val loss changes at matched active parameters.

Reading of the exercise: "matched active parameters" is the whole design, so it
is matched exactly. The dense block gets one FFN of hidden `2 * d_model`; the MoE
gets 4 experts of hidden `d_model` at top-2, which activates
`2 x (2 x 64 x 64) = 16,384` weights per layer against the dense block's
`2 x 64 x 128 = 16,384`. Both arms are rebuilt in numpy at the lesson's own
configuration, run on 3 seeds and scored at the best checkpoint.

**Routing is held, and held in both directions.** The router gets no gradient,
and the gate it produces is treated as a constant in the backward pass -- so what
`step` returns is the exact gradient of a *fixed-routing* objective and not of
the full one. That is what "what four experts buy under fixed routing" means
here; Lesson 11 measures what learned routing adds on top. It also means the MoE
arm optimises a surrogate where the dense arm, whose gate is the constant 1.0,
optimises the loss itself -- a caveat on any conclusion drawn from the gap
below, and one more reason not to read 0.012 nats as a result.

**ANSWER: a wash.**

| seed | 0 | 1 | 2 | mean |
|---|---:|---:|---:|---:|
| dense FFN | 2.934 | 2.944 | **2.829** | 2.902 |
| 4-expert MoE, top-2 | **2.858** | **2.913** | 2.900 | **2.890** |

The two means are **0.012 nats** apart -- inside the **0.115** the dense arm alone
spans across the same three seeds -- and the arm with the lower mean is the one
that loses the single best run of the six.

**FINDING: matched active, 1.47x the memory.** 105,344 parameters against
155,264. The FFN portion doubles exactly -- 4 experts where 2 are used -- and
that is the trade the architecture exists to make: identical FLOPs per token,
more weights to hold.

**FINDING: there is nothing here for four experts to divide.** 899 training
characters over a 46-symbol vocabulary do not contain four separable regimes, so
each expert sees a quarter of an already tiny distribution and learns a noisier
version of what one FFN learns from all of it. That is Lesson 11's locality
finding arriving from the other direction: an expert is only worth its memory if
the tokens it sees have something in common.

Structure: `gates` is the router, written as a mask over the hidden units so that
one FFN matrix holds all four experts; `start` builds either shape; `step` is the
forward and backward pass in one function; `train` is Adam. No gradient is
emitted for the router, which is how it is held.
"""

from __future__ import annotations

import math
import statistics

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "14-build-a-transformer-capstone"
BLOCK, WIDTH, HEADS, LAYERS, BATCH, RATE = 64, 64, 4, 3, 8, 3e-4
HEAD_DIM, CHECKS, SEEDS, EXPERTS, TOP = WIDTH // HEADS, (250, 500, 1_000), 3, 4, 2


def gates(np, a, w, i, moe):
    """Top-TOP softmax routing, as a mask over hidden units; a plain 1.0 when dense."""
    if not moe:
        return 1.0
    logits = a @ w[f"r{i}"]
    kept = np.exp(logits - logits.max(-1, keepdims=True)) * (
        logits >= np.sort(logits, -1)[..., -TOP][..., None])          # zero outside the top TOP
    return np.repeat(kept / kept.sum(-1, keepdims=True), WIDTH, -1)   # WIDTH units per expert


def start(np, rng, vocab, moe):
    """One FFN of hidden 2*WIDTH per block, or EXPERTS of hidden WIDTH plus a router."""
    make = lambda a, b: rng.normal(0, math.sqrt(2 / (a + b)), (a, b))
    hidden = EXPERTS * WIDTH if moe else 2 * WIDTH          # matched active weights
    return {"tok": make(vocab, WIDTH), "pos": rng.normal(0, 0.02, (BLOCK, WIDTH)),
            **{f"qkv{i}": make(WIDTH, 3 * WIDTH) for i in range(LAYERS)},
            **{f"o{i}": make(WIDTH, WIDTH) for i in range(LAYERS)},
            **{f"u{i}": make(WIDTH, hidden) for i in range(LAYERS)},
            **{f"d{i}": make(hidden, WIDTH) for i in range(LAYERS)},
            **({f"r{i}": make(WIDTH, EXPERTS) for i in range(LAYERS)} if moe else {})}


def step(np, idx, target, w, vocab, moe, backward=True):
    """Forward, and the fixed-routing gradient if asked -- the gate is held constant."""
    split = lambda a: a.reshape(a.shape[:-1] + (HEADS, HEAD_DIM)).swapaxes(1, 2)
    merge = lambda a, s: a.swapaxes(1, 2).reshape(s)
    n = idx.shape[1]
    x, cache, mask = w["tok"][idx] + w["pos"][:n], [], np.tril(np.ones((n, n)))
    for i in range(LAYERS):
        q, k, v = (split(t) for t in np.split(x @ w[f"qkv{i}"], 3, axis=-1))
        scores = np.where(mask > 0, q @ k.swapaxes(-1, -2) / math.sqrt(HEAD_DIM), -1e30)
        p = np.exp(scores - scores.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        ctx = (p @ v).swapaxes(1, 2).reshape(x.shape)
        a = x + ctx @ w[f"o{i}"]
        g = gates(np, a, w, i, moe)          # held: no gradient flows back through it
        h = np.maximum(a @ w[f"u{i}"], 0) * g
        cache.append((x, q, k, v, p, ctx, a, h, g))
        x = a + h @ w[f"d{i}"]
    logits, rows = x @ w["tok"].T, idx.size
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    loss = float(-np.log(probs.reshape(-1, vocab)[np.arange(rows), target.ravel()] + 1e-12).mean())
    if not backward:
        return loss, None
    probs.reshape(-1, vocab)[np.arange(rows), target.ravel()] -= 1
    top = probs / rows
    G, dx = {"tok": top.reshape(-1, vocab).T @ x.reshape(-1, WIDTH)}, top @ w["tok"]
    for i in reversed(range(LAYERS)):
        xi, q, k, v, p, ctx, a, h, g = cache[i]
        G[f"d{i}"] = h.reshape(-1, h.shape[-1]).T @ dx.reshape(-1, WIDTH)
        dh = (dx @ w[f"d{i}"].T) * (h > 0) * g          # the gate scales the gradient too
        G[f"u{i}"] = a.reshape(-1, WIDTH).T @ dh.reshape(-1, h.shape[-1])
        da = dx + dh @ w[f"u{i}"].T
        G[f"o{i}"] = ctx.reshape(-1, WIDTH).T @ da.reshape(-1, WIDTH)
        dctx = split(da @ w[f"o{i}"].T)
        dp = dctx @ v.swapaxes(-1, -2)
        ds = p * (dp - (dp * p).sum(-1, keepdims=True)) / math.sqrt(HEAD_DIM)
        cat = np.concatenate([merge(t, xi.shape) for t in
                              (ds @ k, ds.swapaxes(-1, -2) @ q, p.swapaxes(-1, -2) @ dctx)], -1)
        G[f"qkv{i}"] = xi.reshape(-1, WIDTH).T @ cat.reshape(-1, 3 * WIDTH)
        dx = da + cat @ w[f"qkv{i}"].T
    G["pos"] = dx.sum(0)
    np.add.at(G["tok"], idx.ravel(), dx.reshape(-1, WIDTH))
    return loss, G


def batch(np, rng, data, split_at, which):
    """(inputs, next tokens) drawn the way get_batch draws them: with replacement."""
    src = data[:split_at] if which == "train" else data[split_at:]
    ix = rng.integers(0, len(src) - BLOCK - 1, BATCH)
    return tuple(np.stack([src[i + s:i + s + BLOCK] for i in ix]) for s in (0, 1))


def train(np, ref, moe, steps, seed=0, report=()):
    """Adam at the lesson's own lr; returns (weights, {step: val loss})."""
    data, vocab, split_at = ref.CORPUS
    rng, w = np.random.default_rng(seed), start(np, np.random.default_rng(seed), vocab, moe)
    avg, sq, seen = *({k: np.zeros_like(v) for k, v in w.items()} for _ in range(2)), {}
    for t in range(1, steps + 1):
        idx, target = batch(np, rng, data, split_at, "train")
        loss, G = step(np, idx, target, w, vocab, moe)
        for key, grad in G.items():
            avg[key], sq[key] = 0.9 * avg[key] + 0.1 * grad, 0.999 * sq[key] + 0.001 * grad * grad
            w[key] -= RATE * (avg[key] / (1 - 0.9 ** t)) / (
                np.sqrt(sq[key] / (1 - 0.999 ** t)) + 1e-8)
        if t in report:
            val = np.random.default_rng(999)
            seen[t] = sum(step(np, *batch(np, val, data, split_at, "val"), w, vocab, moe,
                               backward=False)[0] for _ in range(20)) / 20
    return w, seen


def solve():
    import numpy as np
    ref = parity.load_reference(PHASE, LESSON, "main")
    chars = sorted(set(ref.TINY_SHAKESPEARE))
    data = np.array([chars.index(c) for c in ref.TINY_SHAKESPEARE])
    ref.CORPUS, runs, sizes = (data, len(chars), int(0.9 * len(data))), {}, {}
    for name, moe in (("dense", False), ("moe", True)):
        pairs = [train(np, ref, moe, max(CHECKS), seed=s, report=CHECKS) for s in range(SEEDS)]
        runs[name] = [min(curve.values()) for _, curve in pairs]
        sizes[name] = sum(v.size for v in pairs[0][0].values())   # both arms, all weights
    return {"runs": runs, "sizes": sizes, "spread": max(runs["dense"]) - min(runs["dense"]),
            "active": {"dense": 4 * WIDTH * WIDTH, "moe": TOP * 2 * WIDTH * WIDTH},
            "means": {k: statistics.fmean(v) for k, v in runs.items()},
            "wins": sum(m < d for d, m in zip(runs["dense"], runs["moe"]))}


def verify(result):
    runs, means, sizes = result["runs"], result["means"], result["sizes"]
    return [
        practice.Check(
            "ANSWER: at matched active parameters it is a wash",
            abs(means["moe"] - means["dense"]) < result["spread"],
            f"best val over {list(CHECKS)}: dense {[round(v, 3) for v in runs['dense']]} against "
            f"MoE {[round(v, 3) for v in runs['moe']]} -- means {means['dense']:.3f} and "
            f"{means['moe']:.3f}, inside the {result['spread']:.3f} the dense arm alone spans",
        ),
        practice.Check(
            "FINDING: matched active, 1.5x the memory",
            result["active"]["dense"] == result["active"]["moe"]
            and sizes["moe"] > 1.4 * sizes["dense"],
            f"{EXPERTS} experts of hidden {WIDTH} at top-{TOP} activate "
            f"{result['active']['moe']:,} FFN weights per layer, exactly the dense block's "
            f"{result['active']['dense']:,} at hidden {2 * WIDTH}, for {sizes['dense']:,} weights "
            f"against {sizes['moe']:,} -- {sizes['moe'] / sizes['dense']:.2f}x the memory to hold, "
            f"and {result['wins']} of {SEEDS} seeds: 899 characters offer no specialisation to "
            "divide, which is Lesson 11 arriving from the other direction",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
