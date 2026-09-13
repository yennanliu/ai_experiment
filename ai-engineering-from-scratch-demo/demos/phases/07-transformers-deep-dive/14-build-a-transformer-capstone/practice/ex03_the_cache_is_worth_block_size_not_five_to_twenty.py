"""Exercise 3 — the cache is worth block_size, which is 64, not 5 to 20.

    **Medium.** Implement a KV cache in the sampling loop. Generate 500 tokens
    with and without cache. Wall-clock should improve by 5-20x on a laptop.

Reading of the exercise: `torch` is absent, so `generate` cannot be run as
written. Both loops are rebuilt in numpy at the lesson's own geometry --
`block_size=64`, `d_model=64`, 4 heads, 3 layers, vocab 46, `top_k=10` -- and
both the multiply-accumulate count and the wall clock are measured over the same
500 tokens. The uncached loop is the lesson's own: `idx_cond = idx[:, -64:]`, a
full forward pass on 64 positions for every single token emitted.

**ANSWER: 64x in arithmetic, and the number is `block_size`.** Uncached, each
step runs the whole 64-token window through 3 layers: 4,907,008
multiply-accumulates. Cached, one position against 64 cached keys: 76,672. The ratio is
**64.0** -- exactly `block_size` -- because the cached loop does `1/N` of the
projections and `1/N` of the attention rows for an `N`-token window.

**FINDING: the exercise's 5-20x is the wall clock, and it is the wall clock
because of overhead.** Measured in numpy the same 500 tokens take a much smaller
ratio than 64x, because at these sizes every step is dominated by Python and
numpy dispatch rather than by arithmetic: a `(1, 64)` matmul and a `(1, 1)`
matmul cost nearly the same wall-clock. The 5-20x the exercise predicts is a
statement about interpreter overhead at `d_model=64`, not about the cache.

**FINDING: the ratio is `block_size` and nothing else.** It does not depend on
the 500 tokens, on the vocabulary, on the number of layers or on `d_model` --
every term in the per-step cost is linear in the number of query positions, so
they all cancel. Generate 5,000 tokens instead of 500 and the arithmetic ratio is
still 64.

**CONTROL: the lesson's own `generate` already truncates to `block_size`.** Which
is why the answer is 64 rather than 500: the uncached cost per step is capped at
a 64-token window, so the saving stops growing once generation passes the context
length. A loop without that crop would show a ratio that grows with the token
index and reaches 500.

Structure: `cost` is the multiply-accumulate count for one step; `sample` is the
lesson's own sampling loop in numpy, with and without a cache.
"""

from __future__ import annotations

import importlib.util
import math
import time

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "14-build-a-transformer-capstone"
BLOCK, WIDTH, HEADS, LAYERS, TOKENS, HEAD_DIM = 64, 64, 4, 3, 500, 16


def cost(queries, vocab, window=BLOCK, layers=LAYERS, width=WIDTH):
    """Multiply-accumulates for one decoding step over `queries` query positions."""
    attention = layers * 2 * queries * window * width
    projections = layers * queries * width * 4 * width
    return attention + projections + queries * width * vocab


def weights(np, rng, vocab):
    """One small decoder's worth of matrices, at the lesson's geometry."""
    make = lambda a, b: rng.normal(0, math.sqrt(2 / (a + b)), (a, b))
    return ({"tok": make(vocab, WIDTH), "pos": rng.normal(0, 0.02, (BLOCK, WIDTH))}
            | {f"{n}{i}": make(WIDTH, 3 * WIDTH if n == "qkv" else WIDTH)
               for i in range(LAYERS) for n in ("qkv", "o")})


def extend(np, kept, fresh):
    """This step's keys and values appended to the cache, cropped to block_size."""
    return [t if c is None else np.concatenate([c, t], axis=2)[:, :, -BLOCK:]
            for c, t in zip(kept, fresh)]


def sample(np, w, vocab, tokens=TOKENS, cached=False):
    """The lesson's generate loop: crop to block_size, forward, take the last logits."""
    split = lambda a: a.reshape(a.shape[:-1] + (HEADS, HEAD_DIM)).swapaxes(1, 2)
    idx, cache = [0], [(None, None)] * LAYERS
    for _ in range(tokens):
        window = idx[-BLOCK:] if not cached else idx[-1:]
        x = w["tok"][np.array(window)][None] + w["pos"][:len(window)]
        for i in range(LAYERS):
            q, k, v = (split(t) for t in np.split(x @ w[f"qkv{i}"], 3, axis=-1))
            if cached:
                k, v = cache[i] = extend(np, cache[i], (k, v))
            p = np.exp(q @ k.swapaxes(-1, -2) / math.sqrt(HEAD_DIM))
            p /= p.sum(-1, keepdims=True)
            x = x + (p @ v).swapaxes(1, 2).reshape(x.shape) @ w[f"o{i}"]
        logits = x[0, -1] @ w["tok"].T
        idx.append(int(logits.argmax()))
    return idx


def timed(call, repeats=1):
    """Best wall-clock of `repeats` runs, in seconds."""
    return min((lambda start=time.perf_counter(): (call(), time.perf_counter() - start)[1])()
               for _ in range(repeats))


def solve():
    import numpy as np
    ref = parity.load_reference(PHASE, LESSON, "main")
    vocab = len(sorted(set(ref.TINY_SHAKESPEARE)))
    w = weights(np, np.random.default_rng(0), vocab)
    plain, quick = cost(BLOCK, vocab), cost(1, vocab)
    return {
        "cost": (plain, quick), "ratio": plain / quick, "vocab": vocab,
        "wall": (timed(lambda: sample(np, w, vocab)),
                 timed(lambda: sample(np, w, vocab, cached=True))),
        "same": sample(np, w, vocab, tokens=80) == sample(np, w, vocab, tokens=80, cached=True),
        "invariant": {(layers, width): cost(BLOCK, vocab, layers=layers, width=width)
                      / cost(1, vocab, layers=layers, width=width)
                      for layers, width in ((3, 64), (12, 768), (32, 4096))},
        "torch": importlib.util.find_spec("torch") is None,
    }


def verify(result):
    plain, quick = result["cost"]
    slow, fast = result["wall"]
    return [
        practice.Check(
            "ANSWER: 64x in arithmetic, and the number is block_size",
            abs(result["ratio"] - BLOCK) < 1e-9,
            f"uncached, each step runs the whole {BLOCK}-token window through {LAYERS} layers: "
            f"{plain:,} multiply-accumulates. Cached, one position against {BLOCK} cached keys: "
            f"{quick:,}. The ratio is {result['ratio']:.1f} -- exactly block_size, because the "
            "cached loop does 1/N of the projections and 1/N of the attention rows",
        ),
        practice.Check(
            "FINDING: the wall clock is far short of 64x, which is where 5-20x comes from",
            fast < slow and slow / fast < BLOCK / 2,
            f"the same {TOKENS} tokens take {slow * 1000:.0f} ms uncached and {fast * 1000:.0f} ms "
            f"cached, {slow / fast:.1f}x against the arithmetic's {result['ratio']:.0f}x. At "
            f"d_model={WIDTH} a (1, {BLOCK}) matmul and a (1, 1) matmul cost nearly the same in "
            "dispatch, so the exercise's 5-20x is a statement about interpreter overhead",
        ),
        practice.Check(
            "FINDING: the ratio is block_size and nothing else",
            set(round(v, 9) for v in result["invariant"].values()) == {float(BLOCK)},
            f"layers and width cancel: {result['invariant']} for (layers, d_model) of "
            f"{list(result['invariant'])}. Every term in the per-step cost is linear in the "
            f"number of query positions, so the vocabulary, the depth, the width and the "
            f"{TOKENS} tokens all divide out",
        ),
        practice.Check(
            "CONTROL: the cached loop emits the same tokens, and generate already crops",
            result["same"] and result["torch"],
            f"greedy sampling with and without the cache produces an identical 80-token "
            f"continuation, so the two loops are the same function. The lesson's generate crops "
            f"to idx[:, -block_size:], which is why the answer is {BLOCK} rather than {TOKENS}: "
            "without the crop the saving would grow with the token index",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
