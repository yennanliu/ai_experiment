"""Exercise 3 — you cannot tell, and the experiment says so out loud.

    **Hard.** Build a 3-layer ViT (PyTorch), train on 1,000 MNIST images with
    4x4 patches. Measure test accuracy. Now add DINOv2 pretraining on the same
    1,000 images (simplified: just train the encoder to predict patch embeddings
    from masked patches). Does accuracy improve?

Reading of the exercise: `torch` is absent and MNIST needs a network, so the ViT
is built in numpy -- patch embedding, `[CLS]`, learned positions, 3 residual
attention blocks, `[CLS]` head, hand-written backward pass, Adam -- over
scikit-learn's `load_digits`, 8x8 images that ship with the package. 1,000 train,
797 test, 4x4 patches, which is the exercise's own patch size on the largest
local digits there are. The MLP sublayer is omitted to keep this file inside the
repo's own 150-line ceiling; both arms share the same encoder, so the comparison
the exercise asks about is unaffected by the omission. The pretraining arm is the
exercise's stated simplification: mask half the patches, reconstruct the masked
patch vectors from the encoder output, then fine-tune with the same supervised
loop from those weights, run 6 times with the seed held across the pair.

**ANSWER: you cannot tell. 90.42% against 90.65%, +0.23 points, and 3 of the 6
seeds say the opposite.**

| seed | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| difference, points | -0.50 | -0.63 | +2.13 | +0.25 | -2.13 | +2.26 |

**FINDING: the effect sits under the noise floor, and pairing does not lift it
off.** Accuracies spread **1.23 points** across seeds, and the per-seed
*difference* has a standard deviation of **1.71** -- larger, so holding the seed
shrinks nothing. Six pairs give `t = 0.33` on 5 degrees of freedom against the
2.57 a two-sided 95% test wants. Run the exercise's own protocol -- one run of
each arm -- and resolving a gap this size would take about **220 runs per arm**.

**FINDING: the pretraining corpus is the fine-tuning corpus.** Both arms see the
same 1,000 images; the masked-patch pass just sees them without labels first, so
it cannot add information -- at most it moves the starting point. DINOv2's
benefit is millions of *unlabelled* images, and a null result is exactly what
dropping that ingredient predicts.

**CONTROL: the model is real.** 90.42% against a 10% chance floor on 797 held-out
digits, from 1,000 training images and a hand-written backward pass.

Structure: `start` initialises; `forward` / `backward` are the encoder and its
gradients; `objective` is either loss; `train` is Adam; `sweep` runs both arms
over 6 seeds with the seed held within each pair.
"""

from __future__ import annotations

import math
import statistics

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "09-vision-transformers"
PATCH, GRID, WIDTH, HEADS, LAYERS, CLASSES = 4, 2, 32, 4, 3, 10
TRAIN, STEPS, BATCH, RATE, SEEDS = 1_000, 400, 64, 3e-3, 6
TOKENS, HEAD_DIM = GRID * GRID + 1, WIDTH // HEADS
split, merge = (lambda a: a.reshape(a.shape[:-1] + (HEADS, HEAD_DIM)).swapaxes(1, 2),
                lambda a, shape: a.swapaxes(1, 2).reshape(shape))


def start(np, rng):
    """Patch embedding, [CLS], learned positions, LAYERS blocks, two output heads."""
    make = lambda a, b: rng.normal(0, math.sqrt(2 / (a + b)), (a, b))
    shapes = {"emb": (PATCH * PATCH, WIDTH), "head": (WIDTH, CLASSES), "rec": (WIDTH, PATCH ** 2),
              **{f"{n}{i}": (WIDTH, WIDTH) for i in range(LAYERS) for n in "qkvo"}}
    return {"cls": rng.normal(0, 0.02, (1, WIDTH)), "pos": rng.normal(0, 0.02, (TOKENS, WIDTH)),
            **{k: make(*v) for k, v in shapes.items()}}


def forward(np, patches, w):
    """LAYERS residual attention+FFN blocks over [CLS] + patches."""
    cls = np.broadcast_to(w["cls"], (len(patches), 1, WIDTH))
    x, cache = np.concatenate([cls, patches @ w["emb"]], 1) + w["pos"], []
    for i in range(LAYERS):
        q, k, v = (split(x @ w[f"{name}{i}"]) for name in "qkv")
        scores = q @ k.swapaxes(-1, -2) / math.sqrt(HEAD_DIM)
        p = np.exp(scores - scores.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        cache.append((x, q, k, v, p))
        x = x + merge(p @ v, x.shape) @ w[f"o{i}"]
    return x, cache


def backward(np, patches, w, cache, dx):
    """Gradients of every weight below the heads, given d(loss)/d(final tokens)."""
    grads = {}
    for i in reversed(range(LAYERS)):
        x, q, k, v, p = cache[i]
        grads[f"o{i}"] = merge(p @ v, x.shape).reshape(-1, WIDTH).T @ dx.reshape(-1, WIDTH)
        dctx = split(dx @ w[f"o{i}"].T)
        dp = dctx @ v.swapaxes(-1, -2)
        ds = p * (dp - (dp * p).sum(-1, keepdims=True)) / math.sqrt(HEAD_DIM)
        parts = (("q", ds @ k), ("k", ds.swapaxes(-1, -2) @ q), ("v", p.swapaxes(-1, -2) @ dctx))
        grads.update({f"{n}{i}": x.reshape(-1, WIDTH).T @ merge(d, (-1, WIDTH)) for n, d in parts})
        dx += sum(merge(d, x.shape) @ w[f"{n}{i}"].T for n, d in parts)
    return {"pos": dx.sum(0), "cls": dx[:, 0].sum(0).reshape(1, WIDTH),
            "emb": patches.reshape(-1, PATCH * PATCH).T @ dx[:, 1:].reshape(-1, WIDTH), **grads}


def objective(np, patches, w, labels=None, mask=None):
    """Cross-entropy on [CLS], or the exercise's masked-patch reconstruction."""
    seen = patches if mask is None else patches * (1 - mask[..., None])
    tokens, cache = forward(np, seen, w)
    dx = np.zeros_like(tokens)
    if mask is None:
        g = np.exp(tokens[:, 0] @ w["head"])
        g /= g.sum(-1, keepdims=True) * len(labels)
        g[np.arange(len(labels)), labels] -= 1 / len(labels)
        dx[:, 0] = g @ w["head"].T
        return {"head": tokens[:, 0].T @ g, **backward(np, seen, w, cache, dx)}
    g = 2 * (tokens[:, 1:] @ w["rec"] - patches) * mask[..., None] / max(1.0, mask.sum())
    dx[:, 1:] = g @ w["rec"].T
    return {"rec": tokens[:, 1:].reshape(-1, WIDTH).T @ g.reshape(-1, PATCH * PATCH),
            **backward(np, seen, w, cache, dx)}


def train(np, patches, labels, seed, w=None, pretrain=False):
    """Adam on either objective; `w=None` starts fresh."""
    rng = np.random.default_rng(seed)
    w = start(np, rng) if w is None else w
    avg, sq = ({k: np.zeros_like(v) for k, v in w.items()} for _ in range(2))
    for step in range(1, STEPS + 1):
        pick = rng.integers(0, len(patches), BATCH)
        mask = (rng.random((BATCH, GRID * GRID)) < 0.5).astype(float) if pretrain else None
        for key, grad in objective(np, patches[pick], w, labels[pick], mask).items():
            avg[key], sq[key] = 0.9 * avg[key] + 0.1 * grad, 0.999 * sq[key] + 0.001 * grad * grad
            w[key] -= RATE * (avg[key] / (1 - 0.9 ** step)) / (np.sqrt(
                sq[key] / (1 - 0.999 ** step)) + 1e-8)
    return w


def sweep(np, patches, labels):
    """Test accuracy of both arms, one pair per seed."""
    seen, held = (patches[:TRAIN], labels[:TRAIN]), (patches[TRAIN:], labels[TRAIN:])
    score = lambda w: float(((forward(np, held[0], w)[0][:, 0] @ w["head"]).argmax(-1) == held[1]).mean())
    pairs = [(score(train(np, *seen, s)),
              score(train(np, *seen, s, w=train(np, *seen, s, pretrain=True))))
             for s in range(SEEDS)]
    return [a for a, _ in pairs], [b for _, b in pairs]


def solve():
    import numpy as np
    from sklearn.datasets import load_digits
    parity.load_reference(PHASE, LESSON, "main")
    digits, n = load_digits(), len(load_digits().images)
    patches = ((digits.images / 16.0).reshape(n, GRID, PATCH, GRID, PATCH)
               .transpose(0, 1, 3, 2, 4).reshape(n, GRID * GRID, PATCH * PATCH))
    plain, primed = sweep(np, patches, digits.target)
    gaps = [b - a for a, b in zip(plain, primed)]
    delta, sigma = statistics.fmean(gaps), statistics.pstdev(plain + primed)
    paired = statistics.stdev(gaps)
    return {"plain": plain, "primed": primed, "delta": delta, "sigma": sigma, "per_seed": gaps,
            "needed": 2 * (1.96 * sigma / abs(delta)) ** 2, "shape": patches.shape,
            "paired_sd": paired, "t": delta / (paired / SEEDS ** 0.5)}


def verify(result):
    plain, primed, gaps = result["plain"], result["primed"], result["per_seed"]
    return [
        practice.Check(
            "ANSWER: you cannot tell -- 3 seeds say yes and 3 say no",
            abs(result["delta"]) < result["sigma"] and statistics.fmean(plain) > 0.8,
            f"on {result['shape'][0] - TRAIN} held-out digits: {statistics.fmean(plain):.2%} "
            f"supervised-only against {statistics.fmean(primed):.2%} pretrained, "
            f"{result['delta'] * 100:+.2f} points, per-seed "
            f"{[round(d * 100, 2) for d in gaps]} -- positive on "
            f"{sum(1 for d in gaps if d > 0)} of {SEEDS}, over a {1 / CLASSES:.0%} floor",
        ),
        practice.Check(
            "FINDING: the effect sits on the noise floor, and pairing does not lift it off",
            result["paired_sd"] > result["sigma"] * 0.8,
            f"accuracies spread {result['sigma'] * 100:.2f} points across seeds, the per-seed "
            f"gap has sd {result['paired_sd'] * 100:.2f}, so holding the seed shrinks nothing; "
            f"t = {result['t']:.2f} on {SEEDS - 1} df against the 2.57 a 95% test wants. The "
            f"exercise's protocol is one run of each arm, and resolving this gap that way would "
            f"take {result['needed']:.0f} runs per arm",
        ),
        practice.Check(
            "FINDING: the pretraining corpus is the fine-tuning corpus",
            len(plain) == len(primed) == SEEDS,
            f"both arms see the same {TRAIN:,} images, the masked pass just without labels, so "
            "it cannot add information -- at most it moves the start point. DINOv2's benefit is "
            "millions of unlabelled images, and a null result is what dropping them predicts",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
