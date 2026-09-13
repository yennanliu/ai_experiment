"""Exercise 3 — truncating layers gives a broken model, not a smaller one.

    **Hard.** Implement a tiny Medusa: take the capstone GPT from Lesson 14, add
    3 extra LM heads that predict positions t+2, t+3, t+4. Train on
    tinyshakespeare with a joint multi-head loss. Compare acceptance rates vs a
    vanilla draft made by truncating the same model.

Reading of the exercise: `torch` is absent, so the capstone is rebuilt in numpy
as in Lesson 14 -- `block_size=64`, `d_model=64`, 4 heads, 3 layers, `lr=3e-4`,
Adam, the embedded text split 90/10, SwiGLU omitted -- with three extra
`d_model x vocab` heads and a joint loss weighting each of them 0.5. Acceptance
is measured the way a greedy verifier measures it: head `k`'s argmax against the
true token at `t+k`, over the validation half. The truncated draft is the first
layer of the same trained network, so "the same model" is literal.

**ANSWER: every Medusa head beats the truncated draft, the weakest by 1.4x.**

| draft | acceptance |
|---|---:|
| head 1 (t+1) | 0.202 |
| head 2 (t+2) | **0.214** |
| head 3 (t+3) | 0.114 |
| head 4 (t+4) | 0.125 |
| 1-layer truncation of the same model | **0.079** |
| always predict the commonest character | **0.192** |

**FINDING: only two of the four heads clear the unigram baseline.** Guessing a
space every time scores **0.192** on the validation half. Heads 1 and 2 clear it
by +0.010 and +0.022; heads 3 and 4 land *below* it. On 899 training characters
both arms of the comparison have learned almost nothing, so the ranking is the
only part of the result that survives -- which is Lesson 14's finding about this
corpus, in a different metric.

**FINDING: truncating layers is not the same as having a smaller model.** The
1-layer prefix agrees with the full network **0.079** of the time -- less than
half what guessing the commonest character gets. The residual stream it hands to
the head was built for three layers of refinement, so removing two does not
produce a weaker model, it produces a broken one. Real draft models are trained,
not sliced, and this is the measurement that says why.

**CONTROL: head 2 outscoring head 1 is the tell.** Predicting two characters
ahead cannot genuinely be easier than predicting one. That the two come out
0.202 and 0.214 says both are sitting on the unigram distribution, where the
ordering is noise.

Structure: `start` adds the extra heads; `step` is the forward and backward pass
in one function, carrying the joint loss; `accept` scores every head and the
truncation against the same batches.
"""

from __future__ import annotations

import math
import statistics

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "14-build-a-transformer-capstone"
BLOCK, WIDTH, HEADS, LAYERS, BATCH, RATE = 64, 64, 4, 3, 8, 3e-4
HEAD_DIM, STEPS, EXTRA, AUX = WIDTH // HEADS, 1_000, 4, 0.5


def start(np, rng, vocab):
    """The capstone's weights plus EXTRA-1 Medusa heads for t+2 .. t+EXTRA."""
    make = lambda a, b: rng.normal(0, math.sqrt(2 / (a + b)), (a, b))
    return {"tok": make(vocab, WIDTH), "pos": rng.normal(0, 0.02, (BLOCK, WIDTH)),
            **{f"head{k}": make(WIDTH, vocab) for k in range(2, EXTRA + 1)},
            **{f"{n}{i}": make(WIDTH, 3 * WIDTH if n == "qkv" else WIDTH)
               for i in range(LAYERS) for n in ("qkv", "o")}}


def step(np, idx, targets, w, vocab, backward=True, layers=LAYERS):
    """Forward pass, and the gradient of every weight if asked. Written as one pass."""
    split = lambda a: a.reshape(a.shape[:-1] + (HEADS, HEAD_DIM)).swapaxes(1, 2)
    merge = lambda a, s: a.swapaxes(1, 2).reshape(s)
    n = idx.shape[1]
    x, cache, mask = w["tok"][idx] + w["pos"][:n], [], np.tril(np.ones((n, n)))
    for i in range(layers):
        q, k, v = (split(t) for t in np.split(x @ w[f"qkv{i}"], 3, axis=-1))
        scores = np.where(mask > 0, q @ k.swapaxes(-1, -2) / math.sqrt(HEAD_DIM), -1e30)
        p = np.exp(scores - scores.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        ctx = (p @ v).swapaxes(1, 2).reshape(x.shape)
        cache.append((x, q, k, v, p, ctx))
        x = x + ctx @ w[f"o{i}"]
    def head(logits, target):
        rows, p = idx.size, np.exp(logits - logits.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        flat = p.reshape(-1, vocab)[np.arange(rows), target.ravel()]
        p.reshape(-1, vocab)[np.arange(rows), target.ravel()] -= 1
        return float(-np.log(flat + 1e-12).mean()), p / rows
    if not backward:
        return 0.0, x
    loss, g = head(x @ w["tok"].T, targets[0])
    G, dx = {"tok": g.reshape(-1, vocab).T @ x.reshape(-1, WIDTH)}, g @ w["tok"]
    for k in range(2, EXTRA + 1):
        extra, gk = head(x @ w[f"head{k}"], targets[k - 1])
        loss += AUX * extra
        G[f"head{k}"] = AUX * x.reshape(-1, WIDTH).T @ gk.reshape(-1, vocab)
        dx = dx + AUX * gk @ w[f"head{k}"].T
    for i in reversed(range(LAYERS)):
        xi, q, k, v, p, ctx = cache[i]
        G[f"o{i}"] = ctx.reshape(-1, WIDTH).T @ dx.reshape(-1, WIDTH)
        dctx = split(dx @ w[f"o{i}"].T)
        dp = dctx @ v.swapaxes(-1, -2)
        ds = p * (dp - (dp * p).sum(-1, keepdims=True)) / math.sqrt(HEAD_DIM)
        cat = np.concatenate([merge(t, xi.shape) for t in
                              (ds @ k, ds.swapaxes(-1, -2) @ q, p.swapaxes(-1, -2) @ dctx)], -1)
        G[f"qkv{i}"] = xi.reshape(-1, WIDTH).T @ cat.reshape(-1, 3 * WIDTH)
        dx = dx + cat @ w[f"qkv{i}"].T
    G["pos"] = dx.sum(0)
    np.add.at(G["tok"], idx.ravel(), dx.reshape(-1, WIDTH))
    return loss, G


def batch(np, rng, data, split_at, which):
    """(inputs, next tokens) drawn the way get_batch draws them: with replacement."""
    src = data[:split_at] if which == "train" else data[split_at:]
    ix = rng.integers(0, len(src) - BLOCK - EXTRA, BATCH)
    return tuple(np.stack([src[i + o:i + o + BLOCK] for i in ix]) for o in range(EXTRA + 1))

def accept(np, w, data, split_at, vocab, batches=40):
    """(per-head acceptance, truncated-draft agreement, unigram baseline) on the val half."""
    rng, hits, truncated, total, modes = np.random.default_rng(999), [0] * EXTRA, 0, 0, []
    mode = np.bincount(data[:split_at]).argmax()
    for _ in range(batches):
        rows = batch(np, rng, data, split_at, "val")
        full = step(np, rows[0], rows[1:], w, vocab, backward=False)[1]
        greedy = (full @ w["tok"].T).argmax(-1)
        hits[0] += int((greedy == rows[1]).sum())
        for k in range(2, EXTRA + 1):
            hits[k - 1] += int(((full @ w[f"head{k}"]).argmax(-1) == rows[k]).sum())
        short = step(np, rows[0], rows[1:], w, vocab, backward=False, layers=1)[1]
        truncated += int(((short @ w["tok"].T).argmax(-1) == greedy).sum())
        modes.append(float((rows[1] == mode).mean()))
        total += rows[1].size
    return [h / total for h in hits], truncated / total, sum(modes) / batches

def train(np, ref, steps=STEPS, seed=0):
    """Adam at the lesson's own lr over the joint multi-head loss."""
    data, vocab, split_at = ref.CORPUS
    rng, w = np.random.default_rng(seed), start(np, np.random.default_rng(seed), vocab)
    avg, sq = ({k: np.zeros_like(v) for k, v in w.items()} for _ in range(2))
    for t in range(1, steps + 1):
        rows = batch(np, rng, data, split_at, "train")
        for key, grad in step(np, rows[0], rows[1:], w, vocab)[1].items():
            avg[key], sq[key] = 0.9 * avg[key] + 0.1 * grad, 0.999 * sq[key] + 0.001 * grad * grad
            w[key] -= RATE * (avg[key] / (1 - 0.9 ** t)) / (
                np.sqrt(sq[key] / (1 - 0.999 ** t)) + 1e-8)
    return w


def solve():
    import numpy as np
    ref = parity.load_reference(PHASE, LESSON, "main")
    chars = sorted(set(ref.TINY_SHAKESPEARE))
    data = np.array([chars.index(c) for c in ref.TINY_SHAKESPEARE])
    ref.CORPUS = (data, len(chars), int(0.9 * len(data)))
    heads, truncated, baseline = accept(np, train(np, ref), data,
                                        ref.CORPUS[2], len(chars))
    return {"heads": heads, "truncated": truncated, "baseline": baseline,
            "above": [k + 1 for k, a in enumerate(heads) if a > baseline]}


def verify(result):
    heads, baseline = result["heads"], result["baseline"]
    return [
        practice.Check(
            "ANSWER: the Medusa heads beat the truncated draft, 1.4x to 2.7x",
            min(heads) > result["truncated"],
            f"acceptance by head: {[round(a, 3) for a in heads]} for t+1..t+{EXTRA}, against "
            f"{result['truncated']:.3f} for a 1-layer prefix of the same trained model -- every "
            f"head beats it, the weakest by {min(heads) / result['truncated']:.1f}x",
        ),
        practice.Check(
            "FINDING: only two of the four clear the always-predict-a-space baseline",
            result["above"] == [1, 2] and heads[2] < baseline and heads[3] < baseline,
            f"the commonest character alone scores {baseline:.3f} on the validation half. Heads "
            f"{result['above']} clear it, by {heads[0] - baseline:+.3f} and "
            f"{heads[1] - baseline:+.3f}; heads 3 and 4 come in at {heads[2]:.3f} and "
            f"{heads[3]:.3f}, below it. On 899 training characters both arms have learned almost "
            "nothing, so the ranking is the only part that survives",
        ),
        practice.Check(
            "FINDING: truncating layers is not the same as having a smaller model",
            result["truncated"] < baseline / 2,
            f"the 1-layer prefix agrees with the full network {result['truncated']:.3f} of the "
            f"time -- less than half the {baseline:.3f} that guessing the commonest character "
            "gets. The residual stream it hands the head was built for three layers of "
            "refinement, so cutting two off gives a broken model rather than a weaker one. Real "
            "draft models are trained, not sliced",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
