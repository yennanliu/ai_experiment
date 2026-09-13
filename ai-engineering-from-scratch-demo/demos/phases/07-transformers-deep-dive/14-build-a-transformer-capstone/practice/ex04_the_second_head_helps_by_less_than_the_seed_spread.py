"""Exercise 4 — the second head helps by less than the seed-to-seed spread.

    **Hard.** Add a second head to the model that predicts the next-plus-one
    token (MTP -- Multi-Token Prediction from DeepSeek-V3). Train jointly. Does
    it help?

Reading of the exercise: `torch` is absent, so both arms are rebuilt in numpy at
the lesson's own configuration -- `block_size=64`, `d_model=64`, 4 heads, 3
layers, `lr=3e-4`, Adam, the embedded text split 90/10 -- differing only in
whether a second `d_model x vocab` head predicts token `t+2` alongside the first
predicting `t+1`. The auxiliary loss is weighted 0.5, DeepSeek-V3's own setting,
and only the *next-token* loss is reported, so the two arms are scored on the
same objective. The SwiGLU sublayer is omitted from both. Each arm is run on 3
seeds and scored at its best checkpoint, because Exercise 1 shows the last step
is the wrong place to look.

**ANSWER: marginally, and not on every seed.**

| seed | 0 | 1 | 2 | mean |
|---|---:|---:|---:|---:|
| single head | 3.188 | 3.033 | **3.022** | 3.081 |
| plus MTP | **2.879** | **3.019** | 3.083 | **2.993** |

MTP is lower on **2 of 3** seeds, by **0.088 nats** on average.

**FINDING: the effect is smaller than the seed-to-seed spread.** The single-head
arm alone spans **0.166** across the same three seeds -- twice the effect. This
is the same shape as Exercise 3 of Lesson 09: a real-looking difference that one
run of each arm would report with a sign chosen by the draw.

**FINDING: what it plausibly does here is regularise.** The second head costs
2,944 parameters -- one untied `d_model x vocab` matrix -- and asks the same
hidden state to carry information about two future tokens instead of one. On 899
training characters, where Exercise 1 shows the model memorising within 500
steps, an extra constraint on the representation is worth more than an extra
prediction. DeepSeek-V3's version is about *speculative decoding throughput*, and
that benefit cannot appear in a validation loss at all.

Structure: `step` is the forward and backward pass in one function, carrying both
heads; `train` is Adam; `batch` returns the token, its successor and its
successor's successor.
"""

from __future__ import annotations

import math
import statistics

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "14-build-a-transformer-capstone"
BLOCK, WIDTH, HEADS, LAYERS, BATCH, RATE = 64, 64, 4, 3, 8, 3e-4
HEAD_DIM, CHECKS, SEEDS, AUX = WIDTH // HEADS, (250, 500, 1_000), 3, 0.5


def start(np, rng, vocab, mtp):
    """Token embedding, positions, LAYERS blocks, and a second head when MTP is on."""
    make = lambda a, b: rng.normal(0, math.sqrt(2 / (a + b)), (a, b))
    return {"tok": make(vocab, WIDTH), "pos": rng.normal(0, 0.02, (BLOCK, WIDTH)),
            **({"head2": make(WIDTH, vocab)} if mtp else {}),
            **{f"{n}{i}": make(WIDTH, 3 * WIDTH if n == "qkv" else WIDTH)
               for i in range(LAYERS) for n in ("qkv", "o")}}


def step(np, idx, targets, w, vocab, backward=True):
    """Forward pass, and the gradient of every weight if asked. Written as one pass."""
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
        cache.append((x, q, k, v, p, ctx))
        x = x + ctx @ w[f"o{i}"]
    def head(logits, target):
        rows = idx.size
        p = np.exp(logits - logits.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        flat = p.reshape(-1, vocab)[np.arange(rows), target.ravel()]
        p.reshape(-1, vocab)[np.arange(rows), target.ravel()] -= 1
        return float(-np.log(flat + 1e-12).mean()), p / rows

    loss, g = head(x @ w["tok"].T, targets[0])
    if not backward:
        return loss, None
    G, dx = {"tok": g.reshape(-1, vocab).T @ x.reshape(-1, WIDTH)}, g @ w["tok"]
    if "head2" in w:
        second, g2 = head(x @ w["head2"], targets[1])
        loss += AUX * second
        G["head2"] = AUX * x.reshape(-1, WIDTH).T @ g2.reshape(-1, vocab)
        dx = dx + AUX * g2 @ w["head2"].T
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
    ix = rng.integers(0, len(src) - BLOCK - 2, BATCH)
    return tuple(np.stack([src[i + o:i + o + BLOCK] for i in ix]) for o in (0, 1, 2))


def evaluate(np, w, data, split_at, vocab, batches=20):
    """Validation loss over 20 batches, so all 36 windows are covered several times."""
    rng, seen = np.random.default_rng(999), []
    for _ in range(batches):
        idx, first, _ = batch(np, rng, data, split_at, "val")
        seen.append(step(np, idx, (first,), w, vocab, backward=False)[0])
    return sum(seen) / batches


def train(np, ref, mtp, steps, seed=0, report=()):
    """Adam at the lesson's own lr; returns (weights, {step: val loss})."""
    data, vocab, split_at = ref.CORPUS
    rng, w = np.random.default_rng(seed), start(np, np.random.default_rng(seed), vocab, mtp)
    avg, sq, seen = *({k: np.zeros_like(v) for k, v in w.items()} for _ in range(2)), {}
    for t in range(1, steps + 1):
        idx, first, second = batch(np, rng, data, split_at, "train")
        loss, G = step(np, idx, (first, second), w, vocab)
        for key, grad in G.items():
            avg[key], sq[key] = 0.9 * avg[key] + 0.1 * grad, 0.999 * sq[key] + 0.001 * grad * grad
            w[key] -= RATE * (avg[key] / (1 - 0.9 ** t)) / (
                np.sqrt(sq[key] / (1 - 0.999 ** t)) + 1e-8)
        if t in report:
            seen[t] = evaluate(np, w, data, split_at, vocab)
    return w, seen


def solve():
    import numpy as np
    ref = parity.load_reference(PHASE, LESSON, "main")
    chars = sorted(set(ref.TINY_SHAKESPEARE))
    data = np.array([chars.index(c) for c in ref.TINY_SHAKESPEARE])
    ref.CORPUS = (data, len(chars), int(0.9 * len(data)))
    runs, sizes = {}, {}
    for name, mtp in (("single", False), ("mtp", True)):
        pairs = [train(np, ref, mtp, max(CHECKS), seed=s, report=CHECKS) for s in range(SEEDS)]
        runs[name] = [min(curve.values()) for _, curve in pairs]
        sizes[name] = sum(v.size for v in pairs[0][0].values())
    return {"runs": runs, "sizes": sizes, "vocab": len(chars),
            "means": {k: statistics.fmean(v) for k, v in runs.items()},
            "wins": sum(m < s for s, m in zip(runs["single"], runs["mtp"])),
            "spread": max(runs["single"]) - min(runs["single"])}


def verify(result):
    runs, means = result["runs"], result["means"]
    return [
        practice.Check(
            "ANSWER: marginally, and not on every seed",
            means["mtp"] < means["single"] and result["wins"] < SEEDS,
            f"best val loss over checkpoints {list(CHECKS)}: single-head "
            f"{[round(v, 3) for v in runs['single']]} against MTP "
            f"{[round(v, 3) for v in runs['mtp']]}, means {means['single']:.3f} and "
            f"{means['mtp']:.3f}. MTP is lower on {result['wins']} of {SEEDS} seeds, by "
            f"{means['single'] - means['mtp']:.3f} nats on average",
        ),
        practice.Check(
            "FINDING: the effect is smaller than the seed-to-seed spread",
            means["single"] - means["mtp"] < result["spread"],
            f"the single-head arm alone spans {result['spread']:.3f} across the same three seeds, "
            f"against an effect of {means['single'] - means['mtp']:.3f}. The second head costs "
            f"{result['sizes']['mtp'] - result['sizes']['single']:,} parameters -- one "
            f"d_model x vocab matrix, untied -- and buys a difference this design cannot resolve. "
            "What it plausibly does at 899 training characters is regularise, not predict",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
