"""Exercise 1 — it never gets under 2.0, and it stopped improving before step 500.

    **Easy.** Run `code/main.py`. Verify your trained model's final-step
    validation loss is under 2.0. Change `max_steps` from 2,000 to 5,000 -- does
    val loss keep improving?

Reading of the exercise: `torch` returns None from `find_spec`, so `try_train`
prints its install message and trains nothing. The model is rebuilt in numpy at
the lesson's own configuration -- `block_size=64`, `d_model=64`, 4 heads, 3
layers, `batch_size=8`, `lr=3e-4`, Adam, the lesson's own embedded text split
90/10 -- with a hand-written backward pass. The SwiGLU sublayer is omitted to
keep this file inside the repo's 150-line ceiling; it changes the loss level and
not the shape of the curve, and Exercise 5 measures what it is worth.

**ANSWER: no, and it never reaches 2.0 either.**

| step | 25 | 100 | 500 | 1000 | 2000 | 5000 |
|---|---:|---:|---:|---:|---:|---:|
| train | 3.66 | 3.24 | 2.58 | 2.03 | 1.63 | **1.05** |
| val | 3.61 | 3.22 | 3.17 | **3.15** | 3.24 | **4.55** |

Validation bottoms out around step 1,000 at **3.15** and rises from there. At
5,000 steps it is **4.55** -- past `ln 46 = 3.83`, the loss of predicting the 46
characters uniformly at random. Training for longer makes the model worse than
having no model.

**FINDING: 63 parameters per training character.** The embedded text is **999
characters**; the 90/10 split leaves **899** for training and **100** for
validation. The model is 56,192 parameters. There is no configuration of this
script in which the validation loss is a measurement rather than a memorisation
counter.

**FINDING: the validation set is 100 characters and each eval sees 36 windows.**
`len(val_data) - block_size` is `100 - 64 = 36`, and `get_batch` draws 8 starts
*with replacement* from those 36. Every evaluation overlaps itself, and the whole
val set is 100 characters of one speech.

**FINDING: `max_steps` is 500 in the file, not 2,000.** The exercise asks you to
change a number the code does not contain -- and 500 is, to within the spacing of
the checkpoints, where the validation loss bottoms. The module docstring also
advertises "4 layers, 4 heads, d_model=128, seq_len=128", where `try_train` sets
3 layers, `d_model=64` and `block_size=64`: three of the four disagree.

Structure: `start` initialises; `step` is the forward and backward pass in one
function; `train` is Adam at the lesson's own learning rate.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import re

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "14-build-a-transformer-capstone"
BLOCK, WIDTH, HEADS, LAYERS, BATCH, RATE = 64, 64, 4, 3, 8, 3e-4
HEAD_DIM, CHECKS = WIDTH // HEADS, (100, 500, 1_000, 2_000, 5_000)


def start(np, rng, vocab):
    """Token embedding, learned positions, and LAYERS causal attention blocks."""
    make = lambda a, b: rng.normal(0, math.sqrt(2 / (a + b)), (a, b))
    return {"tok": make(vocab, WIDTH), "pos": rng.normal(0, 0.02, (BLOCK, WIDTH)),
            **{f"{n}{i}": make(WIDTH, 3 * WIDTH if n == "qkv" else WIDTH)
               for i in range(LAYERS) for n in ("qkv", "o")}}


def step(np, idx, target, w, vocab, backward=True):
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
    ix = rng.integers(0, len(src) - BLOCK - 1, BATCH)
    return tuple(np.stack([src[i + s:i + s + BLOCK] for i in ix]) for s in (0, 1))


def evaluate(np, w, data, split_at, vocab, batches=20):
    """Validation loss over 20 batches, so all 36 windows are covered several times."""
    rng, seen = np.random.default_rng(999), []
    for _ in range(batches):
        idx, target = batch(np, rng, data, split_at, "val")
        seen.append(step(np, idx, target, w, vocab, backward=False)[0])
    return sum(seen) / batches


def train(np, ref, steps, seed=0, report=()):
    """Adam at the lesson's own lr; returns {step: (train loss, val loss)}."""
    data, vocab, split_at = ref.CORPUS
    rng, w = np.random.default_rng(seed), start(np, np.random.default_rng(seed), vocab)
    avg, sq, seen = *({k: np.zeros_like(v) for k, v in w.items()} for _ in range(2)), {}
    for t in range(1, steps + 1):
        idx, target = batch(np, rng, data, split_at, "train")
        loss, G = step(np, idx, target, w, vocab)
        for key, grad in G.items():
            avg[key], sq[key] = 0.9 * avg[key] + 0.1 * grad, 0.999 * sq[key] + 0.001 * grad * grad
            w[key] -= RATE * (avg[key] / (1 - 0.9 ** t)) / (
                np.sqrt(sq[key] / (1 - 0.999 ** t)) + 1e-8)
        if t in report:
            seen[t] = (loss, evaluate(np, w, data, split_at, vocab))
    return w, seen


def solve():
    import numpy as np
    ref = parity.load_reference(PHASE, LESSON, "main")
    chars = sorted(set(ref.TINY_SHAKESPEARE))
    data = np.array([chars.index(c) for c in ref.TINY_SHAKESPEARE])
    ref.CORPUS = (data, len(chars), int(0.9 * len(data)))
    w, curve = train(np, ref, max(CHECKS), report=CHECKS)
    split_at = ref.CORPUS[2]
    return {
        "curve": curve, "vocab": len(chars), "text": len(data), "split": split_at,
        "params": sum(v.size for v in w.values()), "uniform": math.log(len(chars)),
        "windows": (len(data) - split_at) - BLOCK,
        "settings": {k: int(v) for k, v in re.findall(
            r"^\s{4}(block_size|d_model|n_layers|max_steps) = (\d+)$",
            inspect.getsource(ref.try_train), re.M)},
        "torch": importlib.util.find_spec("torch") is None,
    }


def verify(result):
    curve, settings = result["curve"], result["settings"]
    best = min(curve, key=lambda s: curve[s][1])
    return [
        practice.Check(
            "ANSWER: validation never reaches 2.0, and it bottoms before step 1,000",
            min(v for _, v in curve.values()) > 2.0 and best <= 1_000,
            "step: train / val -- " + ", ".join(
                f"{s} {t:.2f}/{v:.2f}" for s, (t, v) in curve.items())
            + f". The minimum is {curve[best][1]:.2f} at step {best}, against the 2.0 asked for",
        ),
        practice.Check(
            "ANSWER: no -- at 5,000 steps it is worse than predicting uniformly",
            curve[5_000][1] > result["uniform"] > curve[best][1],
            f"val rises from {curve[best][1]:.2f} at step {best} to {curve[2_000][1]:.2f} at "
            f"2,000 and {curve[5_000][1]:.2f} at 5,000 while train falls to "
            f"{curve[5_000][0]:.2f}; ln({result['vocab']}) = {result['uniform']:.2f} is the loss "
            f"of guessing uniformly. The corpus is why: {result['text']} characters, "
            f"{result['split']} training, against {result['params']:,} parameters -- "
            f"{result['params'] / result['split']:.0f} per character",
        ),
        practice.Check(
            "FINDING: the val set is 100 characters, and max_steps is 500 rather than 2,000",
            result["windows"] == 36 and settings["max_steps"] == 500 and result["torch"],
            f"len(val_data) - block_size = {result['text'] - result['split']} - {BLOCK} = "
            f"{result['windows']} windows, drawn 16 at a time with replacement. "
            f"try_train sets {settings}: the exercise asks you to change a number the file does "
            "not contain, and 500 is where the curve bottoms. The module docstring advertises "
            "4 layers, d_model=128 and seq_len=128 against those. torch is absent",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
