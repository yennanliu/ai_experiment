"""Exercise 5 — conditioning buys a few percent, because the bottleneck is position, not prompt.

    **Text conditioning.** Extend VAR to take a text embedding (CLIP pooled) as
    an extra conditioning input via adaLN. This is the HART recipe. How much
    does FID improve on text-aligned sampling?

Reading of the exercise: CLIP and a transformer with LayerNorms are not
available, so the "text" is the pattern's class (solid, gradient, ring,
checker, cross), recovered from each image by matching the library's four fixed
templates. adaLN gives each condition its own scale-and-shift parameters; the
histogram analogue is a separate set of tables per condition, so the lesson's
own `fit_predictor` is fitted once per class and the lesson's own `generate`
samples from the prompted class's tables. FID is a pixel-space Frechet
distance against held-out images of the same class (500 val images); "text
alignment" is the share of samples whose nearest held-out image has that class.

**ANSWER: about 10%.** Over the four structured classes the class-matched
distance drops from **17.7** unconditioned to **15.9** conditioned (per class
13.7/12.6/24.9/19.7 to 11.8/11.3/23.3/17.2). Text alignment barely moves: at
most **5%** of conditioned samples look like the class they were prompted with
(from 0-2% unconditioned).

**FINDING: even a perfect prompt-to-token map would not help.** Taking each
class's *true* training tokens and only shuffling positions within each scale
-- what a position-blind predictor that never errs would produce -- scores
**6.4 to 31.9**, against **0.002 to 0.04** for the unshuffled tokens through
the same decoder. Exercise 3 showed the predictor shares one distribution per
scale; that, not missing conditioning, is the ceiling.

**FINDING: with 11-15 identical examples the 1x1 token is still a coin flip.**
Each structured class is one fixed image, yet `fit_predictor`'s add-one prior
over 16 codes leaves its correct scale-1 token at probability **0.44-0.52**,
exactly (n+1)/(n+16). Half the conditioned samples are off-class before the
first scale is done, and every later scale then falls back to uniform.

**CONTROL: the prompt does reach the model, at the 1x1 scale.** The share of
samples carrying the class's true 1x1 token rises from **16-22%** unconditioned
to **45-54%** conditioned. The conditioning is wired through; it has no lever
on spatial structure.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "19-visual-autoregressive-var"
SAMPLES, CLASSES = 300, 5


def frechet(a, b):
    a, b = (np.asarray(x, np.float64).reshape(len(x), -1) for x in (a, b))
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root = np.sqrt(np.clip(np.linalg.eigvals(s1 @ s2).real, 0.0, None)).sum()
    return float(((a.mean(0) - b.mean(0)) ** 2).sum() + np.trace(s1 + s2) - 2 * root)


def label(images, templates):
    f = images.reshape(len(images), -1)  # class 0 = solid, 1..4 = fixed templates
    dist = [f.var(1)] + [((f - t) ** 2).mean(1) for t in templates]
    return np.stack(dist, 1).argmin(1)


def nearest(samples, val, val_labels):
    f, v = samples.reshape(len(samples), -1), val.reshape(len(val), -1)
    return val_labels[((f[:, None, :] - v[None]) ** 2).sum(-1).argmin(1)]


def sample(ref, predictors, books, rng, first):
    drawn = [ref.generate(predictors, books, rng) for _ in range(SAMPLES)]
    hit = float(np.mean([d[1][0].item() == first for d in drawn]))
    return np.stack([d[0] for d in drawn]), hit


def replay(ref, own, books, rng, shuffle):
    out = []
    for i in range(SAMPLES):
        stream = own[i % len(own)]
        if shuffle:
            stream = [rng.permutation(t.reshape(-1)).reshape(t.shape) for t in stream]
        out.append(ref.detokenize_multiscale(stream, books))
    return np.stack(out)


def one_class(ref, c, ctx, rng):
    books, streams, labels, unc, val, val_labels = ctx
    own = [s for s, lab in zip(streams, labels) if lab == c]
    cond, target, n = ref.fit_predictor(own), val[val_labels == c], len(own)
    out = {"count": n, "top": float(cond[0][()].max())}
    out["prior_exact"] = abs(out["top"] - (n + 1) / (n + 16)) < 1e-9
    for name, table in (("unc", unc), ("cond", cond)):
        images, out[f"{name}_first"] = sample(ref, table, books, rng, own[0][0].item())
        out[name] = frechet(images, target)
        out[f"{name}_align"] = float((nearest(images, val, val_labels) == c).mean())
    out["shuffled"] = frechet(replay(ref, own, books, rng, True), target)
    out["exact"] = frechet(replay(ref, own, books, rng, False), target)
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    flat = ref.make_patterns(np.random.default_rng(7), 200).reshape(200, -1)
    templates = np.unique(flat[np.ptp(flat, axis=1) > 0], axis=0)
    train = ref.make_patterns(np.random.default_rng(0), 64)
    val = ref.make_patterns(np.random.default_rng(99), 500)
    books = ref.train_codebooks(train)
    streams = [ref.tokenize_multiscale(img, books) for img in train]
    labels = (label(train, templates), label(val, templates))
    ctx = (books, streams, labels[0], ref.fit_predictor(streams), val, labels[1])
    return {
        c: one_class(ref, c, ctx, np.random.default_rng(2 + c)) for c in range(CLASSES)
    }


def pairs(a, b, fmt):
    return ", ".join(f"{x:{fmt}}->{y:{fmt}}" for x, y in zip(a, b))


def verify(result):
    s = {k: [result[c][k] for c in range(1, CLASSES)] for k in result[1]}
    unc, cond = np.mean(s["unc"]), np.mean(s["cond"])
    return [
        practice.Check(
            "ANSWER: conditioning improves class-matched distance by about a tenth",
            0.8 < cond / unc < 1.0,
            f"structured classes {unc:.1f} -> {cond:.1f} ({1 - cond / unc:.0%}); per class "
            + pairs(s["unc"], s["cond"], ".1f"),
        ),
        practice.Check(
            "ANSWER: text alignment does not move",
            max(s["cond_align"]) <= 0.1,
            "own-class share, unconditioned -> conditioned: "
            + pairs(s["unc_align"], s["cond_align"], ".0%"),
        ),
        practice.Check(
            "FINDING: perfect tokens in a position-blind order stay far from the data",
            min(s["shuffled"]) > 50 * max(s["exact"]),
            f"shuffled true tokens {min(s['shuffled']):.1f}-{max(s['shuffled']):.1f}; "
            f"unshuffled {min(s['exact']):.3f}-{max(s['exact']):.3f}",
        ),
        practice.Check(
            "FINDING: the add-one prior makes the 1x1 token a coin flip",
            all(s["prior_exact"]),
            f"top scale-1 probability {[round(v, 2) for v in s['top']]} at "
            f"n={s['count']}, each exactly (n+1)/(n+16)",
        ),
        practice.Check(
            "CONTROL: the prompt does reach the model -- at the 1x1 scale",
            bool(np.all(np.array(s["cond_first"]) > 2 * np.array(s["unc_first"]))),
            "share with the true 1x1 token, unconditioned -> conditioned: "
            + pairs(s["unc_first"], s["cond_first"], ".0%"),
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
