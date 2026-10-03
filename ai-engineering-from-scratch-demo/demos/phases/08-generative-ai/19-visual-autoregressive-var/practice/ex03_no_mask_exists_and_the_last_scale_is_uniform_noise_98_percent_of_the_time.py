"""Exercise 3 — there is no attention to measure, and the finest scale is usually uniform noise.

    **Parallel-within-scale check.** For a trained VAR, measure the attention
    pattern explicitly. Within scale k, does the model attend to cross-scale
    positions but not intra-scale? Verify the mask implementation.

Reading of the exercise: the lesson's "transformer" is a per-scale table keyed
on `context_key`, so there are no attention weights to read. The attention
pattern is measured the only way that applies to any model: by dependence.
`generate` is run as shipped with `sample_categorical` instrumented, so every
distribution it samples a position from is recorded; and on trained token
streams each earlier-scale token is edited or swapped to see whether the
scale-k distribution moves. The model is the one `main()` trains (64 images).

**ANSWER: intra-scale, no -- and not by a mask.** Over 200 generations every
position of a scale is drawn from **one** shared distribution object (1 per
scale per sample, across 1/4/16/64 positions). Nothing within a scale can
attend to anything, including its own spatial position: in the training data
the 8x8 per-position token histograms differ from the pooled one by a mean
total-variation distance of **0.55**; the model's differ by 0.

**FINDING: cross-scale, only through one number per scale.** `context_key`
keeps `int(mean * 1000)` of each earlier scale. All **19,200** single-token
edits to the 2x2 and 4x4 scales move the key, but **0 of 3,681** swaps of two
different tokens within those scales do: the model sees what values a coarser
scale holds, never where.

**FINDING: at generation the cross-scale conditioning mostly disappears.** A
sampled context is one the table never saw in **0%, 10%, 86%, 98%** of draws at
scales 1x1..8x8, and `generate` then falls back to a uniform distribution. The
8x8 scale -- 64 of the 85 tokens -- is uniform noise in 98% of samples.

**CONTROL: the printed "mask check" is arithmetic, and it disagrees with the
doc.** `main()` prints `sum(s*s for s in SCALES[:k])` -- 0, 1, 5, 21 prior
tokens, 1,428 allowed pairs -- without consulting the model. docs/en.md says a
scale-k token attends to "all of scales 1..k", which on the same pyramid is
**5,797** pairs. Neither mask exists in the code.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "19-visual-autoregressive-var"
DRAWS = 200


def trained(ref):
    rng = np.random.default_rng(0)
    train = ref.make_patterns(rng, 64)
    books = ref.train_codebooks(train)
    streams = [ref.tokenize_multiscale(img, books) for img in train]
    return books, streams, ref.fit_predictor(streams)


def instrumented_generation(ref, books, predictors):
    """(distinct distributions per scale per sample, fallback rate per scale)."""
    seen, original = [], ref.sample_categorical
    ref.sample_categorical = lambda p, r: seen.append(p) or original(p, r)
    rng, shared, fallback = np.random.default_rng(1), set(), np.zeros(len(ref.SCALES))
    for _ in range(DRAWS):
        seen.clear()
        ref.generate(predictors, books, rng)
        start = 0
        for k, scale in enumerate(ref.SCALES):
            block = seen[start : start + scale * scale]
            shared.add(len({id(p) for p in block}))
            fallback[k] += np.allclose(block[0], 1.0 / ref.CODEBOOK)
            start += scale * scale
    ref.sample_categorical = original
    return shared, fallback / DRAWS


def variants(flat, swap):
    """Every within-scale swap of two different tokens, or every single-token edit."""
    for i in range(flat.size):
        for j in range(i + 1, flat.size) if swap else range(16):
            alt = flat.copy()
            alt[[i, j] if swap else [i]] = flat[[j, i]] if swap else j
            if not np.array_equal(alt, flat):
                yield alt


def edits(ref, streams, swap):
    """(edits tried, edits that move the context key) over the earlier scales."""
    tried = moved = 0
    for s in streams:
        for k in (1, 2):
            for alt in variants(s[k].reshape(-1), swap):
                prefix = list(s[:3])
                prefix[k] = alt.reshape(s[k].shape)
                tried += 1
                moved += ref.context_key(prefix) != ref.context_key(s[:3])
    return tried, moved


def positional_tv(ref, streams):
    tokens = np.stack([s[-1].reshape(-1) for s in streams])
    counts = [np.bincount(c, minlength=ref.CODEBOOK) for c in tokens.T]
    hists = np.stack(counts) / len(streams)
    return float((0.5 * np.abs(hists - hists.mean(0)).sum(1)).mean())


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    books, streams, predictors = trained(ref)
    shared, fallback = instrumented_generation(ref, books, predictors)
    doc = parity.doc_text(PHASE, LESSON, "en")
    sizes = [s * s for s in ref.SCALES]
    return {
        "shared": shared,
        "fallback": fallback.tolist(),
        "tv": positional_tv(ref, streams),
        "edits": edits(ref, streams, swap=False),
        "swaps": edits(ref, streams, swap=True),
        "printed": sum(n * sum(sizes[:k]) for k, n in enumerate(sizes)),
        "doc_mask": sum(n * sum(sizes[: k + 1]) for k, n in enumerate(sizes)),
        "doc_says": "attend to all of scales 1..k" in doc,
    }


def verify(result):
    fb = result["fallback"]
    return [
        practice.Check(
            "ANSWER: within a scale every position shares one distribution",
            result["shared"] == {1} and result["tv"] > 0.3,
            f"distinct distributions per scale per sample over {DRAWS} draws: "
            f"{sorted(result['shared'])}; real 8x8 per-position histograms differ from the "
            f"pooled one by mean TV {result['tv']:.2f}, the model's by 0",
        ),
        practice.Check(
            "FINDING: cross-scale conditioning sees values, never positions",
            result["edits"][1] == result["edits"][0] and result["swaps"][1] == 0,
            f"{result['edits'][1]:,}/{result['edits'][0]:,} single-token edits move the key; "
            f"{result['swaps'][1]}/{result['swaps'][0]:,} within-scale swaps do",
        ),
        practice.Check(
            "FINDING: the finest scale falls back to uniform in almost every sample",
            fb[-1] > 0.9 and fb[0] == 0.0,
            "unseen-context fallback rate by scale "
            + ", ".join(f"{v:.0%}" for v in fb),
        ),
        practice.Check(
            "CONTROL: main()'s mask check is arithmetic and disagrees with the doc",
            result["doc_says"] and result["printed"] < result["doc_mask"],
            f"main() prints scales 1..k-1, {result['printed']:,} pairs; docs/en.md says "
            f"'1..k', {result['doc_mask']:,} pairs; the code consults neither",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
