"""Exercise 4 — there is nothing for DiT to be beaten by: the toy VAR's samples match uniform noise.

    **VAR vs DiT scaling.** For the same ImageNet class-conditional task, train
    VAR and DiT at matched param budgets (e.g., 33M, 130M, 458M). Plot FID vs
    compute. VAR should pull ahead of DiT at each size — reproduce the paper's
    result at small scale.

Reading of the exercise: torch, ImageNet and an Inception network are all
unavailable, so the comparison is scaled down (DESIGN D11) to the half that
can be run honestly -- the lesson's own VAR, fitted at growing "param budgets"
-- with a pixel-space Frechet distance standing in for FID. The toy's only
capacity knob is how many contexts its tables memorise, so the budget axis is
the training-set size (16 to 4096 images) and parameters are counted as table
entries times codebook size. Samples come from the lesson's own `generate`.

**ANSWER: the precondition fails, so the paper's result cannot be reproduced
here.** Across a 20x growth in parameters (304 to 6,080) the VAR's Frechet
distance to 500 held-out images is **14.8, 10.5, 9.5, 12.1, 11.1** -- no trend,
and never clearly below **11.4**, the distance of uniform random pixels. A
log-log fit of distance on parameters has slope **-0.05**: no power law, where
the doc says "doubling parameters or compute reliably halves error".

**FINDING: compute does not move at all.** Every budget runs the same 4 passes
and 85 tokens per sample. The toy has no compute axis to plot FID against; it
has one fixed point.

**CONTROL: the failure is the predictor, not the tokenizer or the metric.**
Replaying the training images' own token streams through the lesson's decoder
-- a predictor that is perfect on its training set -- scores **0.06 to 0.21**,
and two independent draws of real images score **0.03**. The tokenizer keeps
the images and the metric can tell; it is the position-blind, mostly-fallback
next-scale table (exercise 3) that turns them into noise.

What a DiT would face is left unrun on purpose. The doc's own counts put DiT
at 28-50 passes against VAR's ~10, a 3-5x gap in passes before quality enters;
the quality half of the claim needs a VAR whose samples beat noise first.
"""

from __future__ import annotations

import importlib.util

import numpy as np

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "19-visual-autoregressive-var"
BUDGETS, SAMPLES = (16, 64, 256, 1024, 4096), 500


def frechet(a, b):
    """Frechet distance between Gaussian fits of two image sets, in pixel space."""
    a, b = (
        a.reshape(len(a), -1).astype(np.float64),
        b.reshape(len(b), -1).astype(np.float64),
    )
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root = np.sqrt(np.clip(np.linalg.eigvals(s1 @ s2).real, 0.0, None)).sum()
    return float(
        ((a.mean(0) - b.mean(0)) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * root
    )


def at_budget(ref, images, val):
    train = ref.make_patterns(np.random.default_rng(0), images)
    books = ref.train_codebooks(train)
    streams = [ref.tokenize_multiscale(img, books) for img in train]
    predictors = ref.fit_predictor(streams)
    rng = np.random.default_rng(1)
    drawn = [ref.generate(predictors, books, rng) for _ in range(SAMPLES)]
    replay = [
        ref.detokenize_multiscale(streams[i % images], books) for i in range(SAMPLES)
    ]
    return {
        "params": sum(len(t) for t in predictors) * ref.CODEBOOK,
        "var": frechet(np.stack([d[0] for d in drawn]), val),
        "oracle": frechet(np.stack(replay), val),
        "tokens": {sum(t.size for t in d[1]) for d in drawn},
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    val = ref.make_patterns(np.random.default_rng(99), SAMPLES)
    rows = {n: at_budget(ref, n, val) for n in BUDGETS}
    params = np.log([rows[n]["params"] for n in BUDGETS])
    slope = np.polyfit(params, np.log([rows[n]["var"] for n in BUDGETS]), 1)[0]
    noise = np.random.default_rng(2).uniform(0.0, 1.0, (SAMPLES, ref.IMG, ref.IMG))
    return {
        "rows": rows,
        "slope": float(slope),
        "noise": frechet(noise, val),
        "real": frechet(ref.make_patterns(np.random.default_rng(98), SAMPLES), val),
        "missing": [
            m for m in ("torch", "torchvision") if importlib.util.find_spec(m) is None
        ],
        "passes": len(ref.SCALES),
    }


def verify(result):
    rows = result["rows"]
    var = [rows[n]["var"] for n in BUDGETS]
    oracle = [rows[n]["oracle"] for n in BUDGETS]
    return [
        practice.Check(
            "ANSWER: no budget brings the toy VAR clearly closer to the data than noise",
            min(var) > 0.75 * result["noise"],
            "Frechet distance by params "
            + ", ".join(f"{rows[n]['params']}:{rows[n]['var']:.1f}" for n in BUDGETS)
            + f"; uniform noise {result['noise']:.1f}; missing {result['missing']}",
        ),
        practice.Check(
            "ANSWER: there is no power law to fit",
            abs(result["slope"]) < 0.2,
            f"log-log slope of distance on params {result['slope']:.2f} (halving per "
            "doubling would be -1)",
        ),
        practice.Check(
            "FINDING: compute is the same at every budget",
            all(rows[n]["tokens"] == {85} for n in BUDGETS) and result["passes"] == 4,
            f"{result['passes']} passes and 85 tokens per sample at every budget",
        ),
        practice.Check(
            "CONTROL: the tokenizer and the metric work; the predictor does not",
            max(oracle) < 0.5 and result["real"] < 0.1 and min(var) > 20 * max(oracle),
            f"replayed training tokens {min(oracle):.2f}-{max(oracle):.2f}; real vs real "
            f"{result['real']:.2f}; sampled VAR {min(var):.1f}-{max(var):.1f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
