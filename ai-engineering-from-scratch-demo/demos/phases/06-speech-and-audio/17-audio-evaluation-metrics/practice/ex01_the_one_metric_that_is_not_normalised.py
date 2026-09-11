"""Exercise 1 — the one metric that is not normalised is CER.

    **Easy.** Run `code/main.py`. Compute WER / CER / EER / SECS / FAD-ish /
    MMAU-ish on toy inputs.

Reading of the exercise: all six numbers come out, so the exercise is read as
"compute them and check each against the definition it is named after". Replaying
`main()`'s random stream exactly reproduces every printed figure -- **EER 0.10%
at threshold 0.661, SECS 0.644, FAD-like 1.748** -- and four of the six diverge
from the metric they claim to be.

**`cer` never normalises, and the file's own takeaway says it must.** `wer` runs
both strings through `normalize`; `cer` does not, and divides by the raw
reference length. On the doc's own Step 1 example -- `"Please turn on the
lights."` against `"please turn on the light"` -- that is **0.1154 raw against
0.0400 normalised, 2.9x**, because the capital `P` and the trailing full stop are
both scored as character errors and the stop is also in the denominator.

**That same example is quoted with the wrong answer.** Step 1 says jiwer returns
"~0.17"; one substitution over a five-word reference is **0.2000**, and the
lesson's own `wer` returns exactly that. No normalisation reaches 0.17.

**`embedding_fad_like` is the square root of a *diagonal* FAD.** It keeps only
per-dimension variance, so every off-diagonal covariance is discarded. Two sets
matched to **4.6e-17** in per-dimension mean and **1.3e-15** in per-dimension
variance, differing only in correlation (mean |off-diagonal| **0.036 against
0.896**), score **0.000000** under it and **6.07** under the real definition. On
`main()`'s own data it prints **1.748** where the real FAD is **15.608** -- and
the line beside it compares that to a published **4.5**.

**SECS is pinned by construction below the target printed next to it.** The clone
is `ref + noise` at equal variance, so its expected cosine is exactly
`1/sqrt(2) = 0.7071`, under the "> 0.75 for recognizable clone" the same line
states. Reaching 0.75 needs the noise no larger than **0.8819x** the signal;
raising the dimension only tightens the estimate around 0.7071.

Structure: `replay` re-runs `main()`'s random draws in order so every printed
figure is reproducible; `real_fad` is the published Frechet formula; `decorrelated`
builds the moment-matched pair that separates the two definitions.
"""

from __future__ import annotations

import math
import random

import numpy as np
from scipy.linalg import sqrtm

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "17-audio-evaluation-metrics"
DOC_REF, DOC_HYP, DOC_CLAIM = "Please turn on the lights.", "please turn on the light", 0.17
SECS_TARGET, DIM, ROWS = 0.75, 8, 400
PUBLISHED_FAD = 4.5
EER_TABLE = {"ECAPA": 0.87, "3D-Speaker": 0.50, "ASVspoof 5": 7.23}


def block(rng, rows, dim, mean, sigma):
    """A rows x dim embedding block, drawn in `main()`'s own order."""
    return [[rng.gauss(mean, sigma) for _ in range(dim)] for _ in range(rows)]


def replay():
    """`main()`'s random stream, in order, so every printed figure is reproducible."""
    random.seed(0)
    rng = random.Random(0)
    same = [rng.gauss(0.80, 0.06) for _ in range(100)]
    diff = [rng.gauss(0.20, 0.15) for _ in range(500)]
    reference = [rng.gauss(0, 0.1) for _ in range(192)]
    clone = [value + rng.gauss(0, 0.1) for value in reference]
    return (same, diff, reference, clone,
            block(rng, 50, 32, 0, 1.0), block(rng, 50, 32, 0.1, 1.1))


def real_fad(first, second):
    """The published Frechet distance: squared, and with the full covariance."""
    left, right = np.cov(first, rowvar=False), np.cov(second, rowvar=False)
    root = sqrtm(left @ right)
    root = root.real if np.iscomplexobj(root) else root
    gap = np.mean(first, 0) - np.mean(second, 0)
    return float(gap @ gap + np.trace(left + right - 2 * root))


def standardise(block):
    return (block - block.mean(0)) / block.std(0)


def decorrelated(dim=DIM, rows=ROWS):
    """Two blocks matched in every per-dimension moment, differing only in correlation."""
    gen = np.random.default_rng(0)
    chol = np.linalg.cholesky(np.full((dim, dim), 0.9) + np.eye(dim) * 0.1)
    return standardise(gen.standard_normal((rows, dim))), standardise(
        gen.standard_normal((rows, dim)) @ chol.T)


def off_diagonal(block):
    dim = block.shape[1]
    return float(np.abs(np.corrcoef(block, rowvar=False)[np.triu_indices(dim, 1)]).mean())


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    same, diff, reference, clone, real, fake = replay()
    plain, mixed = decorrelated()
    rate, threshold = ref.eer_from_scores(same, diff)
    step = 1 / (2 * len(diff))
    return {
        "eer": rate, "threshold": threshold, "secs": ref.cosine(reference, clone),
        "fad_like": ref.embedding_fad_like(real, fake), "fad_real": real_fad(real, fake),
        "doc_wer": ref.wer(DOC_REF, DOC_HYP),
        "cer_raw": ref.cer(DOC_REF, DOC_HYP),
        "cer_norm": ref.cer(ref.normalize(DOC_REF), ref.normalize(DOC_HYP)),
        "flat_like": ref.embedding_fad_like(plain.tolist(), mixed.tolist()),
        "flat_real": real_fad(plain, mixed),
        "mean_gap": float(np.abs(plain.mean(0) - mixed.mean(0)).max()),
        "var_gap": float(np.abs(plain.var(0) - mixed.var(0)).max()),
        "corr": (off_diagonal(plain), off_diagonal(mixed)),
        "expected_secs": 1 / math.sqrt(2), "ratio": math.sqrt(1 / SECS_TARGET**2 - 1),
        "off_grid": [n for n, v in EER_TABLE.items()
                     if abs(v / (step * 100) - round(v / (step * 100))) > 1e-9],
        "step": step * 100,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the replay reproduces every figure `main()` prints",
            abs(result["eer"] - 0.001) < 1e-12 and round(result["secs"], 3) == 0.644,
            f"EER {result['eer'] * 100:.2f}% at threshold {result['threshold']:.3f}, SECS "
            f"{result['secs']:.3f}, FAD-like {result['fad_like']:.3f} -- the same numbers the "
            "file prints, so every divergence below is a divergence from the definition and "
            "not from the demo",
        ),
        practice.Check(
            "FINDING: `cer` never normalises, and the file's own takeaway says it must",
            result["cer_raw"] > 2.5 * result["cer_norm"],
            f"`wer` runs both strings through `normalize` and `cer` does not, so the doc's own "
            f"Step 1 example scores {result['cer_raw']:.4f} raw against "
            f"{result['cer_norm']:.4f} normalised -- {result['cer_raw'] / result['cer_norm']:.1f}x "
            "-- because the capital P and the trailing stop are both character errors and the "
            "stop is in the denominator too",
        ),
        practice.Check(
            "FINDING: Step 1 quotes its own example with the wrong answer",
            abs(result["doc_wer"] - 0.2) < 1e-12,
            f"it says jiwer returns ~{DOC_CLAIM}; one substitution over a five-word reference "
            f"is {result['doc_wer']:.4f}, which is what the lesson's own `wer` returns. There "
            "is no normalisation that makes it 0.17",
        ),
        practice.Check(
            "FINDING: `embedding_fad_like` is the square root of a diagonal FAD",
            result["flat_like"] < 1e-9 < result["flat_real"],
            f"two blocks matched to {result['mean_gap']:.1e} in per-dimension mean and "
            f"{result['var_gap']:.1e} in per-dimension variance, differing only in correlation "
            f"({result['corr'][0]:.3f} against {result['corr'][1]:.3f} mean |off-diagonal|), "
            f"score {result['flat_like']:.6f} under it and {result['flat_real']:.2f} under the "
            f"real formula. On `main()`'s data it is {result['fad_like']:.3f} where the real "
            f"FAD is {result['fad_real']:.3f}, beside a published {PUBLISHED_FAD}",
        ),
        practice.Check(
            "FINDING: SECS is pinned below the target printed next to it",
            result["secs"] < SECS_TARGET and abs(result["expected_secs"] - 0.7071) < 1e-4,
            f"the clone is `ref + noise` at equal variance, so its expected cosine is "
            f"1/sqrt(2) = {result['expected_secs']:.4f}, under the '> {SECS_TARGET} for "
            f"recognizable clone' the same line states. Reaching {SECS_TARGET} needs the noise "
            f"at most {result['ratio']:.4f}x the signal; more dimensions only tighten the "
            f"estimate around {result['expected_secs']:.4f}",
        ),
        practice.Check(
            "CONTROL: 100 same and 500 different pairs put EER on a 0.10 pp grid",
            len(result["off_grid"]) == 2,
            f"false rejects move in 1.00 pp steps and false accepts in 0.20 pp, so the "
            f"reachable EERs are the multiples of {result['step']:.2f} pp. Of the three EER "
            f"figures in the file's own benchmark table, {result['off_grid']} fall between them",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
