"""Exercise 2 — PLDA has four dimensions to learn from, and twenty-five to fill.

    **Medium.** Use SpeechBrain ECAPA on 30 VoxCeleb1 utterances (5 speakers x 6
    each). Compute EER with cosine vs PLDA.

Reading of the exercise: `speechbrain`, `torch`, `torchaudio` and `pyannote` are
all absent and VoxCeleb1 is not here, so the ECAPA half cannot run. The half that
decides the exercise survives the substitution exactly, because it is about the
sample size the exercise itself names: **PLDA cannot be estimated from 30
utterances of 5 speakers.**

PLDA is a two-covariance model. Its within-speaker scatter is built from `N - S`
independent residuals and its between-speaker scatter from `S - 1`, so 30
utterances of 5 speakers give ranks of at most **25** and **4** whatever the
embedding is. Scoring needs the within-class matrix inverted.

For SpeechBrain ECAPA that embedding is **192-dimensional**: a rank-25 matrix in
192 dimensions, 13% of the space, and a speaker subspace of 4 directions. The
demonstration substitutes the lesson's own 26-dimensional `embed_mfcc_stats`,
where the arithmetic is the same and the failure is visible:

| 5 speakers x | N | rank(within) / dim | cond(within) | cosine EER | PLDA EER |
|---|---:|---:|---:|---:|---:|
| **6 (the exercise)** | 30 | **25 / 26** | **6.0e+16** | **1.22%** | **69.67%** |
| 20 | 100 | 26 / 26 | 2.2e+03 | 0.96% | 2.12% |

At the exercise's own sample size the matrix is singular one dimension short.
`numpy.linalg.inv` returns something anyway -- it does not raise -- and the scores
it produces are **worse than chance**, 69.67% against a coin's 50%. Four times the
data makes the matrix invertible and the arm sane.

It still loses to cosine, and that is the second half of the answer: rank(between)
is `S - 1` = **4** at every sample size, because it is bounded by the number of
*speakers* and not the number of utterances. PLDA's entire model is a speaker
subspace, and five speakers give it four directions to find. Collecting more
utterances cannot fix that; collecting more speakers is the only thing that can.

Structure: `enroll` builds the corpus at a chosen utterances-per-speaker;
`scatters` returns the two scatter matrices; `plda_scorer` is the two-covariance
score; `arm` scores one trial list and returns its EER.
"""

from __future__ import annotations

import importlib.util
import random

import numpy as np

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "06-speaker-recognition-verification"
SR, SECONDS, NOISE = 8000, 0.4, 0.35
SPEAKERS = {"alice": [200, 400, 600], "bob": [220, 330, 880], "carol": [300, 600, 1200],
            "dave": [180, 540, 1080], "eve": [260, 520, 780]}
SIZES = (6, 20)
ABSENT = ("speechbrain", "torch", "torchaudio", "pyannote")
ECAPA_DIM = 192


def enroll(ref, per_speaker, seed=7):
    random.seed(seed)
    return {name: [ref.embed_mfcc_stats(ref.tone_mix(freqs, SR, SECONDS, noise=NOISE), SR)
                   for _ in range(per_speaker)] for name, freqs in SPEAKERS.items()}


def scatters(enrolled):
    """Within- and between-speaker scatter, the two matrices a PLDA is made of."""
    rows = np.array([v for vectors in enrolled.values() for v in vectors])
    centre = rows.mean(0)
    within = np.zeros((rows.shape[1],) * 2)
    between = np.zeros_like(within)
    for vectors in enrolled.values():
        block = np.array(vectors)
        mean = block.mean(0)
        within += (block - mean).T @ (block - mean)
        between += len(block) * np.outer(mean - centre, mean - centre)
    return rows, centre, within, between


def plda_scorer(centre, within, count, speakers):
    """Two-covariance scoring: the inner product under the inverted within-class matrix."""
    precision = np.linalg.inv(within / (count - speakers))
    return lambda a, b: float((np.array(a) - centre) @ precision @ (np.array(b) - centre))


def arm(ref, enrolled, score):
    """EER of one scorer over the full same/different trial list."""
    names, same, different = list(enrolled), [], []
    for name in names:
        vectors = enrolled[name]
        same += [score(vectors[i], vectors[j])
                 for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            different += [score(a, b) for a in enrolled[first] for b in enrolled[second]]
    return ref.eer(same, different)[0]


def measure(ref, per_speaker):
    enrolled = enroll(ref, per_speaker)
    rows, centre, within, between = scatters(enrolled)
    return {
        "n": len(rows), "dim": rows.shape[1],
        "rank_within": int(np.linalg.matrix_rank(within)),
        "rank_between": int(np.linalg.matrix_rank(between)),
        "condition": float(np.linalg.cond(within)),
        "cosine": arm(ref, enrolled, ref.cosine),
        "plda": arm(ref, enrolled, plda_scorer(centre, within, len(rows), len(enrolled))),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "runs": {size: measure(ref, size) for size in SIZES},
        "speakers": len(SPEAKERS), "ecapa": ECAPA_DIM,
    }


def verify(result):
    small, large = (result["runs"][size] for size in SIZES)
    return [
        practice.Check(
            "CONTROL: the ECAPA half cannot run, and the arithmetic does not need it",
            len(result["absent"]) == len(ABSENT),
            f"find_spec is None for {result['absent']} and VoxCeleb1 is not here, so the "
            f"lesson's own {small['dim']}-dimensional `embed_mfcc_stats` stands in. The ranks "
            "below are set by N and S alone, so they carry over to ECAPA unchanged",
        ),
        practice.Check(
            "ANSWER: at 5 speakers x 6 the within-class scatter is singular",
            small["rank_within"] == small["n"] - result["speakers"] < small["dim"],
            f"a two-covariance model gets N - S = {small['rank_within']} independent residuals, "
            f"one short of the {small['dim']} dimensions it must invert; the condition number is "
            f"{small['condition']:.1e}. For ECAPA's {result['ecapa']}-dimensional embedding it "
            f"would be rank {small['rank_within']} in {result['ecapa']} dimensions, 13% of the "
            "space",
        ),
        practice.Check(
            "ANSWER: the PLDA arm then scores worse than a coin, and nothing raises",
            small["plda"] > 0.5 > small["cosine"],
            f"`numpy.linalg.inv` returns a matrix for a singular input rather than raising, and "
            f"the scores it produces give EER {small['plda'] * 100:.2f}% against cosine's "
            f"{small['cosine'] * 100:.2f}% -- worse than the 50% a coin would score. The arm "
            "fails silently, which is the only way a rank deficiency can fail",
        ),
        practice.Check(
            "MECHANISM: four times the data makes the matrix invertible and the arm sane",
            large["rank_within"] == large["dim"] and large["plda"] < 0.1,
            f"at {large['n']} utterances the scatter reaches full rank "
            f"{large['rank_within']}/{large['dim']}, the condition number falls to "
            f"{large['condition']:.1e}, and PLDA scores {large['plda'] * 100:.2f}% against "
            f"cosine's {large['cosine'] * 100:.2f}%",
        ),
        practice.Check(
            "FINDING: the speaker subspace has four directions at any sample size",
            small["rank_between"] == large["rank_between"] == result["speakers"] - 1,
            f"rank(between) is bounded by S - 1 = {small['rank_between']} whether there are "
            f"{small['n']} utterances or {large['n']}, because it counts *speakers* and not "
            "utterances. PLDA's whole model is a speaker subspace, so more recordings cannot "
            "help it -- only more speakers can, and the exercise fixes that number at five",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
