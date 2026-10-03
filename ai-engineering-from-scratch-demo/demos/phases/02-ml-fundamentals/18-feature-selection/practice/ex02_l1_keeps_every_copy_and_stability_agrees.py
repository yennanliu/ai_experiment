"""Exercise 2 — the lesson's L1 keeps every copy of a signal, and stability selection agrees.

    **Stability selection**: run L1 feature selection 50 times, each time on a
    random 80% subsample of the data, with slightly different alpha values.
    Count how often each feature is selected. Features selected in > 80% of runs
    are "stable." Compare stable features against single-run L1 selection. Which
    is more reliable?

Reading of the exercise: L1 selection is the lesson's `l1_feature_selection`
with its `main` settings (alpha 0.05, lr 0.01, 1000 epochs) on the lesson's 400
standardised training rows. Each of the 50 runs takes 320 of those rows without
replacement and draws alpha uniformly from 0.04 to 0.06; a feature is stable if
selected in more than 40 runs.

**ANSWER: the stable set and the single run are the same 8 features** --
info_0-4 and corr_0-2 -- with corr_2 at 98% and the rest at 100%. Stability
selection is more *informative* rather than more selective here: it is the only
one that shows noise_9 entering 38% of runs, a borderline the single run hides
by reporting it as a clean zero. Every other noise feature, and corr_3 and
corr_4, are selected in 0 of 50 runs.

**FINDING: L1 does not "pick one and zero the others" as the lesson says.** The
doc credits L1 with handling correlated features by keeping one of them. The
single run keeps all three copies of x1 -- info_0, info_3, corr_0, pairwise r
>= 0.989 -- with near-equal weights 0.554, 0.553 and 0.534. Run 50 times longer
(50,000 epochs) it still keeps all three (0.846, 0.712, 0.277).

**FINDING: stability selection cannot see redundancy either.** All three x1
copies are selected in 50 of 50 runs: stability measures whether a selection
repeats, and a redundant selection repeats perfectly.

Structure: `frequencies` runs the 50 subsampled fits; `solve` adds the single
and the long run.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "18-feature-selection"
RUNS, ROWS, ALPHA = 50, 320, (0.04, 0.06)
X1_COPIES = ("info_0", "info_3", "corr_0")


def training_rows(ref):
    X, y, names = ref.make_feature_selection_data(500, seed=42)
    Z = X[:400]
    return (Z - Z.mean(0)) / Z.std(0), y[:400], names


def frequencies(ref, Z, y):
    rng, counts = np.random.RandomState(0), np.zeros(Z.shape[1])
    for _ in range(RUNS):
        rows = rng.choice(len(y), ROWS, replace=False)
        mask, _ = ref.l1_feature_selection(Z[rows], y[rows], alpha=rng.uniform(*ALPHA),
                                           lr=0.01, epochs=1000)
        counts += mask
    return counts / RUNS


def x1_copies(names, values):
    """`values` at the three copies of x1, in X1_COPIES order."""
    return [float(values[names.index(n)]) for n in X1_COPIES]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "feature_selection")
    Z, y, names = training_rows(ref)
    freq = frequencies(ref, Z, y)
    single, w = ref.l1_feature_selection(Z, y, alpha=0.05, lr=0.01, epochs=1000)
    _, w_long = ref.l1_feature_selection(Z, y, alpha=0.05, lr=0.01, epochs=50000)
    r = np.corrcoef(Z.T)
    rows = [names.index(n) for n in X1_COPIES]
    return {
        "freq": dict(zip(names, freq)),
        "quiet": [n for n, f in zip(names, freq) if f == 0],
        "stable": [n for n, f in zip(names, freq) if f > 0.8],
        "single": [n for n, keep in zip(names, single) if keep],
        "copies_freq": x1_copies(names, freq),
        "weights": x1_copies(names, w), "long": x1_copies(names, w_long),
        "min_r": float(r[np.ix_(rows, rows)].min()),
        "doc_says": "picks one and zeros the others" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    freq, quiet = result["freq"], result["quiet"]
    fmt = lambda ws: ", ".join(f"{v:.3f}" for v in ws)  # noqa: E731
    return [
        practice.Check(
            "ANSWER: the stable set equals the single run; stability alone exposes noise_9",
            result["stable"] == result["single"] and len(result["stable"]) == 8
            and 0.2 < freq["noise_9"] < 0.8,
            f"stable and single-run: {result['stable']}; corr_2 at {freq['corr_2']:.0%}; "
            f"noise_9 at {freq['noise_9']:.0%}; {len(quiet)} features at 0%"),
        practice.Check(
            "FINDING: L1 keeps every copy of x1, not one as the doc says",
            result["doc_says"] and min(result["weights"]) > 0.4 and min(result["long"]) > 0.1,
            f"doc: L1 'picks one and zeros the others'; {X1_COPIES} (pairwise r >= "
            f"{result['min_r']:.3f}) get weights {fmt(result['weights'])} after 1000 epochs and "
            f"{fmt(result['long'])} after 50,000"),
        practice.Check(
            "FINDING: stability selection is blind to redundancy too",
            min(result["copies_freq"]) == 1.0,
            "all three x1 copies are selected in "
            + ", ".join(f"{f:.0%}" for f in result["copies_freq"]) + " of runs"),
        practice.Check(
            "CONTROL: nine noise features and both mixtures are never selected",
            quiet == ["corr_3", "corr_4"] + [f"noise_{i}" for i in range(9)],
            f"selected in 0 of {RUNS} runs: {quiet}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
