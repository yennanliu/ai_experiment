"""Exercise 4 — the pipeline is 5x cheaper because its MI stage throws away a true signal.

    **Feature selection pipeline**: chain variance threshold, mutual information
    filter, and RFE into a single pipeline. First remove near-zero-variance
    features, then keep the top 50% by mutual information, then run RFE on the
    survivors. Compare this pipeline against running RFE alone on all features.
    Is the pipeline faster? Is it equally accurate?

Reading of the exercise: every stage is the lesson's own function with its
`main` settings, on the lesson's 400 training rows: `variance_threshold(0.01)`
on raw features, `mutual_information` (10 bins) keeping the top half of the
survivors, then `rfe` to 5 features (lr 0.1, 200 epochs) on standardised
features. "Faster" is counted in RFE fits and feature-columns trained, not
wall-clock. "Accurate" is each 5-feature subset refitted with
`simple_logistic_importance` (300 epochs) and scored on the lesson's 100 test
rows and on 5000 fresh rows from the same generator (seed 7).

**ANSWER: faster, yes; equally accurate, no.** The pipeline's RFE runs 5 fits
over 40 feature-columns against 15 fits over 195 (4.9x less work), but its 5
features score 0.854 on the fresh rows against 0.919 for RFE alone (0.90 against
0.96 on the lesson's test rows).

**FINDING: the MI filter is what costs the accuracy -- it discards x3.** The
target is `2 x1 - 1.5 x2 + x3 + noise`, but x3 (info_2) has binned MI 0.023,
ranked 11th of 20, below the pure-noise columns noise_9 (8th) and noise_5
(10th), so the top half drops it. Only its weaker copy corr_2 gets through, and
RFE then discards that: the pipeline ends with three copies of x1 (info_0,
info_3, corr_0), two of x2 (info_1, info_4) and nothing for x3. RFE alone keeps
info_2.

**FINDING: the variance stage does nothing on this data.** The smallest variance
is 0.217 against a threshold of 0.01 -- the noise columns have variance 0.25 by
construction -- so the first stage keeps all 20 features.

**CONTROL: without the MI stage the pipeline is RFE alone.** Variance threshold
then RFE selects exactly RFE's own five features.

Structure: `pipeline` chains the three stages; `accuracy` scores a subset.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "18-feature-selection"


def rfe_on(ref, Z, y, cols):
    mask, _ = ref.rfe(Z[:, cols], y, n_features_to_select=5, lr=0.1, epochs=200)
    return [int(c) for c in np.asarray(cols)[mask]]


def pipeline(ref, X, Z, y, use_mi=True):
    """Variance threshold, then (optionally) the MI top half, then RFE to 5."""
    kept, variances = ref.variance_threshold(X, threshold=0.01)
    cols = np.where(kept)[0]
    mi = ref.mutual_information(X[:, cols], y, n_bins=10)
    if use_mi:
        cols = np.sort(cols[np.argsort(mi)[::-1][: len(cols) // 2]])
    return rfe_on(ref, Z, y, cols), cols, variances, mi


def accuracy(ref, cols, Z, y, fresh, y_fresh):
    w, b = ref.simple_logistic_importance(Z[:400, cols], y[:400], 0.1, 300)
    return (float(np.mean(((Z[400:, cols] @ w + b) >= 0) == y[400:])),
            float(np.mean(((fresh[:, cols] @ w + b) >= 0) == y_fresh)))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "feature_selection")
    X, y, names = ref.make_feature_selection_data(500, seed=42)
    mu, sd = X[:400].mean(0), X[:400].std(0)
    Z = (X - mu) / sd
    fresh, y_fresh, _ = ref.make_feature_selection_data(5000, seed=7)
    fresh = (fresh - mu) / sd
    piped, survivors, variances, mi = pipeline(ref, X[:400], Z[:400], y[:400])
    alone = rfe_on(ref, Z[:400], y[:400], np.arange(20))
    rank = {names[i]: r + 1 for r, i in enumerate(np.argsort(mi)[::-1])}
    k = len(survivors)
    return {
        "piped": [names[i] for i in piped], "alone": [names[i] for i in alone],
        "survivors": [names[i] for i in survivors], "min_var": float(variances.min()),
        "mi_x3": float(mi[2]), "rank": rank,
        "work": ((k - 5, sum(range(6, k + 1))), (15, sum(range(6, 21)))),
        "acc": {"piped": accuracy(ref, piped, Z, y, fresh, y_fresh),
                "alone": accuracy(ref, alone, Z, y, fresh, y_fresh)},
        "no_mi": [names[i] for i in pipeline(ref, X[:400], Z[:400], y[:400], use_mi=False)[0]],
    }


def verify(result):
    acc, rank, (piped_work, alone_work) = result["acc"], result["rank"], result["work"]
    return [
        practice.Check(
            "ANSWER: the pipeline is ~5x cheaper but clearly less accurate",
            alone_work[1] > 4 * piped_work[1] and acc["alone"][1] > acc["piped"][1] + 0.03,
            f"RFE work {piped_work[0]} fits / {piped_work[1]} columns against {alone_work[0]} / "
            f"{alone_work[1]}; fresh-row accuracy {acc['piped'][1]:.3f} against "
            f"{acc['alone'][1]:.3f} (lesson test rows {acc['piped'][0]:.2f} / "
            f"{acc['alone'][0]:.2f})"),
        practice.Check(
            "FINDING: the MI filter discards x3, a true signal, below two noise columns",
            "info_2" not in result["survivors"] and "info_2" in result["alone"]
            and rank["info_2"] > max(rank["noise_5"], rank["noise_9"]),
            f"info_2 (x3) has MI {result['mi_x3']:.3f}, rank {rank['info_2']} of 20, below "
            f"noise_5 ({rank['noise_5']}) and noise_9 ({rank['noise_9']}); pipeline keeps "
            f"{result['piped']}, RFE alone {result['alone']}"),
        practice.Check(
            "FINDING: the variance stage removes nothing",
            result["min_var"] > 0.1,
            f"smallest variance {result['min_var']:.3f} against a threshold of 0.01"),
        practice.Check(
            "CONTROL: without the MI stage the pipeline is RFE alone",
            result["no_mi"] == result["alone"],
            f"variance threshold then RFE selects {result['no_mi']}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
