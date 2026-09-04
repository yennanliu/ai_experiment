"""Exercise 1 — robust scaling (median/IQR) against standardisation, with outliers.

    Add robust scaling (using median and interquartile range instead of mean and
    standard deviation) to the numerical transforms. Compare it to standard
    scaling on data with extreme outliers.

Reading of the exercise: "compare it on data with extreme outliers" only says
something if the comparison is of what happens to the *inliers*. Both transforms
are affine, so neither hides an outlier; what differs is how much resolution the
95 ordinary points keep. Checks 3 and 4 measure that, and check 6 runs the same
data with the outliers removed, where the two transforms are the same map up to
the Gaussian constant IQR/sigma.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "08-feature-engineering"
SEED, N_INLIERS = 42, 95
OUTLIERS = (500.0, 600.0, -400.0, 700.0, 550.0)
IQR_OVER_SIGMA = 1.3490  # for a Gaussian: Phi^-1(0.75) - Phi^-1(0.25)


def quantile(values, q: float) -> float:
    """Linear interpolation between order statistics, as numpy's default does."""
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    if low >= len(ordered) - 1:
        return ordered[-1]
    return ordered[low] + (position - low) * (ordered[low + 1] - ordered[low])


def robust_scale(values):
    """(v - median) / IQR — the transform the exercise asks to be added."""
    median = quantile(values, 0.5)
    spread = quantile(values, 0.75) - quantile(values, 0.25)
    return [(v - median) / (spread or 1.0) for v in values]


def _shape(scaled, n_inliers: int) -> dict:
    inliers = scaled[:n_inliers]
    mean = sum(inliers) / n_inliers
    return {"sd": (sum((v - mean) ** 2 for v in inliers) / n_inliers) ** 0.5,
            "iqr": quantile(inliers, 0.75) - quantile(inliers, 0.25),
            "span": max(inliers) - min(inliers),
            "within_one": sum(abs(v) <= 1 for v in inliers) / n_inliers,
            "worst_outlier": max(abs(v) for v in scaled[n_inliers:] or [0.0])}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "features")
    rng = random.Random(SEED)
    inliers = [rng.gauss(10, 1) for _ in range(N_INLIERS)]
    contaminated = inliers + list(OUTLIERS)
    clean_standard, clean_robust = ref.standardize(inliers), robust_scale(inliers)
    return {
        "standard": _shape(ref.standardize(contaminated), N_INLIERS),
        "robust": _shape(robust_scale(contaminated), N_INLIERS),
        "clean_agreement": ref.correlation(clean_standard, clean_robust),
        "clean_ratio": (_shape(clean_standard, N_INLIERS)["sd"]
                        / _shape(clean_robust, N_INLIERS)["sd"]),
        "n_inliers": N_INLIERS, "n_outliers": len(OUTLIERS),
    }


def verify(result):
    std, rob = result["standard"], result["robust"]
    return [
        practice.Check(f"{result['n_inliers']} inliers plus {result['n_outliers']} "
                       f"extreme outliers, scaled both ways", std["sd"] > 0 and rob["sd"] > 0,
                       f"outliers at {OUTLIERS} against inliers drawn from N(10, 1)"),
        practice.Check("both transforms are affine, so neither conceals an outlier",
                       std["worst_outlier"] > 3 and rob["worst_outlier"] > 3,
                       f"largest |scaled outlier|: {std['worst_outlier']:.2f} standardised, "
                       f"{rob['worst_outlier']:.2f} robust — visible either way, so 'it "
                       f"handles outliers better' has to mean something about the inliers"),
        practice.Check("ANSWER: standardisation crushes the inliers; robust scaling does not",
                       rob["iqr"] > 50 * std["iqr"],
                       f"inlier IQR after scaling: {std['iqr']:.4f} standardised against "
                       f"{rob['iqr']:.4f} robust, a factor of {rob['iqr'] / std['iqr']:.0f}. "
                       f"Five values in a hundred set the standard deviation, and every "
                       f"ordinary point is then divided by it"),
        practice.Check("…so standardised inliers become mutually indistinguishable",
                       std["within_one"] == 1.0 and std["span"] < 0.1,
                       f"all {100 * std['within_one']:.0f}% of standardised inliers fall "
                       f"within ±1, spanning {std['span']:.4f} in total, against "
                       f"{100 * rob['within_one']:.0f}% and a span of {rob['span']:.4f} under "
                       f"robust scaling. The z-score's usual reading — ±1 is typical, ±3 is "
                       f"far — is gone: here ±1 covers everything"),
        practice.Check("MECHANISM: the median and IQR do not move; the mean and sd do",
                       rob["iqr"] > 0.9,
                       f"robust scaling puts the inlier IQR at {rob['iqr']:.4f}, near the 1.0 "
                       f"it would be with no outliers at all — a quantile ignores how far "
                       f"away the tails are, a moment does not"),
        practice.Check("CONTROL: with the outliers removed the two agree exactly",
                       result["clean_agreement"] > 1 - 1e-9
                       and abs(result["clean_ratio"] / IQR_OVER_SIGMA - 1) < 0.06,
                       f"correlation {result['clean_agreement']:.6f} — the same affine map — "
                       f"differing only in scale by {result['clean_ratio']:.4f}, against the "
                       f"Gaussian constant IQR/σ = {IQR_OVER_SIGMA}. Robust scaling costs "
                       f"nothing when there is nothing to be robust to"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
