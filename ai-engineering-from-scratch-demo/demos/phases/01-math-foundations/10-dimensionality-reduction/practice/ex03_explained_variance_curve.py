"""Exercise 3 — does the explained-variance curve find the 5 informative dims?

    Take a dataset with 50 features where only 5 are informative (generate one
    with `sklearn.datasets.make_classification`). Apply PCA and check whether the
    explained variance curve correctly identifies that the data is effectively
    5-dimensional.

Reading of the exercise: "check whether" is a real question and the answer is
**no**. `make_classification(n_informative=5)` does not produce data whose
*variance* is 5-dimensional — the 45 remaining features are independent noise of
comparable scale, so the covariance spectrum is nearly flat and the elbow is not
at 5. The exercise is worth doing precisely because it fails: informative and
high-variance are different properties, and PCA only ever sees the second.
Check 5 builds the dataset where the curve *does* work, to show the difference is
the data and not the method.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "10-dimensionality-reduction"
N_FEATURES, N_INFORMATIVE, N_SAMPLES, SEED = 50, 5, 800, 42


def elbow_at(ratios, threshold=0.90):
    """Smallest k whose cumulative explained variance reaches `threshold`."""
    running = 0.0
    for k, value in enumerate(ratios, 1):
        running += value
        if running >= threshold:
            return k
    return len(ratios)


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy and scikit-learn — uv sync --extra math")
    try:
        from sklearn.datasets import make_classification
    except ImportError:
        raise practice.Skip("needs scikit-learn — uv sync --extra math") from None
    ref = parity.load_reference(PHASE, LESSON, "dim_reduction")
    X, _ = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES,
                               n_informative=N_INFORMATIVE, n_redundant=0,
                               n_repeated=0, shuffle=False, random_state=SEED)
    model = ref.PCA(n_components=N_FEATURES).fit(X)
    ratios = [float(v) for v in model.explained_variance_ratio_]
    # a genuinely low-rank dataset: 5 latent factors projected into 50 dims
    rng = numpy.random.default_rng(SEED)
    latent = rng.normal(size=(N_SAMPLES, N_INFORMATIVE))
    mixing = rng.normal(size=(N_INFORMATIVE, N_FEATURES))
    low_rank = latent @ mixing + rng.normal(scale=0.01, size=(N_SAMPLES, N_FEATURES))
    lr_ratios = [float(v) for v in
                 ref.PCA(n_components=N_FEATURES).fit(low_rank).explained_variance_ratio_]
    return {"ratios": ratios, "elbow": elbow_at(ratios),
            "top5": sum(ratios[:N_INFORMATIVE]),
            "flat_ratio": ratios[0] / ratios[N_FEATURES // 2],
            "low_rank_elbow": elbow_at(lr_ratios), "low_rank_top5": sum(lr_ratios[:5]),
            "low_rank_drop": lr_ratios[4] / lr_ratios[5]}


def verify(result):
    ratios = result["ratios"]
    return [
        practice.Check(f"PCA fitted on {N_SAMPLES}x{N_FEATURES} with "
                       f"{N_INFORMATIVE} informative features",
                       len(ratios) == N_FEATURES and abs(sum(ratios) - 1.0) < 1e-9,
                       f"explained variance sums to {sum(ratios):.9f}"),
        practice.Check("ANSWER: no — the curve does not identify 5 dimensions",
                       result["elbow"] != N_INFORMATIVE,
                       f"90% of variance needs {result['elbow']} components, not "
                       f"{N_INFORMATIVE}; the top {N_INFORMATIVE} carry only "
                       f"{result['top5']:.1%}"),
        practice.Check("…because the spectrum has no cliff in it",
                       result["flat_ratio"] < result["low_rank_drop"] / 1000,
                       f"largest component is {result['flat_ratio']:.2f}x the median one, "
                       f"against a {result['low_rank_drop']:.0f}x drop at the true rank in "
                       f"check 5's dataset — five orders of magnitude apart. First five "
                       f"ratios {[round(v, 4) for v in ratios[:5]]}: a gentle slope, not "
                       f"an elbow"),
        practice.Check("informative ≠ high-variance, which is all PCA can see",
                       result["top5"] < 0.35,
                       f"make_classification's 45 non-informative features are independent "
                       f"noise at comparable scale, so they occupy real variance. PCA is "
                       f"unsupervised — it never sees the labels the 5 features are "
                       f"informative *about*"),
        practice.Check("the same curve nails a genuinely rank-5 dataset",
                       result["low_rank_elbow"] == N_INFORMATIVE
                       and result["low_rank_top5"] > 0.999,
                       f"5 latent factors projected into 50 dims: elbow at "
                       f"{result['low_rank_elbow']}, top 5 carry "
                       f"{result['low_rank_top5']:.4%}, and component 5 is "
                       f"{result['low_rank_drop']:.0f}x component 6 — so the failure above "
                       f"is a property of the data, not of the method"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
