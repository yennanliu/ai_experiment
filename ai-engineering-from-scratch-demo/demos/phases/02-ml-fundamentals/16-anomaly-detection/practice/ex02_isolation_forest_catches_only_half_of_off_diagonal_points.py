"""Exercise 2 — z-score flags 0 of 25, but Isolation Forest also finds only about half.

    **Multivariate anomalies.** Create 2D data where each feature individually
    looks normal, but the combination is anomalous (e.g., points far from the
    main cluster diagonal). Show that Z-score per feature misses these but
    Isolation Forest catches them.

Reading of the exercise: 500 normals from a unit-variance Gaussian with
correlation 0.95, and 25 anomalies placed 1.2-2.0 units off the diagonal along
the anti-diagonal, with every coordinate inside the normals' own range. The
anti-diagonal sd is `sqrt(1 - 0.95) = 0.22`, so they sit 5.4-8.9 sigma off.
The lesson's `zscore_detect` and `IsolationForest` (100 trees, 256 samples)
score them. "Catches" is precision at 25 (P@25), averaged over five seeds.
Mahalanobis distance is the reference detector.

**ANSWER: the z-score misses all of them; Isolation Forest catches some.** At
threshold 3.0 the z-score flags 0 of 25 on every seed: no anomaly coordinate
exceeds |2.77|. By ranking, P@25 is 0.33 for z-score and 0.55 for Isolation
Forest.

**FINDING: "catches them" overstates it.** Mahalanobis distance ranks all 25
first (P@25 = 1.00) on every seed. Isolation Forest only reaches 0.55, even
though the anomalies are 5+ sigma out. Its splits are axis-parallel, so the two
ends of the normal diagonal are as easy to cut off as the off-diagonal points,
and they share the top of the ranking: the normals in its top 25 have a
largest coordinate of |2.71| on average.

**FINDING: it is the algorithm, not the lesson's implementation.** sklearn's
`IsolationForest` scores P@25 = 0.54 on the same data.

**CONTROL: with no correlation, Isolation Forest is fine.** On uncorrelated
normals with anomalies 3.5-4.5 units out, P@25 is 0.90, against 0.98 for
Mahalanobis.

Structure: `make_data` builds one seed; `precisions` scores every detector.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "16-anomaly-detection"
N, K, SEEDS = 500, 25, 5
NAMES = ("z", "iforest", "sklearn", "mahalanobis")


def make_data(seed, rho=0.95, off=(1.2, 2.0)):
    rng = np.random.RandomState(seed)
    normal = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], N)
    along = rng.uniform(-1.5, 1.5, K)
    across = rng.choice([-1, 1], K) * rng.uniform(*off, K) / np.sqrt(2)
    anomalies = np.column_stack([along + across, along - across])
    X = np.vstack([normal, anomalies])
    return X, np.r_[np.zeros(N), np.ones(K)].astype(int), normal


def precisions(ref, X, y, normal):
    _, z = ref.zscore_detect(X)
    iso = ref.IsolationForest(100, 256, seed=42).fit(X).anomaly_score(X)
    sk = -IsolationForest(n_estimators=100, random_state=0).fit(X).score_samples(X)
    d = X - normal.mean(0)
    maha = np.einsum("ij,jk,ik->i", d, np.linalg.inv(np.cov(normal.T)), d)
    p_at_k = [ref.precision_at_k(y, s, K) for s in (z, iso, sk, maha)]
    flagged = int(ref.zscore_detect(X, threshold=3.0)[0][y == 1].sum())
    top = np.argsort(iso)[-K:]
    tips = np.abs(X[top][y[top] == 0]).max(1).mean()
    return p_at_k + [flagged, float(np.abs(X[y == 1]).max()), tips]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "anomaly_detection")
    tilted = np.array([precisions(ref, *make_data(s)) for s in range(SEEDS)])
    round_ = np.array([precisions(ref, *make_data(s, rho=0.0, off=(3.5, 4.5)))
                       for s in range(SEEDS)])
    return {
        "p": dict(zip(NAMES, tilted[:, :4].mean(0))),
        "maha_min": float(tilted[:, 3].min()),
        "flagged": int(tilted[:, 4].max()), "max_coord": float(tilted[:, 5].max()),
        "tips": float(tilted[:, 6].mean()),
        "round": dict(zip(NAMES, round_[:, :4].mean(0))),
        "sigma_off": (1.2 / np.sqrt(0.05), 2.0 / np.sqrt(0.05)),
    }


def verify(result):
    p, rnd = result["p"], result["round"]
    return [
        practice.Check(
            "ANSWER: z-score flags none; Isolation Forest ranks more of them first",
            result["flagged"] == 0 and p["iforest"] > p["z"] + 0.1,
            f"z > 3 flags {result['flagged']} of {K} on every seed (largest anomaly coordinate "
            f"|{result['max_coord']:.2f}|); P@{K} z-score {p['z']:.2f}, Isolation Forest "
            f"{p['iforest']:.2f}",
        ),
        practice.Check(
            "FINDING: Isolation Forest finds about half of what Mahalanobis finds",
            p["iforest"] < 0.7 and result["maha_min"] == 1.0 and result["tips"] > 2.0,
            f"anomalies {result['sigma_off'][0]:.1f}-{result['sigma_off'][1]:.1f} sigma off the "
            f"diagonal; Mahalanobis P@{K} = 1.00 on every seed, Isolation Forest "
            f"{p['iforest']:.2f}; the normals in its top {K} sit at |x| {result['tips']:.2f} "
            "on average, the tips of the diagonal",
        ),
        practice.Check(
            "FINDING: sklearn's Isolation Forest does the same",
            abs(p["sklearn"] - p["iforest"]) < 0.1,
            f"sklearn P@{K} {p['sklearn']:.2f} against the lesson's {p['iforest']:.2f}",
        ),
        practice.Check(
            "CONTROL: without correlation Isolation Forest is near the reference",
            rnd["iforest"] > 0.85 and rnd["mahalanobis"] > 0.95,
            f"uncorrelated normals: Isolation Forest {rnd['iforest']:.2f}, Mahalanobis "
            f"{rnd['mahalanobis']:.2f}, z-score {rnd['z']:.2f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
