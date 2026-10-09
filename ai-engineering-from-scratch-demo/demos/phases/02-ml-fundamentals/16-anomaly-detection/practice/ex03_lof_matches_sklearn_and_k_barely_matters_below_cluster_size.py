"""Exercise 3 — LOF matches sklearn to 2e-9; k barely matters until it reaches the cluster size.

    **LOF from scratch.** Implement Local Outlier Factor using k-nearest
    neighbors. Compare against sklearn's LocalOutlierFactor on the same data.
    Use k=10 and k=50 -- how does the choice of k affect results?

Reading of the exercise: "the same data" is the lesson's own
`make_multimodal_data` (three clusters of 200 with different densities, plus
20 uniform anomalies), the case the lesson built for local methods. LOF is
written from its definition: k-distance, reachability distance, local
reachability density, then the ratio to the neighbours'. Results are ranked
with the lesson's `precision_at_k` at k = 20 anomalies, plus average precision.
k = 200 is added because it equals the cluster size.

**ANSWER: from k=10 to k=50 the ranking hardly moves.** P@20 is 0.85 at both,
and average precision is 0.840 against 0.875. At k=200, the size of one
cluster, neighbourhoods start to span clusters: the number of anomalies with
LOF above 1.5 falls from 17 to 8.

**FINDING: 0.85 is the ceiling, not the detector's miss.** The generator draws
anomalies uniformly over the whole box, and 3 of the 20 land inside a cluster,
within 2.35 Mahalanobis units of its centre. Their k=10 ranks are 106, 143 and
353. At k=50 the other 17 hold ranks 0-16 exactly, so k=50 ranks every
detectable anomaly first.

**FINDING: the score scale moves with k even when the ranking does not.**
Normals above a fixed LOF of 1.5 number 18 at k=10, 26 at k=50 and 0 at
k=200. A fixed LOF threshold is a different detector at every k.

**CONTROL: the from-scratch LOF is sklearn's.** The largest difference over all
620 points is 1.5e-9 at k=10 and 1.2e-9 at k=50. sklearn adds 1e-10 to the
mean reachability distance, and that accounts for the gap.

Structure: `lof` is the from-scratch score; `buried` measures how deep each
anomaly sits inside its nearest cluster.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.neighbors import LocalOutlierFactor

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "16-anomaly-detection"
KS = (10, 50, 200)
CLUSTERS = [([0, 0], [[0.3, 0], [0, 0.3]]), ([5, 5], [[0.5, 0.1], [0.1, 0.5]]),
            ([-3, 4], [[0.4, -0.1], [-0.1, 0.4]])]  # the generator's own parameters


def lof(X, k):
    """LOF_k(p) = mean lrd of p's k neighbours / lrd(p)."""
    dist = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    rows = np.arange(len(X))[:, None]
    neigh = np.argsort(dist, axis=1)[:, :k]
    k_dist = dist[rows, neigh][:, -1]
    reach = np.maximum(dist[rows, neigh], k_dist[neigh])
    lrd = 1.0 / reach.mean(1)
    return lrd[neigh].mean(1) / lrd


def buried(points):
    """Mahalanobis distance from each point to its nearest cluster centre."""
    dists = []
    for centre, cov in CLUSTERS:
        d = points - np.array(centre)
        dists.append(np.sqrt(np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d)))
    return np.min(dists, axis=0)


def per_k(ref, X, y, k):
    mine = lof(X, k)
    theirs = -LocalOutlierFactor(n_neighbors=k).fit(X).negative_outlier_factor_
    ranks = np.argsort(np.argsort(-mine))[y == 1]
    return {"gap": float(np.abs(mine - theirs).max()),
            "p_at": ref.precision_at_k(y, mine, int(y.sum())),
            "ap": average_precision_score(y, mine), "ranks": sorted(ranks.tolist()),
            "hits": int((mine[y == 1] > 1.5).sum()), "false": int((mine[y == 0] > 1.5).sum())}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "anomaly_detection")
    X, y = ref.make_multimodal_data()
    return {"k": {k: per_k(ref, X, y, k) for k in KS}, "depth": sorted(buried(X[y == 1]))}


def verify(result):
    r10, r50, r200 = (result["k"][k] for k in KS)
    inside = [d for d in result["depth"] if d < 3]
    return [
        practice.Check(
            "ANSWER: k=10 and k=50 rank alike; k=200 (the cluster size) does not",
            r10["p_at"] == r50["p_at"] and abs(r10["ap"] - r50["ap"]) < 0.05
            and r200["hits"] < r10["hits"] / 2,
            f"P@20 {r10['p_at']:.2f} and {r50['p_at']:.2f}, AP {r10['ap']:.3f} and "
            f"{r50['ap']:.3f}; anomalies above LOF 1.5: {r10['hits']}, {r50['hits']}, "
            f"{r200['hits']} at k=10, 50, 200",
        ),
        practice.Check(
            "FINDING: three anomalies are inside clusters, so 0.85 is the ceiling",
            len(inside) == 3 and r50["ranks"][:17] == list(range(17)),
            f"{len(inside)} of 20 lie within {max(inside):.2f} Mahalanobis units of a centre; "
            f"their k=10 ranks are {r10['ranks'][17:]}; at k=50 the other 17 hold ranks 0-16",
        ),
        practice.Check(
            "FINDING: the LOF scale moves with k, so a fixed threshold does too",
            len({r10["false"], r50["false"], r200["false"]}) == 3,
            f"normals above 1.5: {r10['false']}, {r50['false']}, {r200['false']} at k=10, 50, 200",
        ),
        practice.Check(
            "CONTROL: the from-scratch LOF equals sklearn's",
            all(r["gap"] < 1e-8 for r in result["k"].values()),
            "largest difference " + ", ".join(f"k={k}: {r['gap']:.1e}"
                                               for k, r in result["k"].items()),
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
