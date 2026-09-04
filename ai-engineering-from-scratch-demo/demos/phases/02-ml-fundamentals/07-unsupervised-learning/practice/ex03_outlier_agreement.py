"""Exercise 3 — DBSCAN and GMM outliers: overlap, and where they disagree.

    Build a simple anomaly detection pipeline: run DBSCAN and GMM on the same
    data, flag points that both methods agree are outliers (noise in DBSCAN, low
    probability in GMM). Measure the overlap and discuss when the methods
    disagree.

Reading of the exercise: "low probability in GMM" does not say *which*
probability, and the obvious reading is wrong. Taking max responsibility as the
score inverts the answer — planted outliers score **0.931** against inliers'
0.889, because a point far from every component is still assigned unambiguously
to its nearest one. Responsibility measures *which* component, not *whether* any.

Scored by distance to the nearest component mean instead, GMM works. Checks 2-3
show both, so the failure is exhibited rather than sidestepped.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "07-unsupervised-learning"
SEED, N, NOISE, EPS, MIN_SAMPLES, K = 42, 200, 0.08, 0.2, 5, 3
N_PLANTED, FLAG_FRACTION = 25, 0.10


def _dbscan_noise(ref, points):
    labels = ref.dbscan(points, eps=EPS, min_samples=MIN_SAMPLES)
    labels = labels[0] if isinstance(labels, tuple) else labels
    return {i for i, v in enumerate(labels) if v == -1}


def _gmm_scores(ref, points):
    """Two candidate outlier scores: max responsibility, and distance to a mean."""
    with parity.quiet():
        _, means, _, responsibilities = ref.gmm(points, K, seed=SEED)
    return ([max(r) for r in responsibilities],
            [min(ref.euclidean_distance(p, m) for m in means) for p in points])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "clustering")
    moons, _ = ref.make_moons(n_samples=N, noise=NOISE, seed=SEED)
    rng = random.Random(1)
    planted = [[rng.uniform(-2.5, 3.5), rng.uniform(-2.0, 2.5)] for _ in range(N_PLANTED)]
    points = list(moons) + planted
    truth = set(range(len(moons), len(points)))

    dbscan_flags = _dbscan_noise(ref, points)
    max_resp, distances = _gmm_scores(ref, points)
    budget = int(FLAG_FRACTION * len(points))
    by_resp = set(sorted(range(len(points)), key=lambda i: max_resp[i])[:budget])
    by_distance = set(sorted(range(len(points)), key=lambda i: -distances[i])[:budget])

    def score(flags):
        return {"caught": len(flags & truth), "false": len(flags - truth)}

    return {
        "n": len(points), "n_planted": N_PLANTED, "budget": budget,
        "dbscan": score(dbscan_flags), "by_resp": score(by_resp),
        "by_distance": score(by_distance),
        "agree": score(dbscan_flags & by_distance),
        "either": len(dbscan_flags | by_distance),
        "dbscan_only": len(dbscan_flags - by_distance),
        "gmm_only": len(by_distance - dbscan_flags),
        "planted_resp": sum(max_resp[i] for i in truth) / len(truth),
        "inlier_resp": sum(max_resp[i] for i in range(len(moons))) / len(moons),
    }


def verify(result):
    db, resp, dist, agree = (result["dbscan"], result["by_resp"],
                             result["by_distance"], result["agree"])
    return [
        practice.Check(f"DBSCAN flags {db['caught']} of {result['n_planted']} planted "
                       f"outliers with {db['false']} false positives",
                       db["caught"] > 15 and db["false"] == 0,
                       f"eps={EPS}, min_samples={MIN_SAMPLES} over {result['n']} points — "
                       f"density is the natural criterion when the clusters are curved"),
        practice.Check("FINDING: 'low probability' read as low responsibility inverts it",
                       result["planted_resp"] > result["inlier_resp"],
                       f"planted outliers average {result['planted_resp']:.4f} max "
                       f"responsibility against inliers' {result['inlier_resp']:.4f} — "
                       f"*higher*. A point far from every component is still assigned "
                       f"unambiguously to its nearest, so responsibility says which "
                       f"component, not whether any. It catches {resp['caught']} of "
                       f"{result['n_planted']} with {resp['false']} false positives"),
        practice.Check("…scored by distance to the nearest mean, GMM works",
                       dist["caught"] > 5 * resp["caught"],
                       f"{dist['caught']} of {result['n_planted']} caught with "
                       f"{dist['false']} false positives at the same "
                       f"{result['budget']}-point budget"),
        practice.Check(f"ANSWER: the two agree on {agree['caught']} planted outliers",
                       agree["caught"] >= 15 and agree["false"] <= 3,
                       f"intersection catches {agree['caught']} with {agree['false']} false "
                       f"positives — better precision than either alone, which is the point "
                       f"of requiring agreement"),
        practice.Check("ANSWER: they disagree where density and distance disagree",
                       result["dbscan_only"] > 0 or result["gmm_only"] > 0,
                       f"{result['dbscan_only']} flagged only by DBSCAN, "
                       f"{result['gmm_only']} only by GMM, {result['either']} by either. "
                       f"DBSCAN calls a point normal if it has neighbours, wherever it is; "
                       f"GMM calls it normal if it is near a fitted centre. A sparse arm of "
                       f"the moon is fine to DBSCAN and far to GMM"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
