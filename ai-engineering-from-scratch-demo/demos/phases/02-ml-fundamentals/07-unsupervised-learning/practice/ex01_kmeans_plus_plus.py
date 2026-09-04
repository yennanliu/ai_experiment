"""Exercise 1 — K-Means++ initialisation against random, on convergence speed.

    Implement K-Means++ initialization: instead of picking random centroids, pick
    the first randomly and each subsequent centroid with probability proportional
    to its squared distance from the nearest existing centroid. Compare
    convergence speed to random initialization.

Reading of the exercise: the comparison only says something where random
initialisation can *fail*, so an easy layout (4 separated blobs) and a hard one
(9 tight clusters) both run — and on the easy one the advantage vanishes. Check 5
is the sharper point: k-means++ finds no better optimum, it finds one sooner.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "07-unsupervised-learning"
TRIALS, MAX_ITERS = 40, 200  # layouts below are (centres, spread, k, points per cluster)
EASY = ([(0, 0), (6, 0), (3, 6), (9, 6)], 1.2, 4, 40)
HARD = ([(i * 3.0, j * 3.0) for i in range(3) for j in range(3)], 0.8, 9, 40)


def kmeans_pp_init(ref, data, k, rng):
    """First centroid uniform; each next with P ∝ squared distance to the nearest."""
    centroids = [list(data[rng.randrange(len(data))])]
    for _ in range(k - 1):
        weights = [min(ref.euclidean_distance(p, c) ** 2 for c in centroids) for p in data]
        target, running = rng.random() * sum(weights), 0.0
        for point, weight in zip(data, weights):
            running += weight
            if running >= target:
                centroids.append(list(point))
                break
    return centroids


def random_init(ref, data, k, rng):
    return [list(data[i]) for i in rng.sample(range(len(data)), k)]


def _assign(ref, data, centroids, k):
    return [min(range(k), key=lambda j: ref.euclidean_distance(p, centroids[j]))
            for p in data]


def _recentre(data, assign, centroids, k):
    groups = [[p for p, a in zip(data, assign) if a == j] for j in range(k)]
    return [[sum(c) / len(g) for c in zip(*g)] if g else centroids[j]
            for j, g in enumerate(groups)]


def lloyd(ref, data, centroids, k):
    """Plain Lloyd iteration, so only the initialisation differs between arms."""
    for iteration in range(MAX_ITERS):
        fresh = _recentre(data, _assign(ref, data, centroids, k), centroids, k)
        moved = max(ref.euclidean_distance(a, b) for a, b in zip(fresh, centroids))
        centroids = fresh
        if moved < 1e-9:
            break
    return iteration + 1, ref.compute_inertia(data, _assign(ref, data, centroids, k),
                                              centroids)


def sweep(ref, layout):
    centres, spread, k, per = layout
    data, _ = ref.make_blobs(centres, n_per_cluster=per, spread=spread, seed=42)
    out = {}
    for name, init in (("random", random_init), ("kmeans++", kmeans_pp_init)):
        runs = [lloyd(ref, data, init(ref, data, k, random.Random(s)), k)
                for s in range(TRIALS)]
        inertias = [r[1] for r in runs]
        out[name] = {"iters": sum(r[0] for r in runs) / TRIALS,
                     "mean": sum(inertias) / TRIALS, "best": min(inertias)}
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "clustering")
    return {"easy": sweep(ref, EASY), "hard": sweep(ref, HARD),
            "trials": TRIALS, "k_hard": HARD[2]}


def verify(result):
    easy, hard = result["easy"], result["hard"]
    return [
        practice.Check(f"both initialisations run {result['trials']} seeds on two layouts",
                       len(easy) == len(hard) == 2,
                       f"mean iterations, random vs kmeans++ — easy (4 separated blobs) "
                       f"{easy['random']['iters']:.2f} vs {easy['kmeans++']['iters']:.2f}; "
                       f"hard ({result['k_hard']} tight clusters) "
                       f"{hard['random']['iters']:.2f} vs {hard['kmeans++']['iters']:.2f}"),
        practice.Check("ANSWER: k-means++ converges in fewer iterations, on both layouts",
                       hard["kmeans++"]["iters"] < hard["random"]["iters"]
                       and easy["kmeans++"]["iters"] < easy["random"]["iters"],
                       f"{100 * (1 - hard['kmeans++']['iters'] / hard['random']['iters']):.0f}% "
                       f"fewer on the hard layout: spread-out seeds start closer"),
        practice.Check("…and on the hard layout it also reaches better inertia",
                       hard["kmeans++"]["mean"] < hard["random"]["mean"],
                       f"mean inertia {hard['kmeans++']['mean']:.2f} vs "
                       f"{hard['random']['mean']:.2f} over {result['trials']} seeds"),
        practice.Check("FINDING: on the easy layout the inertia advantage disappears",
                       easy["kmeans++"]["mean"] >= easy["random"]["mean"],
                       f"{easy['kmeans++']['mean']:.2f} against {easy['random']['mean']:.2f} "
                       f"— random init rarely goes wrong on 4 separated blobs, so a better "
                       f"seeding has nothing to fix: insurance, not a free improvement"),
        practice.Check("MECHANISM: it does not find better optima, it finds them more often",
                       abs(hard["kmeans++"]["best"] - hard["random"]["best"]) < 1.0,
                       f"best over {result['trials']} seeds: {hard['kmeans++']['best']:.2f} "
                       f"vs {hard['random']['best']:.2f} — the same optimum; Lloyd is "
                       f"identical in both arms, only its start differs"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
