"""Exercise 3 — a dataset where L1, L2, cosine and Mahalanobis all disagree.

    Implement a function that takes a dataset and a query point and returns the
    nearest neighbor under L1, L2, cosine, and Mahalanobis distance. Find a
    dataset where all four disagree on which point is nearest.

Reading of the exercise: "find a dataset where all four disagree" is a search, so
the solution searches rather than hand-tunes — a seeded random sweep over
candidate datasets, stopping at the first that separates all four. Reporting the
number of trials matters: it tells you how rare the case is, and the answer
(rare, but a few hundred tries) is more informative than a single hand-built
example that might have been reverse-engineered from the metrics.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "14-norms-and-distances"
SEED, MAX_TRIALS, N_POINTS, DIM = 20260904, 20_000, 6, 3


def nearest_under_all(ref, query, dataset, covariance):
    """The winning index under each of the four metrics."""
    metrics = {
        "L1": lambda p: ref.l1_distance(query, p),
        "L2": lambda p: ref.l2_distance(query, p),
        "cosine": lambda p: ref.cosine_distance(query, p),
        "mahalanobis": lambda p: ref.mahalanobis_distance(query, p, covariance),
    }
    return {name: min(range(len(dataset)), key=lambda i: fn(dataset[i]))
            for name, fn in metrics.items()}


def _attempt(ref, rng):
    """One candidate dataset; None if the metrics cannot all be evaluated."""
    dataset = [[rng.uniform(-8, 8) for _ in range(DIM)] for _ in range(N_POINTS)]
    query = [rng.uniform(-2, 2) for _ in range(DIM)]
    try:
        winners = nearest_under_all(ref, query, dataset, ref.compute_covariance(dataset))
    except Exception:
        return None
    return (query, dataset, winners) if len(set(winners.values())) == 4 else None


def solve():
    ref = parity.load_reference(PHASE, LESSON, "distances")
    rng = random.Random(SEED)
    for trial in range(1, MAX_TRIALS + 1):
        found = _attempt(ref, rng)
        if found is not None:
            query, dataset, winners = found
            return {"trials": trial, "winners": winners, "query": query,
                    "dataset": dataset, "found": True}
    return {"found": False, "trials": MAX_TRIALS}


def verify(result):
    if not result["found"]:
        return [practice.Check("a disagreeing dataset was found", False,
                               f"none in {result['trials']} trials")]
    winners = result["winners"]
    return [
        practice.Check("all four metrics pick a different nearest neighbour",
                       len(set(winners.values())) == 4,
                       ", ".join(f"{k} → point {v}" for k, v in winners.items())),
        practice.Check(f"found by search, not by hand: trial {result['trials']}",
                       result["trials"] > 1,
                       f"{N_POINTS} points in {DIM}-D, seeded sweep — total disagreement is "
                       f"uncommon but not exotic, and the search reports how uncommon"),
        practice.Check("L1 and L2 disagree, which needs the axes to trade off",
                       winners["L1"] != winners["L2"],
                       f"L1 → {winners['L1']}, L2 → {winners['L2']}: L1 sums the "
                       f"coordinates, L2 punishes the largest one, so a point that is "
                       f"slightly off on every axis beats one badly off on a single axis "
                       f"under L1 and loses under L2"),
        practice.Check("cosine disagrees with both, because it ignores magnitude",
                       winners["cosine"] not in (winners["L1"], winners["L2"]),
                       f"cosine → {winners['cosine']}: it can pick a point far away in "
                       f"absolute terms as long as the direction from the origin matches"),
        practice.Check("Mahalanobis disagrees with all three, because it rescales the space",
                       winners["mahalanobis"] not in
                       (winners["L1"], winners["L2"], winners["cosine"]),
                       f"mahalanobis → {winners['mahalanobis']}: dividing by the data's own "
                       f"covariance makes a step along a high-variance direction cheap and "
                       f"a step along a low-variance one expensive, so 'near' depends on "
                       f"the whole dataset, not just the two points"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
