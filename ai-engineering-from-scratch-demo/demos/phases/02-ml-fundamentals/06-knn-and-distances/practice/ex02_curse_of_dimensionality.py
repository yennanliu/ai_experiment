"""Exercise 2 — max/min pairwise distance ratio from 2 to 500 dimensions.

    Generate 1000 random points in 2, 5, 10, 50, 100, and 500 dimensions. For
    each dimensionality, compute the ratio of the maximum pairwise distance to
    the minimum pairwise distance. Plot the ratio vs dimensionality to visualize
    the curse of dimensionality.

Reading of the exercise: the ratio is the right statistic and it collapses
exactly as advertised. But the ratio alone is fragile: it is set by the two most
extreme of the 499,500 pairs, so one unusually close pair moves it a lot. Check 4
adds the relative spread (max − min)/mean, which uses every pair, and check 5
shows what actually drives the collapse — the mean distance grows as √d while the
spread does not.

Tier T1: the exercise's own 1000 points mean half a million pure-Python distance
computations per dimensionality, ~20 seconds in total. Reducing n would speed it
up without changing the answer, but the ratio's own fragility is n-dependent —
its value is set by the extremes of the pair distribution — so the reported
numbers only mean what the exercise says at the size the exercise asks for.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "06-knn-and-distances"
SEED, N = 42, 1000
DIMENSIONS = (2, 5, 10, 50, 100, 500)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "knn")
    rows = {}
    for d in DIMENSIONS:
        rng = random.Random(SEED)
        points = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(N)]
        distances = [ref.l2_distance(points[i], points[j])
                     for i in range(N) for j in range(i + 1, N)]
        mean = sum(distances) / len(distances)
        rows[d] = {"ratio": max(distances) / min(distances), "mean": mean,
                   "spread": (max(distances) - min(distances)) / mean,
                   "n_pairs": len(distances),
                   "mean_over_sqrt_d": mean / math.sqrt(d)}
    return {"rows": rows, "n": N}


def _join(rows, template) -> str:
    return ", ".join(template(d, rows[d]) for d in DIMENSIONS)


def _falling(values) -> bool:
    return all(a > b for a, b in zip(values, values[1:]))


def verify(result):
    rows = result["rows"]
    ratios = [rows[d]["ratio"] for d in DIMENSIONS]
    spreads = [rows[d]["spread"] for d in DIMENSIONS]
    scaled = [rows[d]["mean_over_sqrt_d"] for d in DIMENSIONS]
    return [
        practice.Check(f"{result['n']} points, {rows[2]['n_pairs']:,} pairs per dimension",
                       min(rows[d]["n_pairs"] for d in DIMENSIONS)
                       == result["n"] * (result["n"] - 1) // 2,
                       f"every pair measured at each of {len(DIMENSIONS)} dimensionalities"),
        practice.Check("ANSWER: the max/min ratio collapses toward 1",
                       _falling(ratios) and ratios[-1] < 1.5,
                       _join(rows, lambda d, r: f"d={d}: {r['ratio']:.2f}")
                       + f" — at d=500 the furthest pair is only "
                         f"{100 * (ratios[-1] - 1):.0f}% further apart than the closest, so "
                         f"'nearest' stops meaning much. The d=2 value is n-dependent (it was "
                         f"170 at n=120): more points push the closest pair closer without "
                         f"moving the furthest, which is the fragility check 4 answers"),
        practice.Check("…which is what breaks KNN: no neighbour is distinctively near",
                       ratios[0] / ratios[-1] > 100,
                       f"the ratio falls {ratios[0] / ratios[-1]:.0f}x from d=2 to d=500. "
                       f"A nearest-neighbour rule needs some points to be closer than "
                       f"others by a margin that survives noise"),
        practice.Check("the relative spread confirms it, using every pair not just two",
                       _falling(spreads),
                       "(max − min)/mean: "
                       + _join(rows, lambda d, r: f"d={d}: {r['spread']:.3f}")
                       + " — the max/min ratio depends on the two most extreme pairs, so "
                         "this is the more robust statement of the same collapse"),
        practice.Check("MECHANISM: the mean distance grows as √d while the spread does not",
                       max(scaled) / min(scaled) < 1.3,
                       "mean distance / √d: "
                       + _join(rows, lambda d, r: f"d={d}: {r['mean_over_sqrt_d']:.3f}")
                       + f" — nearly constant, so mean distance ≈ √(2d) for unit Gaussians. "
                         f"Distances all grow together, the differences between them do not, "
                         f"and the ratio is what is left"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
