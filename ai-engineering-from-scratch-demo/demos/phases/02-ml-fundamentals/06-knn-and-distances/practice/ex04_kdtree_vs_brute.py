"""Exercise 4 — KD-tree against brute force: where does the tree stop winning?

    Implement a KD-tree and measure query time vs brute force for datasets of
    1k, 10k, and 100k points in 2D, 10D, and 50D. At what dimensionality does the
    KD-tree stop being faster than brute force?

Reading of the exercise: the sizes are the exercise's own 1k/10k/100k. Timings
are best-of-3 at 1k and 10k; at 100k a single run is used, because a 6-second
measurement is far less sensitive to machine load than a 0.5-millisecond one and
three of them would triple the tier cost for no accuracy.

ANSWER: "at what dimensionality does the KD-tree stop being faster" has no single
answer — the crossover moves with n. At d=50 the tree is slower at every size
(0.94-0.98x). At d=10 it is a wash at n=1k (1.00x) but wins 3.1x at n=100k, so
the full sweep contradicts the reading the reduced one invites; see the README.

Tier T1: the 100k-point 50D runs are ~6 seconds each.
"""

from __future__ import annotations

import random
import time

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "06-knn-and-distances"
SEED, K, N_QUERIES = 42, 5, 20
DIMENSIONS = (2, 10, 50)
SIZES = (1_000, 10_000, 100_000)


def timed(fn, repeats):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def measure(ref, dim, n):
    rng = random.Random(SEED)
    points = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(n)]
    queries = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(N_QUERIES)]
    repeats = 3 if n <= 10_000 else 1
    tree = ref.KDTree(points)
    kd = timed(lambda: [tree.query(q, k=K) for q in queries], repeats)
    brute = timed(lambda: [sorted(range(n), key=lambda i: ref.l2_distance(points[i], q))[:K]
                           for q in queries], repeats)
    return {"kd": kd, "brute": brute, "speedup": brute / kd}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "knn")
    rows = {(d, n): measure(ref, d, n) for d in DIMENSIONS for n in SIZES}
    return {"rows": rows}


def _by_size(rows, dim, template) -> str:
    return ", ".join(template(n, rows[(dim, n)]) for n in SIZES)


def verify(result):
    rows = result["rows"]
    big = SIZES[-1]
    low = [rows[(DIMENSIONS[0], n)]["speedup"] for n in SIZES]
    return [
        practice.Check(f"all {len(DIMENSIONS) * len(SIZES)} (dim, n) combinations timed",
                       len(rows) == len(DIMENSIONS) * len(SIZES),
                       "; ".join(f"d={d} n={n}: {rows[(d, n)]['speedup']:.2f}x"
                                 for d in DIMENSIONS for n in (SIZES[0], big))),
        practice.Check("at d=2 the tree wins, and wins harder as n grows",
                       low[0] < low[1] < low[2] and low[-1] > 50,
                       _by_size(rows, 2, lambda n, r: f"n={n}: {r['speedup']:.0f}x")
                       + " — the tree is O(log n) per query where the scan is O(n), so the "
                         "gap widens with n rather than staying fixed"),
        practice.Check("ANSWER: d=10 is the crossover — but only at the smallest n",
                       rows[(10, SIZES[0])]["speedup"] < 1.2
                       and rows[(10, big)]["speedup"] > 2,
                       _by_size(rows, 10, lambda n, r: f"n={n}: {r['speedup']:.2f}x")
                       + f" — a wash at n={SIZES[0]:,}, but the tree is well ahead again by "
                         f"n={big:,}. FINDING: the question presumes one crossover dimension; "
                         f"there is a crossover *surface*, and a sweep stopping at 20k points "
                         f"would have reported d=10 as the answer"),
        practice.Check("ANSWER: at d=50 the tree is SLOWER than brute force",
                       max(rows[(50, n)]["speedup"] for n in SIZES) < 1.05,
                       _by_size(rows, 50, lambda n, r: f"n={n}: {r['speedup']:.2f}x")
                       + " — it does all the scan's distance work plus the traversal "
                         "bookkeeping on top"),
        practice.Check("MECHANISM: pruning fails once no coordinate separates the points",
                       rows[(2, big)]["speedup"] / rows[(50, big)]["speedup"] > 100,
                       f"speedup falls from {rows[(2, big)]['speedup']:.0f}x to "
                       f"{rows[(50, big)]['speedup']:.2f}x at n={big:,}. A branch is pruned "
                       f"only when the splitting coordinate's gap alone exceeds the current "
                       f"best radius — and by exercise 2's measurement, in high dimensions "
                       f"every point is roughly equidistant, so almost nothing prunes and "
                       f"the tree visits nearly every leaf"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
