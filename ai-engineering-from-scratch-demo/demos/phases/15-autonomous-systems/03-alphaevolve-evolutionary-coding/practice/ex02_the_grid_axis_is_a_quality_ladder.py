"""Exercise 2 — the grid axis is a quality ladder.

    Read Section 3 of the AlphaEvolve paper on the MAP-elites grid. Design a
    feature-vector descriptor for a new problem (e.g. compiler optimization
    passes) that would keep the search diverse.

Reading of the exercise: a descriptor proposed in prose cannot be compared
with the shipped one, so the design is stated for compiler passes and then
ported to this grammar, where both can key the same archive. MAP-elites asks
for axes that describe *behaviour* rather than quality, and the test used
here is where each axis puts its best row: at an end of the axis means the
axis is ranking, in the interior means it is describing.

**ANSWER: two compositional ratio axes -- pass-kind mix and target-scope
mix.** For compiler passes: the fraction of the sequence that transforms
rather than analyses, and the fraction that works on loops rather than
straight-line code, five buckets each. Ported here as operator mix and leaf
mix, its best row is **interior on both axes** -- index **1** of 5 and **3**
of 5 -- because the target expression has a particular shape, not a maximal
one.

**FINDING: the shipped depth axis is a quality ladder.** Median elite error
down its six rows is **99.6, 57.3, 21.0, 22.3, 14.2, 6.7**, and the best row
is the last -- the capped one. The axis is a proxy for the objective, so the
diversity the grid preserves is largely diversity in how bad a candidate is
allowed to be. The archive replayed here reproduces `run_loop`'s own returned
best in **10** of 10 seeds before any of this is read off it.

**FINDING: both grids saturate, so cardinality is the binding constraint.**
`cell_key` caps its axes at **30** cells and a 1500-generation run fills
**28.4** on average, all 30 in **5** of 10 seeds; the proposed descriptor's
**25** cells fill **24.8** on average. Past saturation MAP-elites is not a
diversity mechanism, it is N parallel hill climbs, and the fix for that is
more cells rather than better axes.

**FINDING: the capped axis is nearly one-way.** Of `mutate`'s **4**
branches, **2** grow depth by wrapping the parent, **1** leaves it and **1**
resets to a leaf. Depth drifts upward into the capped row, which is the one
cell where arbitrarily different programs compete as neighbours.

Structure: `archive()` replays the grid `run_loop` builds and does not
return, keyed by either descriptor; `ladder()` reads the median elite error
down one axis of it.
"""

from __future__ import annotations

import inspect
import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "03-alphaevolve-evolutionary-coding"

SEEDS, GENERATIONS, POP, BUCKETS = range(10), 1500, 20, 5
TRAIN_XS, TEST_XS = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
SHIPPED_CELLS, PROPOSED_CELLS = 30, BUCKETS * BUCKETS


def nodes(expr):
    yield expr
    if expr[0] in ("add", "mul"):
        yield from nodes(expr[1])
        yield from nodes(expr[2])
def mix_key(expr):
    """The proposed descriptor, ported: operator mix and leaf mix, five buckets each."""
    seen = list(nodes(expr))
    inner = [n for n in seen if n[0] in ("add", "mul")]
    ratios = (sum(n[0] == "mul" for n in inner) / len(inner) if inner else 0.0,
              sum(n[0] == "x" for n in seen if n[0] in ("num", "x"))
              / max(1, len(seen) - len(inner)))
    return tuple(min(int(r * BUCKETS), BUCKETS - 1) for r in ratios)


def signal(candidate):
    """`run_loop`'s acceptance signal with use_holdout=True."""
    return 0.5 * (candidate.train_score + candidate.test_score)


def archive(ref, seed, key):
    """The grid `run_loop` builds and does not return, keyed by any descriptor."""
    random.seed(seed)
    grid = {}

    def offer(candidate):
        cell = key(candidate.expr)
        if cell not in grid or signal(candidate) < signal(grid[cell]):
            grid[cell] = candidate

    for _slot in range(POP):
        offer(ref.seed_candidate(TEST_XS, TRAIN_XS, 0))
    for generation in range(1, GENERATIONS + 1):
        expr = ref.mutate(random.choice(list(grid.values())).expr)
        offer(ref.Candidate(expr, ref.mse(expr, TRAIN_XS), ref.mse(expr, TEST_XS),
                            generation))
    return grid


def ladder(ref, key, axis):
    """Median elite training error in each row of one axis, pooled over the seeds."""
    rows = {}
    for seed in SEEDS:
        for cell, best in archive(ref, seed, key).items():
            if math.isfinite(best.train_score):
                rows.setdefault(cell[axis], []).append(best.train_score)
    return [round(statistics.median(rows[r]), 1) for r in sorted(rows)]


def best_row(values):
    return values.index(min(values))


def occupancy(grids, capacity):
    counts = [len(grid) for grid in grids]
    return round(statistics.mean(counts), 1), sum(count == capacity for count in counts)


def reproduces(ref, grids):
    """Every replayed archive's best against the one `run_loop` returns."""
    return sum(min(grid.values(), key=signal).expr
               == ref.run_loop(GENERATIONS, POP, True, seed=seed)[0].expr
               for seed, grid in enumerate(grids))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped = [archive(ref, seed, ref.cell_key) for seed in SEEDS]
    proposed = [archive(ref, seed, mix_key) for seed in SEEDS]
    rows, mixes = ladder(ref, ref.cell_key, 0), [ladder(ref, mix_key, a) for a in (0, 1)]
    branches, shipped_fill = inspect.getsource(ref.mutate), occupancy(shipped, SHIPPED_CELLS)
    return {
        "shipped_ladder": rows,
        "shipped_best_row": best_row(rows),
        "proposed_best_rows": [best_row(values) for values in mixes],
        "proposed_ladders": mixes,
        "matches_run_loop": reproduces(ref, shipped),
        "shipped_capacity": SHIPPED_CELLS,
        "shipped_mean": shipped_fill[0],
        "shipped_full": shipped_fill[1],
        "proposed_capacity": PROPOSED_CELLS,
        "proposed_mean": occupancy(proposed, PROPOSED_CELLS)[0],
        "growing": branches.count('return ("'),
        "reset": branches.count("return random_leaf()"),
        "perturb": branches.count("return perturb(e)"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the proposed axes put their best row in the interior",
            all([result["proposed_best_rows"] == [1, 3],
                 0 < min(result["proposed_best_rows"]),
                 max(result["proposed_best_rows"]) < BUCKETS - 1]),
            f"operator mix and leaf mix bottom out at rows "
            f"{result['proposed_best_rows']} of {BUCKETS}, interior on both: "
            f"{result['proposed_ladders']}",
        ),
        practice.Check(
            "FINDING: the shipped depth axis is a quality ladder",
            all([result["shipped_ladder"] == [99.6, 57.3, 21.0, 22.3, 14.2, 6.7],
                 result["shipped_best_row"] == 5, result["matches_run_loop"] == 10]),
            f"median elite error down the six depth rows is "
            f"{result['shipped_ladder']}, best at row {result['shipped_best_row']}, the "
            f"capped end; the replay reproduces run_loop's best in "
            f"{result['matches_run_loop']} of 10 seeds",
        ),
        practice.Check(
            "FINDING: both grids saturate, so cardinality is the binding constraint",
            all([result["shipped_mean"] >= 27.0, result["shipped_full"] == 5,
                 result["proposed_mean"] >= 24.0]),
            f"the shipped grid fills {result['shipped_mean']} of "
            f"{result['shipped_capacity']} cells, all of them in "
            f"{result['shipped_full']} of 10 seeds; the proposed one "
            f"{result['proposed_mean']} of {result['proposed_capacity']}",
        ),
        practice.Check(
            "FINDING: the capped axis is nearly one-way",
            all([result["growing"] == 2, result["reset"] == 1, result["perturb"] == 1]),
            f"{result['growing']} of mutate's four branches grow depth, "
            f"{result['perturb']} leaves it and {result['reset']} resets to a leaf, so "
            "depth drifts into the capped row",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
