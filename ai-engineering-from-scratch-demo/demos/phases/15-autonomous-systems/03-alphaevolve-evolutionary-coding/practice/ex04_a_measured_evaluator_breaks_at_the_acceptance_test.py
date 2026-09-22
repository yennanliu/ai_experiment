"""Exercise 4 — a measured evaluator breaks at the acceptance test.

    Propose one domain where AlphaEvolve would fail. Identify exactly where
    the evaluator breaks and why.

Reading of the exercise: "exactly where" means a line, not a paragraph, so
the domain is chosen to make one line fail and the failure is then run. The
domain is kernel and query latency optimisation benchmarked on shared
hardware -- a domain AlphaEvolve is actually pointed at, where the evaluator
is a *measurement* rather than a proof, and repeating it returns a different
number.

**ANSWER: it breaks at `signal_of(child) < signal_of(incumbent)`.** Replaying
the archive with the score perturbed by a sigma of 1.0 -- a benchmark on a
noisy machine -- leaves **5** of **29.3** cells held by a program that is not
the best one that ever entered them, median regret **0.33** and worst
**3.46**. At sigma 0 the same replay displaces **0**. Nothing in the loop is
wrong about arithmetic; the comparison simply treats one measurement as the
truth.

**FINDING: the loop never re-measures.** `run_loop` calls `mse` **2** times
per child, both on the child, and **0** times on an incumbent. A score is
taken once and frozen into the `Candidate`, so a lucky draw is not corrected
by the thousand generations that follow -- it is defended by them, because the
cell it holds is the cell its challengers must beat.

**FINDING: the data model cannot express a measurement.** `Candidate` has
**4** fields, **2** of which are scores, and **0** of which is a sample count,
a variance or a confidence interval. There is no way to write "0.42 +/- 0.15
over 5 runs" into this archive, so there is also no way for the acceptance
test to demand a margin before it swaps.

**FINDING: the final selection inherits the same frozen numbers.** `best =
min(archive.values(), key=signal_of)` re-reads what was stored, so at sigma
1.0 the program the run returns is worse than the best in its own archive in
**2** of **10** seeds, by as much as **1.00**, where an exact evaluator is
never wrong. The reported winner is partly a ranking of who got lucky.

Structure: `replay()` rebuilds `run_loop`'s archive with a measured rather
than exact evaluator, tracking each cell's true best alongside its occupant.
"""

from __future__ import annotations

import inspect
import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "03-alphaevolve-evolutionary-coding"

SEEDS, GENERATIONS, POP = range(10), 1500, 20
TRAIN_XS, TEST_XS = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
SIGMA = 1.0                      # one benchmark's worth of machine noise


def replay(ref, seed, sigma):
    """`run_loop`'s archive, scored by measurement: (occupant, observed, true) per cell."""
    random.seed(seed)
    noise, grid, floor = random.Random(90000 + seed), {}, {}

    def offer(expr):
        true = 0.5 * (ref.mse(expr, TRAIN_XS) + ref.mse(expr, TEST_XS))
        observed = true if sigma == 0 or not math.isfinite(true) else \
            max(0.0, true + noise.gauss(0, sigma))
        key = ref.cell_key(expr)
        floor[key] = min(floor.get(key, true), true)
        if key not in grid or observed < grid[key][1]:
            grid[key] = (expr, observed, true)

    for _slot in range(POP):
        offer(ref.random_leaf())
    for _generation in range(GENERATIONS):
        offer(ref.mutate(random.choice([held[0] for held in grid.values()])))
    return grid, floor


def displaced(grid, floor):
    """Cells whose occupant is not the best candidate that ever entered them."""
    return [grid[key][2] - floor[key] for key in grid
            if math.isfinite(grid[key][2]) and grid[key][2] > floor[key] + 1e-9]


def selection_regret(grid):
    """How much worse the returned best is than the best truly in the archive."""
    picked = min(grid.values(), key=lambda held: held[1])
    finite = [held[2] for held in grid.values() if math.isfinite(held[2])]
    return picked[2] - min(finite)


def spread(values):
    if not values:
        return [0.0, 0.0]
    return [round(statistics.median(values), 2), round(max(values), 2)]


def mean(values):
    return round(statistics.mean(values), 1)


def pooled(per_seed):
    return [value for row in per_seed for value in row]


def losses_of(built):
    return [selection_regret(grid) for grid, _floor in built]


def sweep(ref, sigma):
    built = [replay(ref, seed, sigma) for seed in SEEDS]
    per_seed = [displaced(grid, floor) for grid, floor in built]
    regrets, losses = spread(pooled(per_seed)), losses_of(built)
    return {
        "cells": mean([len(grid) for grid, _floor in built]),
        "displaced": mean([len(row) for row in per_seed]),
        "median_regret": regrets[0],
        "worst_regret": regrets[1],
        "selection_losses": len([v for v in losses if v > 1e-9]),
        "selection_worst": round(max(losses), 2),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    loop = inspect.getsource(ref.run_loop)
    fields = list(ref.Candidate.__dataclass_fields__)
    return {
        "exact": sweep(ref, 0.0),
        "measured": sweep(ref, SIGMA),
        "sigma": SIGMA,
        "mse_calls": loop.count("mse("),
        "incumbent_rescores": loop.count("mse(incumbent"),
        "fields": fields,
        "score_fields": [name for name in fields if name.endswith("_score")],
        "spread_fields": [name for name in fields
                          if any(word in name for word in ("var", "std", "samples", "ci"))],
        "final_selection": "min(archive.values(), key=signal_of)" in loop,
    }


def verify(result):
    exact, measured = result["exact"], result["measured"]
    return [
        practice.Check(
            "ANSWER: it breaks at the acceptance test, and measurement is enough",
            all([exact["displaced"] == 0.0, measured["displaced"] == 5.0,
                 measured["median_regret"] == 0.33, measured["worst_regret"] == 3.46]),
            f"at sigma {result['sigma']} {measured['displaced']} of "
            f"{measured['cells']} cells hold a program that is not the best that "
            f"entered them -- median regret {measured['median_regret']}, worst "
            f"{measured['worst_regret']} -- against {exact['displaced']} at sigma 0",
        ),
        practice.Check(
            "FINDING: the loop never re-measures",
            all([result["mse_calls"] == 2, result["incumbent_rescores"] == 0]),
            f"run_loop calls mse {result['mse_calls']} times, both on the child, and "
            f"{result['incumbent_rescores']} times on an incumbent -- a score is frozen "
            "into the Candidate at first sight",
        ),
        practice.Check(
            "FINDING: the data model cannot express a measurement",
            all([len(result["fields"]) == 4, len(result["score_fields"]) == 2,
                 result["spread_fields"] == []]),
            f"Candidate's {len(result['fields'])} fields {result['fields']} hold "
            f"{len(result['score_fields'])} scores and "
            f"{len(result['spread_fields'])} variances, so the acceptance test cannot "
            "demand a margin",
        ),
        practice.Check(
            "FINDING: the final selection inherits the frozen numbers",
            all([result["final_selection"], measured["selection_losses"] == 2,
                 measured["selection_worst"] == 1.0,
                 exact["selection_losses"] == 0]),
            f"the returned best is worse than the best truly in its own archive in "
            f"{measured['selection_losses']} of {len(SEEDS)} seeds, by up to "
            f"{measured['selection_worst']}, against "
            f"{exact['selection_losses']} with an exact evaluator",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
