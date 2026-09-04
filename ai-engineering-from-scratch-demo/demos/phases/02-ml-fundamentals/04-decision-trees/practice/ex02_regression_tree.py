"""Exercise 2 — a regression tree on sin(x) + noise, piecewise-constant by design.

    Implement variance reduction splitting for regression trees. Generate
    y = sin(x) + noise for 200 points and fit your regression tree. Plot the
    tree's piecewise-constant predictions against the true curve.

Reading of the exercise: "plot the piecewise-constant predictions" is not
assertable, but *piecewise-constant* is — check 3 counts distinct predicted values
and shows each forms one contiguous run over sorted x, which is what makes the
plot a staircase rather than a curve.

Two things depend on depth in ways the exercise does not mention. RMSE against
the true sine is **U-shaped** (0.253 / 0.111 / 0.161 at depths 2/6/12), so depth
stops helping once leaves fit noise. And "a constant cannot follow a slope" is
only visible at the right depth — steep/flat error ratio 2.35 at depth 6 against
~1.0 at both extremes. See the README.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "04-decision-trees"
SEED, N, NOISE = 42, 200, 0.15
DEPTHS = (2, 6, 12)


def make_data(rng):
    xs = sorted(rng.uniform(-math.pi, math.pi) for _ in range(N))
    return [[x] for x in xs], [math.sin(x) + rng.gauss(0, NOISE) for x in xs]


def _monotone_blocks(predictions):
    """Number of maximal runs of an identical prediction over sorted x."""
    return 1 + sum(1 for a, b in zip(predictions, predictions[1:]) if a != b)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "trees")
    rng = random.Random(SEED)
    X, y = make_data(rng)
    truth = [math.sin(row[0]) for row in X]
    rows = {d: _measure(ref, X, y, truth, d) for d in DEPTHS}
    mid = len(y) // 2
    return {"rows": rows, "n": N,
            "gain": ref.variance_reduction(y, y[:mid], y[mid:])}


def _measure(ref, X, y, truth, depth):
    tree = ref.DecisionTree(max_depth=depth, task="regression")
    tree.fit(X, y)
    predictions = tree.predict(X)
    errors = [abs(p - t) for p, t in zip(predictions, truth)]
    slope = [abs(math.cos(row[0])) for row in X]
    steep = [e for e, s in zip(errors, slope) if s > 0.8]
    flat = [e for e, s in zip(errors, slope) if s < 0.2]
    return {"levels": len({round(p, 12) for p in predictions}),
            "rmse": math.sqrt(sum(e * e for e in errors) / N),
            "steep_error": sum(steep) / len(steep),
            "flat_error": sum(flat) / len(flat),
            "monotone_within": _monotone_blocks(predictions)}


def _ratio(row) -> float:
    return row["steep_error"] / row["flat_error"]


def _piecewise(rows) -> bool:
    return all(rows[d]["levels"] == rows[d]["monotone_within"] for d in DEPTHS)


def _join(rows, template) -> str:
    return ", ".join(template(d, rows[d]) for d in DEPTHS)


def verify(result):
    rows = result["rows"]
    shallow, deep = rows[DEPTHS[0]], rows[DEPTHS[-1]]
    return [
        practice.Check("variance reduction is a valid split criterion here",
                       result["gain"] >= 0,
                       f"splitting sorted targets in half reduces variance by "
                       f"{result['gain']:.4f} — non-negative, as required"),
        practice.Check("FINDING: RMSE is U-shaped in depth, not monotone",
                       rows[DEPTHS[1]]["rmse"] < shallow["rmse"]
                       and rows[DEPTHS[1]]["rmse"] < deep["rmse"],
                       _join(rows, lambda d, r: f"depth {d}: {r['rmse']:.4f}")
                       + f" — depth {DEPTHS[1]} wins; depth {DEPTHS[-1]} has "
                         f"{deep['levels']} distinct values for {result['n']} points and "
                         f"is fitting noise"),
        practice.Check("the fit is piecewise constant — distinct values equal run count",
                       _piecewise(rows) and deep["levels"] > shallow["levels"],
                       _join(rows, lambda d, r: f"depth {d}: {r['levels']} distinct")
                       + " — each one contiguous over sorted x, so a staircase"),
        practice.Check(f"depth {DEPTHS[0]} emits only {shallow['levels']} values over "
                       f"a full sine period",
                       shallow["levels"] <= 4,
                       f"{shallow['levels']} levels across [−π, π] — the coarsest "
                       f"staircase"),
        practice.Check("FINDING: 'a constant cannot follow a slope' shows only at the "
                       "right depth",
                       _ratio(rows[DEPTHS[1]]) > 2.0
                       and _ratio(shallow) < 1.2 and _ratio(deep) < 1.5,
                       "steep/flat error ratio: "
                       + _join(rows, lambda d, r: f"depth {d}: {_ratio(r):.2f}")
                       + f". At depth {DEPTHS[1]} the error is "
                         f"{_ratio(rows[DEPTHS[1]]):.1f}x worse where |cos x| > 0.8 than "
                         f"where it is < 0.2. At depth {DEPTHS[0]} everything is bad and at "
                         f"depth {DEPTHS[-1]} everything is noise, so both wash the effect "
                         f"out"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
