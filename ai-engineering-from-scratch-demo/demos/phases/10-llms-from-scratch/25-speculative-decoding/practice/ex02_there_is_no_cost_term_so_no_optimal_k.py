"""Exercise 2 — expected tokens rises to 1/(1-alpha) and stays, so "optimal K" has no answer in this module.

    Compute the speedup formula. Given fixed `α` and `K`, plot expected tokens
    per target-forward. Find the optimal K for α ∈ {0.5, 0.7, 0.9}.

Reading of the exercise: the plot is the lesson's own `expected_tokens` across
K, and "find the optimal K" is then asked of it directly -- the function is
non-decreasing in K, so an optimum requires a cost that falls on the other side
of the trade, and whether the module contains one is checkable.

**ANSWER: the curve saturates and never turns, so the optimum is the largest K
tried.**

    alpha    K=1    K=2    K=4    K=8   K=16   K=32   K=64   ceiling
     0.5    1.50   1.75   1.94   2.00   2.00   2.00   2.00     2.00
     0.7    1.70   2.19   2.77   3.20   3.33   3.33   3.33     3.33
     0.9    1.90   2.71   4.10   6.13   8.33   9.69   9.99    10.00

`expected_tokens` is `(1 - alpha^(K+1)) / (1 - alpha)`, a partial geometric sum:
strictly increasing in K, bounded by `1 / (1 - alpha)`, and reaching **96.9%** of
that bound by K=32 at alpha=0.9 and 100.0% at the other two.

**FINDING: there is no cost term anywhere in the module.** No name in it mentions
cost, latency, time or speed; `expected_tokens` takes `(alpha, K)` and nothing
else. The quantity that makes K a trade -- the draft's own forward passes, K of
them per round -- is not represented, so the exercise's question has no answer
the lesson can supply.

**MECHANISM: the answer K is where the marginal token stops paying.** With a
draft costing `c` per token relative to the target's 1, the wall-clock figure is
`(1 + K*c) / expected_tokens(alpha, K)`, and *that* has an interior minimum: at
`c = 0.05` it is **K=3** for alpha=0.5, **K=6** for 0.7 and **K=13** for 0.9. The
optimum moves with alpha, which is the shape the exercise is asking for.

**FINDING: what saturates is not the speedup but the acceptance run.** At
alpha=0.5 a chain longer than 8 buys nothing at all -- 2.00 tokens at K=8 and
2.00 at K=64 -- because the expected run of accepted tokens before a rejection is
`alpha / (1 - alpha)` = **1.0**. Half the drafts are rejected at the first
position, so the ninth speculative token is reached in one round in 512, and
K=8 and K=64 agree to three decimal places.

Structure: `curve` evaluates the lesson's own formula across K; `wall_clock`
adds the draft cost the module lacks and `optimum` finds where it turns.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "25-speculative-decoding"
ALPHAS = (0.5, 0.7, 0.9)
KS = (1, 2, 4, 8, 16, 32, 64)
DRAFT_COST, SCAN = 0.05, range(1, 61)


def curve(ref, alpha):
    return {k: ref.expected_tokens(alpha, k) for k in KS}


def wall_clock(ref, alpha, k, cost=DRAFT_COST):
    """Target-forwards per token once the draft's own K passes are charged."""
    return (1 + k * cost) / ref.expected_tokens(alpha, k)


def optimum(ref, alpha, cost=DRAFT_COST):
    return min(SCAN, key=lambda k: wall_clock(ref, alpha, k, cost))


def cost_terms(ref):
    """Any name in the module touching the quantity that would make K a trade."""
    words = ("cost", "latency", "time", "speed")
    return [name for name in dir(ref) if any(word in name.lower() for word in words)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {alpha: curve(ref, alpha) for alpha in ALPHAS}
    optima = {alpha: optimum(ref, alpha) for alpha in ALPHAS}
    return {
        "rows": rows,
        "ceilings": {alpha: 1 / (1 - alpha) for alpha in ALPHAS},
        "saturation": {alpha: rows[alpha][32] * (1 - alpha) for alpha in ALPHAS},
        "optima": optima,
        "optimal_value": {alpha: wall_clock(ref, alpha, k) for alpha, k in optima.items()},
        "params": list(inspect.signature(ref.expected_tokens).parameters),
        "cost_names": cost_terms(ref),
        "run_length": {alpha: alpha / (1 - alpha) for alpha in ALPHAS},
        "flat": abs(rows[0.5][8] - rows[0.5][64]) < 0.01,
    }


def column(rows, key, fmt):
    return ", ".join(f"a={alpha} {format(row[key], fmt)}" for alpha, row in rows.items())


def joined(values, fmt):
    return ", ".join(f"a={alpha} {format(value, fmt)}" for alpha, value in values.items())


def optima_line(result):
    return ", ".join(f"a={alpha} K={k} at {result['optimal_value'][alpha]:.4f}"
                     for alpha, k in result["optima"].items())


def verify(result):
    rows, ceilings = result["rows"], result["ceilings"]
    return [
        practice.Check(
            "ANSWER: the curve saturates and never turns, so the optimum is the largest K tried",
            result["flat"] and all(rows[a][64] <= ceilings[a] + 1e-9 for a in ALPHAS),
            "expected tokens at K=1 is " + column(rows, 1, ".2f")
            + " and at K=64 " + column(rows, 64, ".2f")
            + ", against ceilings of 1/(1-alpha) = " + joined(ceilings, ".2f")
            + ". The formula is a partial geometric sum: non-decreasing in K, bounded, and at "
            + joined(result["saturation"], ".1%") + " of its bound by K=32",
        ),
        practice.Check(
            "FINDING: there is no cost term anywhere in the module",
            not result["cost_names"] and result["params"] == ["alpha", "K"],
            f"expected_tokens takes {result['params']} and nothing else, and no name in the "
            f"module mentions cost, latency, time or speed -- {result['cost_names']}. The "
            "quantity that makes K a trade, the draft's own K forward passes per round, is not "
            "represented, so the exercise's question has no answer the lesson can supply",
        ),
        practice.Check(
            "MECHANISM: with a draft cost the curve turns, and the optimum moves with alpha",
            (result["optima"][0.5] < result["optima"][0.7] < result["optima"][0.9]
             and max(result["optima"].values()) < max(SCAN)),
            f"charging the draft {DRAFT_COST} per token relative to the target's 1 makes the "
            "wall-clock figure (1 + K*c) / expected_tokens(alpha, K), which has an interior "
            "minimum: " + optima_line(result)
            + ". The optimum moves with alpha, which is the shape the exercise is asking for",
        ),
        practice.Check(
            "FINDING: what saturates is the acceptance run, and at alpha=0.5 it is 1.0",
            abs(result["run_length"][0.5] - 1.0) < 1e-9,
            "the expected run of accepted tokens before a rejection is alpha / (1 - alpha) = "
            + joined(result["run_length"], ".1f")
            + f". At alpha=0.5 a chain longer than 8 buys nothing -- {rows[0.5][8]:.2f} tokens at "
            f"K=8 and {rows[0.5][64]:.2f} at K=64, a difference of "
            f"{rows[0.5][64] - rows[0.5][8]:.2e} -- because half the drafts are rejected at the "
            "first position and the ninth speculative token is reached in one round in 512",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
