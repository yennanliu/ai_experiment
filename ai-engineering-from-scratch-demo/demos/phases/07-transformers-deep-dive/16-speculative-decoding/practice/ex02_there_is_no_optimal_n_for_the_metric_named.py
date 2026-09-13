"""Exercise 2 — the metric the exercise names has no optimal N.

    **Medium.** Plot speedup (tokens per big-model forward) as a function of `N`
    for `α = 0.5, 0.7, 0.85`. Identify the optimal `N` for each α. (Hint:
    expected tokens per verify call = `(1 - α^{N+1}) / (1 - α)`.)

Reading of the exercise: the hint *is* "tokens per big-model forward", so the
function to be optimised is the one the hint gives. It is examined before it is
plotted, and the plot is a text raster since `matplotlib` is absent.

**ANSWER: there is no optimal N. The function is strictly increasing in N for
every α, and saturates at `1 / (1 - α)`.**

| α | N=1 | N=3 | N=5 | N=12 | ceiling |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 1.500 | 1.875 | 1.969 | 2.000 | **2.000** |
| 0.70 | 1.700 | 2.533 | 2.941 | 3.301 | **3.333** |
| 0.85 | 1.850 | 3.187 | 4.152 | 5.861 | **6.667** |

Each extra draft position adds exactly `α^N` tokens, so the gain never turns
negative -- it decays geometrically to zero. "Identify the optimal N" has the
answer "as large as you like", which is not what the exercise means.

**FINDING: the optimum appears as soon as the draft is charged for.** Tokens per
unit of *time* is `E(N) / (1 + N*c)` for a draft costing `c` verifier-forwards,
and that has an interior maximum:

| draft cost c | α=0.5 | α=0.7 | α=0.85 |
|---|---|---|---|
| 1/3 | N*=1 (1.12x) | N*=2 (1.31x) | N*=3 (1.59x) |
| 1/10 | N*=2 (1.46x) | N*=4 (1.98x) | N*=7 (2.85x) |
| 1/20 | N*=3 (1.63x) | N*=6 (2.35x) | N*=10 (3.70x) |

The missing variable is the draft model's cost, and it is missing from the hint,
from `expected_tokens_per_verify`, and from the exercise.

**FINDING: the formula assumes α does not decay with depth, and so does the
lesson's simulator.** `spec_step_n`'s docstring says "Simplified: q and p are
fixed per call" -- it redraws every draft token from the same `p` without
advancing the draft model's context, which is exactly the independence the
closed form needs. Give the acceptance rate a 10% decay per position and E(5) at
α₀ = 0.85 falls from **4.152 to 3.380**, 19% lower; at 20% decay, to **2.927**.

**CONTROL: the two guards in `expected_tokens_per_verify` are unreachable and
redundant.** `alpha == 0` returns 1, which the formula also gives; `alpha >= 1`
returns `N + 1`, which is the limit. Neither changes an answer.

Structure: `curve` is the closed form over N; `timed` charges the draft;
`decayed` recomputes E(N) when acceptance falls with depth.
"""

from __future__ import annotations

import importlib.util

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "16-speculative-decoding"
ALPHAS, DEPTHS, COSTS, DECAYS = (0.5, 0.7, 0.85), (1, 3, 5, 12), (1 / 3, 1 / 10, 1 / 20), (0.9, 0.8)
MAX_N, ROWS, COLS = 30, 12, 40


def curve(ref, alpha, upto=MAX_N):
    """Expected tokens per verifier call for N = 1 .. upto."""
    return [ref.expected_tokens_per_verify(alpha, n) for n in range(1, upto + 1)]


def timed(ref, alpha, cost, upto=MAX_N):
    """(best N, best tokens per unit time) when a draft forward costs `cost` verifiers."""
    scores = [(ref.expected_tokens_per_verify(alpha, n) / (1 + n * cost), n)
              for n in range(1, upto + 1)]
    best = max(scores)
    return best[1], best[0]


def decayed(alpha, n, decay):
    """E(N) when the acceptance rate is multiplied by `decay` at each further position."""
    total, running = 1.0, 1.0
    for step in range(n):
        running *= alpha * decay ** step
        total += running
    return total


def raster(ref, rows=ROWS, cols=COLS):
    """The requested plot: E(N) against N for each alpha, as text."""
    ceiling = 1 / (1 - max(ALPHAS))
    grid = [[" "] * cols for _ in range(rows)]
    for mark, alpha in zip("abc", ALPHAS):
        for n in range(cols):
            value = ref.expected_tokens_per_verify(alpha, n + 1)
            grid[min(rows - 1, int((ceiling - value) / ceiling * rows))][n] = mark
    return "\n".join("".join(row) for row in grid)


def shape(curves):
    """(strictly rising everywhere, each gain is exactly alpha^N, the printable rows)."""
    rising = all(b > a for c in curves.values() for a, b in zip(c, c[1:]))
    gains = all(abs((c[n] - c[n - 1]) - a ** (n + 1)) < 1e-12
                for a, c in curves.items() for n in range(1, len(c)))
    rows = ", ".join(f"{a}: " + "/".join(f"{curves[a][n - 1]:.3f}" for n in DEPTHS)
                     + f" -> {1 / (1 - a):.3f}" for a in ALPHAS)
    return rising, gains, rows


def charged(ref):
    """{(alpha, cost): (best N, best tokens per unit time)} plus its printable form."""
    best = {(a, round(c, 3)): timed(ref, a, c) for a in ALPHAS for c in COSTS}
    return best, ", ".join(f"c={c:.3f} alpha={a} N*={best[(a, round(c, 3))][0]} "
                           f"({best[(a, round(c, 3))][1]:.2f}x)" for c in COSTS for a in ALPHAS)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    curves = {a: curve(ref, a) for a in ALPHAS}
    rising, gains, rows = shape(curves)
    optimal, shown = charged(ref)
    return {
        "optimal": optimal, "rising": rising, "gains": gains, "rows": rows, "shown": shown,
        "interior": all(n < MAX_N for n, _ in optimal.values()),
        "decay": {d: decayed(0.85, 5, d) for d in DECAYS},
        "flat": ref.expected_tokens_per_verify(0.85, 5),
        "guards": (ref.expected_tokens_per_verify(0.0, 5), ref.expected_tokens_per_verify(1.0, 5)),
        "raster": raster(ref), "plotting": importlib.util.find_spec("matplotlib") is None,
    }


def verify(result):
    optimal = result["optimal"]
    return [
        practice.Check(
            "ANSWER: there is no optimal N -- the function rises for every alpha",
            result["rising"] and result["gains"],
            f"E(N) by alpha: {result['rows']}"
            + f". Strictly increasing out to N={MAX_N}, saturating at 1/(1-alpha), with each "
              "extra position adding exactly alpha^N -- a gain that decays to zero and never "
              "turns negative",
        ),
        practice.Check(
            "FINDING: the optimum appears as soon as the draft is charged for",
            result["interior"] and optimal[(0.85, 0.05)][0] > optimal[(0.85, 0.333)][0],
            f"E(N)/(1 + N*c) has an interior maximum: {result['shown']}"
            + ". The missing variable is the draft model's cost, and it is missing from the hint, "
              "from expected_tokens_per_verify and from the exercise",
        ),
        practice.Check(
            "FINDING: the formula assumes alpha does not decay with depth",
            result["decay"][0.9] < result["flat"] * 0.85,
            f"spec_step_n redraws every draft from the same p without advancing the draft model's "
            f"context -- its docstring says 'Simplified: q and p are fixed per call' -- which is "
            f"exactly the independence the closed form needs. At alpha0 = 0.85, E(5) is "
            f"{result['flat']:.3f} flat, {result['decay'][0.9]:.3f} with a 10% per-position decay "
            f"and {result['decay'][0.8]:.3f} with 20%",
        ),
        practice.Check(
            "CONTROL: the two guards in expected_tokens_per_verify are redundant",
            result["guards"] == (1.0, 6),
            f"alpha == 0 returns {result['guards'][0]}, which (1 - 0^(N+1))/(1 - 0) also gives; "
            f"alpha >= 1 returns N + 1 = {result['guards'][1]}, which is the limit of the same "
            "expression. Neither branch changes an answer, and neither is reachable from main()",
        ),
        practice.Check(
            "CONTROL: matplotlib is absent, so the plot is a text raster of the same curves",
            result["plotting"] and len(result["raster"].splitlines()) == ROWS,
            f"find_spec('matplotlib') is None. Three curves over N = 1..{COLS}, a for alpha=0.5, "
            f"b for 0.7, c for 0.85, each flattening onto its own ceiling rather than turning "
            "over -- which is the shape the answer above is about",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
