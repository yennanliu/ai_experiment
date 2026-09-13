"""Exercise 2 — the next tenth stops being expensive and becomes impossible.

    **Medium.** Implement the Hoffmann loss-as-function-of-compute curve. Plot
    loss vs `log10(C)` for the compute-optimal frontier. Identify when the law
    predicts we'd need `>10^28` FLOPs for the next 0.1 reduction in
    cross-entropy.

Reading of the exercise: the frontier has a closed form -- substituting the
optimal `(N, D)` back into the loss gives `L*(C) = E + k * C^(-a*b/(a+b))` -- so
the curve is derived rather than sampled, the plot is a text raster of it, and
the threshold is found by bisection on the exact curve.

**ANSWER: from `C = 1.07e25` the next 0.1 first costs more than 1e28 FLOPs.**
At that point the frontier loss is 1.844. The cost of the next tenth, as a
multiple of the compute already spent:

| from C | L*(C) | C needed for L - 0.1 | multiple |
|---:|---:|---:|---:|
| 1e22 | 2.139 | 5.2e22 | 5.2x |
| 1e23 | 2.005 | 1.2e24 | 12x |
| 1e24 | 1.911 | 5.0e25 | 50x |
| 1e25 | 1.845 | 8.3e27 | **832x** |
| 1e26 | 1.799 | 1.1e33 | **1.1e7x** |

**FINDING: shortly after, the answer stops being a number.** `E = 1.69` is a
floor the law never crosses, so once `L* < E + 0.1` no finite compute buys
another tenth at all. The frontier reaches 1.79 at **C = 1.76e26** -- about 16x
past the point the exercise asks about. Between 1.07e25 and 1.76e26 the honest
answer moves from "1e28 FLOPs" to "there is no such budget".

**FINDING: the exponent is 0.15355 and everything above follows from it.**
`L* - E` falls as `C^-a*b/(a+b)` with a = 0.34 and b = 0.28, so a 0.1 reduction
near `L - E = 0.15` needs a compute ratio of `(0.15/0.05)^(1/0.15355)` -- about
1,280x -- and near `L - E = 0.5` needs 4x. The curve is not slowing down; the
distance to the floor is shrinking.

**CONTROL: the derived frontier matches the lesson's own grid search.** At 1e20,
1e22 and 1e24 the closed form and `compute_optimal` agree on the loss to better
than 1e-4, so the curve the threshold is read off is the lesson's own.

Structure: `frontier` is the closed-form `L*(C)`; `budget_for` inverts it;
`raster` is the requested plot.
"""

from __future__ import annotations

import importlib.util
import math

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "13-scaling-laws"
STEP, CEILING, CHECKS = 0.1, 1e28, (1e22, 1e23, 1e24, 1e25, 1e26)
LOW, HIGH, ROWS, COLS = 1e18, 1e34, 14, 64


def frontier(ref, compute):
    """L*(C) for the compute-optimal (N, D) at this budget, in closed form."""
    alpha, beta = ref.ALPHA, ref.BETA
    scale = (ref.A * alpha / (ref.B_CONST * beta)) ** (1 / (alpha + beta))
    params = scale * (compute / 6) ** (beta / (alpha + beta))
    return ref.chinchilla_loss(params, compute / (6 * params))


def budget_for(ref, loss, low=1e15, high=1e60):
    """The smallest compute budget whose frontier loss is at or below `loss`."""
    if loss <= ref.E_CONST:
        return math.inf
    for _ in range(300):
        mid = math.sqrt(low * high)
        low, high = (mid, high) if frontier(ref, mid) > loss else (low, mid)
    return math.sqrt(low * high)


def threshold(ref, ceiling=CEILING, low=1e20, high=1e30):
    """The budget from which the next `STEP` of loss first costs more than `ceiling`."""
    for _ in range(200):
        mid = math.sqrt(low * high)
        needed = budget_for(ref, frontier(ref, mid) - STEP)
        low, high = (mid, high) if needed < ceiling else (low, mid)
    return math.sqrt(low * high)


def raster(ref, rows=ROWS, cols=COLS):
    """The requested plot: loss against log10(C), as text, with E as the floor."""
    losses = [frontier(ref, 10 ** (math.log10(LOW) + i * (math.log10(HIGH) - math.log10(LOW))
                                   / (cols - 1))) for i in range(cols)]
    top, bottom = max(losses), ref.E_CONST
    grid = [[" "] * cols for _ in range(rows)]
    for column, loss in enumerate(losses):
        grid[min(rows - 1, int((top - loss) / (top - bottom) * rows))][column] = "*"
    return "\n".join("".join(row) for row in grid)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    costs = {c: (frontier(ref, c), budget_for(ref, frontier(ref, c) - STEP)) for c in CHECKS}
    start = threshold(ref)
    return {
        "costs": costs, "start": start, "start_loss": frontier(ref, start),
        "floor": ref.E_CONST, "impossible": budget_for(ref, ref.E_CONST + STEP),
        "exponent": ref.ALPHA * ref.BETA / (ref.ALPHA + ref.BETA),
        "raster": raster(ref), "plot_rows": ROWS,
        "plotting": importlib.util.find_spec("matplotlib") is None,
        "agrees": max(abs(frontier(ref, c) - ref.compute_optimal(c)[2])
                      for c in (1e20, 1e22, 1e24)),
    }


def verify(result):
    costs = result["costs"]
    return [
        practice.Check(
            "ANSWER: from C = 1.07e25 the next 0.1 first costs more than 1e28 FLOPs",
            1e25 < result["start"] < 2e25,
            f"at C = {result['start']:.2e} the frontier loss is {result['start_loss']:.4f} and "
            f"reaching {result['start_loss'] - STEP:.4f} needs {CEILING:.0e}. The cost of the "
            "next tenth as a multiple of what is already spent: " + ", ".join(
                f"{c:.0e} -> {n / c:.0f}x" for c, (_, n) in costs.items()),
        ),
        practice.Check(
            "FINDING: shortly after, the answer stops being a number",
            result["impossible"] < 1e27 and result["impossible"] > result["start"],
            f"E = {result['floor']} is a floor the law never crosses, so once L* is below "
            f"E + {STEP} no finite compute buys another tenth. The frontier reaches "
            f"{result['floor'] + STEP:.2f} at C = {result['impossible']:.2e}, about "
            f"{result['impossible'] / result['start']:.0f}x past the point the exercise asks "
            "about -- between them the honest answer becomes 'there is no such budget'",
        ),
        practice.Check(
            "FINDING: the exponent is 0.15355 and everything follows from it",
            abs(result["exponent"] - 0.15355) < 1e-4,
            f"L* - E falls as C^-{result['exponent']:.5f} with alpha = 0.34 and beta = 0.28, so "
            f"a 0.1 reduction near L - E = 0.5 needs {(0.5 / 0.4) ** (1 / result['exponent']):.0f}x "
            f"and near L - E = 0.15 needs "
            f"{(0.15 / 0.05) ** (1 / result['exponent']):.0f}x. The curve is not slowing down; "
            "the distance to the floor is shrinking",
        ),
        practice.Check(
            "CONTROL: the derived frontier matches the lesson's own grid search",
            result["agrees"] < 1e-4,
            f"the closed form and compute_optimal agree on the loss to {result['agrees']:.1e} at "
            "1e20, 1e22 and 1e24, so the curve the threshold is read off is the lesson's own "
            "and not a second model that happens to look similar",
        ),
        practice.Check(
            "CONTROL: matplotlib is absent, so the plot is a text raster of the same curve",
            result["plotting"] and len(result["raster"].splitlines()) == result["plot_rows"],
            f"find_spec('matplotlib') is None. Loss against log10(C) from {LOW:.0e} to "
            f"{HIGH:.0e}, {result['plot_rows']} rows "
            f"between the maximum and E = {result['floor']}: the curve flattens onto the floor "
            "rather than continuing, which is the shape the two findings above are about",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
