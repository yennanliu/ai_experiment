"""Exercise 5 — SVR on sin(x): the ε-tube, and the points outside it.

    Implement SVR (epsilon-insensitive loss). Fit it to y = sin(x) + noise. Plot
    the epsilon tube around the predictions and highlight the support vectors
    (points outside the tube).

Reading of the exercise: a *linear* SVR cannot fit a sine, so the basis has to be
stated rather than assumed — [x, x³/6, x⁵/120], sin's own Taylor terms, which
makes check 5 possible: the learned weights should come out near (1, −1, 1).

"Points outside the tube" are exactly the support vectors, because the
ε-insensitive loss is flat inside it — and check 3 shows the consequence the plot
cannot: once ε exceeds the noise scale, **zero** points are outside, there are no
support vectors at all, and the weights collapse toward zero.
"""

from __future__ import annotations

import math
import random

from harness import practice

SEED, N, NOISE = 42, 120, 0.1
LR, EPOCHS, LAM = 0.01, 4_000, 0.001
EPSILONS = (0.05, 0.15, 0.4, 1.0)
TAYLOR = (1.0, -1.0, 1.0)


def features(x):
    return [x, x ** 3 / 6, x ** 5 / 120]


def fit(X, y, epsilon):
    """ε-insensitive loss: no gradient at all while |error| <= ε."""
    w, b = [0.0] * len(X[0]), 0.0
    for _ in range(EPOCHS):
        for row, target in zip(X, y):
            error = target - (sum(a * c for a, c in zip(w, row)) + b)
            if abs(error) > epsilon:
                sign = 1.0 if error > 0 else -1.0
                w = [wj + LR * (sign * row[j] - LAM * wj) for j, wj in enumerate(w)]
                b += LR * sign
            else:
                w = [wj - LR * LAM * wj for wj in w]
    return w, b


def _measure(X, y, xs, epsilon):
    w, b = fit(X, y, epsilon)
    predictions = [sum(a * c for a, c in zip(w, row)) + b for row in X]
    return {"outside": sum(1 for p, t in zip(predictions, y) if abs(p - t) > epsilon),
            "true_error": sum(abs(p - math.sin(x))
                              for p, x in zip(predictions, xs)) / len(xs),
            "w": w, "norm": math.sqrt(sum(v * v for v in w))}


def solve():
    rng = random.Random(SEED)
    xs = sorted(rng.uniform(-math.pi, math.pi) for _ in range(N))
    y = [math.sin(x) + rng.gauss(0, NOISE) for x in xs]
    X = [features(x) for x in xs]
    rows = {e: _measure(X, y, xs, e) for e in EPSILONS}
    best = min(EPSILONS, key=lambda e: rows[e]["true_error"])
    return {"rows": rows, "best": best, "n": N, "noise": NOISE}


def _join(rows, template) -> str:
    return ", ".join(template(e, rows[e]) for e in EPSILONS)


def verify(result):
    rows, best = result["rows"], result["best"]
    tight, wide = rows[EPSILONS[0]], rows[EPSILONS[-1]]
    outside = [rows[e]["outside"] for e in EPSILONS]
    return [
        practice.Check(f"a tight tube leaves most points outside it",
                       tight["outside"] > result["n"] // 2,
                       f"ε={EPSILONS[0]:g}: {tight['outside']} of {result['n']} points "
                       f"outside — those are the support vectors, the only ones with a "
                       f"non-zero gradient"),
        practice.Check("the support count falls monotonically as the tube widens",
                       all(a >= b for a, b in zip(outside, outside[1:])),
                       "outside the tube: "
                       + _join(rows, lambda e, r: f"ε={e:g}: {r['outside']}")),
        practice.Check(f"FINDING: past ε≈{EPSILONS[2]:g} there are NO support vectors left",
                       outside[2] == 0 and outside[-1] == 0,
                       f"once ε exceeds the residual scale, every point sits inside the "
                       f"tube, the data term vanishes for all of them, and only weight "
                       f"decay remains. At ε={EPSILONS[-1]:g} the weights collapse to "
                       f"‖w‖ = {wide['norm']:.3f} against {rows[best]['norm']:.3f} at the "
                       f"best ε — the model stops learning rather than fitting loosely"),
        practice.Check(f"error against the true sine is U-shaped, best at ε={best:g}",
                       rows[best]["true_error"] < tight["true_error"]
                       and rows[best]["true_error"] < wide["true_error"],
                       "mean |prediction − sin(x)|: "
                       + _join(rows, lambda e, r: f"ε={e:g}: {r['true_error']:.4f}")
                       + f" — the optimum sits near the noise scale σ={result['noise']:g}, "
                         f"which is what ε is for: ignore residuals you cannot explain"),
        practice.Check("…and at that ε the weights recover sin's Taylor coefficients",
                       max(abs(a - b) for a, b in zip(rows[best]["w"], TAYLOR)) < 0.3,
                       f"learned {[round(v, 3) for v in rows[best]['w']]} against "
                       f"{list(TAYLOR)} for x − x³/6 + x⁵/120 — the basis was chosen to "
                       f"make this checkable, since a linear SVR on x alone could not fit "
                       f"a sine at all"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
