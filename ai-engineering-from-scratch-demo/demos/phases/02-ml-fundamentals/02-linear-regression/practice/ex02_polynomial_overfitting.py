"""Exercise 2 — fit degree 1, 3 and 10 to cubic data; where does overfitting show?

    Generate data from a cubic function (y = ax^3 + bx^2 + cx + d + noise). Fit
    polynomials of degree 1, 3, and 10. Compare training R^2 and test R^2. At
    what degree does overfitting become obvious?

Reading of the exercise: "obvious" needs a criterion, so the train/test R² gap is
measured rather than eyeballed. Two things the framing hides, both measured: a
small gap is *not* the signature of underfitting (degree 1's +0.078 exceeds degree
3's +0.060 — the low training score of 0.789 is what identifies it), and the
coefficients are not the tell either (degree 10's largest is 2.9x degree 3's while
its test R² is −0.89, worse than the mean). See the README.
"""

from __future__ import annotations

import random

from harness import practice

SEED, N_TRAIN, N_TEST, NOISE = 42, 30, 200, 3.0
DEGREES = (1, 3, 10)
COEFFS = (0.5, -1.0, 2.0, 1.0)          # a x³ + b x² + c x + d


def truth(x):
    a, b, c, d = COEFFS
    return a * x ** 3 + b * x ** 2 + c * x + d


def design(xs, degree):
    return [[x ** p for p in range(degree + 1)] for x in xs]


def _normal_system(A, y):
    """[AᵀA | Aᵀy], the augmented normal-equation system."""
    n = len(A[0])
    return [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(n)]
            + [sum(A[k][i] * y[k] for k in range(len(A)))] for i in range(n)]


def _eliminate(M, col, n):
    pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
    M[col], M[pivot] = M[pivot], M[col]
    if abs(M[col][col]) < 1e-14:
        return
    M[col] = [v / M[col][col] for v in M[col]]
    for r in range(n):
        if r != col and M[r][col]:
            factor = M[r][col]
            M[r] = [v - factor * w for v, w in zip(M[r], M[col])]


def solve_normal(A, y):
    """Least squares by Gauss-Jordan on the normal equations."""
    M, n = _normal_system(A, y), len(A[0])
    for col in range(n):
        _eliminate(M, col, n)
    return [row[n] for row in M]


def r_squared(xs, ys, coeffs):
    mean = sum(ys) / len(ys)
    predict = lambda x: sum(c * x ** p for p, c in enumerate(coeffs))
    ss_res = sum((y - predict(x)) ** 2 for x, y in zip(xs, ys))
    return 1 - ss_res / sum((y - mean) ** 2 for y in ys)


def solve():
    rng = random.Random(SEED)
    train_x = [rng.uniform(-3, 3) for _ in range(N_TRAIN)]
    train_y = [truth(x) + rng.gauss(0, NOISE) for x in train_x]
    test_x = [rng.uniform(-3, 3) for _ in range(N_TEST)]
    test_y = [truth(x) + rng.gauss(0, NOISE) for x in test_x]
    rows = {}
    for degree in DEGREES:
        coeffs = solve_normal(design(train_x, degree), train_y)
        train, test = (r_squared(train_x, train_y, coeffs),
                       r_squared(test_x, test_y, coeffs))
        rows[degree] = {"train": train, "test": test, "gap": train - test,
                        "max_coeff": max(abs(c) for c in coeffs),
                        "n_params": degree + 1}
    return {"rows": rows, "n_train": N_TRAIN}


def verify(result):
    rows = result["rows"]
    one, three, ten = rows[1], rows[3], rows[10]
    best = max(DEGREES, key=lambda d: rows[d]["test"])
    return [
        practice.Check("training R² rises with degree, as it must",
                       one["train"] < three["train"] < ten["train"],
                       ", ".join(f"deg {d}: {rows[d]['train']:.4f}" for d in DEGREES)
                       + " — training R² alone says nothing"),
        practice.Check("degree 3 wins on test — it is the true model",
                       best == 3,
                       ", ".join(f"deg {d}: {rows[d]['test']:.4f}" for d in DEGREES)),
        practice.Check("degree 1 underfits — identified by a low TRAINING score",
                       one["train"] < 0.8 and one["gap"] > three["gap"],
                       f"train {one['train']:.4f}, test {one['test']:.4f}; its gap "
                       f"{one['gap']:+.4f} *exceeds* degree 3's {three['gap']:+.4f}"),
        practice.Check("ANSWER: overfitting is obvious at degree 10, from the gap",
                       ten["gap"] > 5 * three["gap"],
                       f"gap: {one['gap']:+.4f} / {three['gap']:+.4f} / {ten['gap']:+.4f} "
                       f"— {ten['gap'] / three['gap']:.0f}x degree 3's, with "
                       f"{ten['n_params']} parameters for {result['n_train']} points"),
        practice.Check("…and degree 10's test R² is NEGATIVE — worse than the mean",
                       ten["test"] < 0 and ten["max_coeff"] < 10 * three["max_coeff"],
                       f"test R² {ten['test']:.4f} — predicting the mean scores 0 and "
                       f"beats it. Yet its largest coefficient is {ten['max_coeff']:.1f} "
                       f"against {three['max_coeff']:.3f}, only "
                       f"{ten['max_coeff'] / three['max_coeff']:.1f}x"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
