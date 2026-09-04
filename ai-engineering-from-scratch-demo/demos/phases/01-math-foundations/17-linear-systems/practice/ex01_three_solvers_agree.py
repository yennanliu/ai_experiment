"""Exercise 1 — one system, three solvers, same answer.

    Solve the system `[[1,2,3],[4,5,6],[7,8,10]] x = [6, 15, 27]` using your
    Gaussian elimination, your LU solver, and `np.linalg.solve`. Verify all three
    give the same answer within floating-point tolerance.

Reading of the exercise: three solvers agreeing is necessary but weak — they
could agree on a wrong answer. So the residual ‖Ax − b‖ is checked, and a fourth
solution is computed by **exact rational elimination** over `fractions.Fraction`,
which has no floating-point error at all. That is a ground truth none of the
three float solvers produced, and it comes out [3, −3, 3] exactly.
"""

from __future__ import annotations

from fractions import Fraction

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "17-linear-systems"
A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]
B = [6.0, 15.0, 27.0]
TOL = 1e-9


def _eliminate(rows, col, n):
    """Normalise the pivot row, then clear the column above and below it."""
    pivot = next(r for r in range(col, n) if rows[r][col] != 0)
    rows[col], rows[pivot] = rows[pivot], rows[col]
    rows[col] = [v / rows[col][col] for v in rows[col]]
    for r in range(n):
        if r != col and rows[r][col] != 0:
            factor = rows[r][col]
            rows[r] = [v - factor * w for v, w in zip(rows[r], rows[col])]


def exact_solve(a, b):
    """Gauss-Jordan over Fractions — no rounding at all, so this is the truth."""
    n = len(b)
    rows = [[Fraction(v) for v in row] + [Fraction(rhs)] for row, rhs in zip(a, b)]
    for col in range(n):
        _eliminate(rows, col, n)
    return [row[n] for row in rows]


def residual(a, x, b):
    return max(abs(sum(row[j] * x[j] for j in range(len(x))) - rhs)
               for row, rhs in zip(a, b))


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy — uv sync --extra math")
    ref = parity.load_reference(PHASE, LESSON, "linear_systems")
    # the lesson's solvers take numpy arrays, not lists
    matrix, rhs = numpy.array(A), numpy.array(B)
    gauss = list(ref.gaussian_elimination(matrix, rhs))
    P, L, U = ref.lu_decompose(matrix)
    lu = list(ref.lu_solve(P, L, U, rhs))
    theirs = list(numpy.linalg.solve(matrix, rhs))
    rows = {"gaussian_elimination": gauss, "lu_solve": lu, "np.linalg.solve": theirs}
    exact = exact_solve(A, B)
    return {
        "rows": rows,
        "exact": [str(v) for v in exact],
        "residuals": {k: residual(A, v, B) for k, v in rows.items()},
        "errors": {k: max(abs(float(a) - float(b)) for a, b in zip(v, exact))
                   for k, v in rows.items()},
        "condition": float(ref.condition_number(matrix)),
        "det": float(numpy.linalg.det(matrix)),
    }


def verify(result):
    rows = result["rows"]
    names = list(rows)
    pairwise = max(abs(a - b) for i, x in enumerate(names) for y in names[i + 1:]
                   for a, b in zip(rows[x], rows[y]))
    return [
        practice.Check("all three solvers agree within 1e-9",
                       pairwise < TOL,
                       f"worst pairwise difference {pairwise:.3g}; "
                       + ", ".join(f"{k} -> {[round(v, 9) for v in x]}"
                                   for k, x in rows.items())),
        practice.Check("…and they agree with an exact rational solution",
                       max(result["errors"].values()) < TOL,
                       f"Fraction-based elimination gives x = {result['exact']} with no "
                       f"rounding at all — an independent ground truth. Worst float error "
                       f"{max(result['errors'].values()):.3g}"),
        practice.Check("every residual ‖Ax − b‖∞ is at machine precision",
                       max(result["residuals"].values()) < 1e-12,
                       ", ".join(f"{k}: {v:.2g}" for k, v in result["residuals"].items())),
        practice.Check("the matrix is genuinely invertible, det = −3",
                       abs(result["det"] + 3.0) < 1e-9,
                       f"det = {result['det']:.9f}; note [[1,2,3],[4,5,6],[7,8,9]] would be "
                       f"singular — the 10 in the corner is what makes this solvable"),
        practice.Check("…and well enough conditioned that agreement means something",
                       result["condition"] < 1e3,
                       f"κ(A) = {result['condition']:.1f}, so all three methods keep about "
                       f"{16 - len(str(int(result['condition']))):.0f} digits. Three "
                       f"solvers agreeing on an ill-conditioned system would prove much "
                       f"less, which is what exercise 3 goes after"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
