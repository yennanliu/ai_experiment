"""Exercise 2 — a 3x3 inverse by the adjugate method.

    **Implement 3x3 inverse.** Extend the Matrix class to compute inverses for
    3x3 matrices using the adjugate method. Test it against NumPy's
    `np.linalg.inv`.

Reading of the exercise: "extend the Matrix class" is taken as *subclass it*,
not edit the lesson's file — this repo never modifies the reference (D5). The
adjugate method is prescribed, so the solution uses cofactors rather than
Gaussian elimination even though elimination is better conditioned; check 4
measures exactly how much that choice costs on an ill-conditioned matrix.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "02-vectors-matrices-operations"
TOL = 1e-9

CASES = {
    "generic": [[2, -1, 0], [-1, 2, -1], [0, -1, 2]],
    "permutation-like": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    "ill-conditioned (Hilbert 3x3)": [[1, 1 / 2, 1 / 3], [1 / 2, 1 / 3, 1 / 4],
                                      [1 / 3, 1 / 4, 1 / 5]],
}


def _minor(data, row, col):
    return [[value for j, value in enumerate(r) if j != col]
            for i, r in enumerate(data) if i != row]


def _det2(m):
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


def det3(data):
    return sum((-1) ** j * data[0][j] * _det2(_minor(data, 0, j)) for j in range(3))


def inverse_3x3(data):
    """adj(A)ᵀ / det(A) — the cofactor route the exercise asks for."""
    determinant = det3(data)
    if abs(determinant) < 1e-15:
        raise ValueError("Matrix is singular, no inverse exists")
    cofactors = [[(-1) ** (i + j) * _det2(_minor(data, i, j)) for j in range(3)]
                 for i in range(3)]
    return [[cofactors[j][i] / determinant for j in range(3)] for i in range(3)]


def _residual(a, b):
    product = [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return max(abs(product[i][j] - (1.0 if i == j else 0.0))
               for i in range(3) for j in range(3))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "matrices")

    class Matrix3(ref.Matrix):
        """Extends the lesson's Matrix without touching its file."""

        def inverse_3x3(self):
            return Matrix3(inverse_3x3(self.data))

    rows = {}
    for label, data in CASES.items():
        inverse = Matrix3(data).inverse_3x3()
        rows[label] = {"inverse": inverse.data, "det": det3(data),
                       "residual": _residual(data, inverse.data)}
    return {"rows": rows, "subclass_ok": issubclass(Matrix3, ref.Matrix)}


def verify(result):
    checks = [practice.Check("Matrix is extended, not edited",
                             result["subclass_ok"], "Matrix3 subclasses the lesson's Matrix")]
    for label, row in result["rows"].items():
        checks.append(practice.Check(
            f"{label}: A A⁻¹ == I", row["residual"] <= TOL,
            f"det {row['det']:.3g}, worst |entry − I| {row['residual']:.3g} (tol {TOL:g})"))
    numpy = parity.try_numpy()
    if numpy is None:
        checks.append(practice.Check("numpy comparison", False,
                                     "the exercise names np.linalg.inv; uv sync --extra math"))
        return checks
    worst_gap, where = 0.0, ""
    for label, row in result["rows"].items():
        theirs = numpy.linalg.inv(numpy.array(CASES[label], dtype=float))
        gap = float(abs(numpy.array(row["inverse"]) - theirs).max())
        if gap > worst_gap:
            worst_gap, where = gap, label
    checks.append(practice.Check("matches np.linalg.inv entry-wise",
                                 worst_gap <= 1e-9, f"worst gap {worst_gap:.3g} on {where}"))
    return checks


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
