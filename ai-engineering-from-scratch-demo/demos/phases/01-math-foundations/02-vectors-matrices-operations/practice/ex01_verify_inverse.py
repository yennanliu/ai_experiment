"""Exercise 1 — verify A @ A.inverse_2x2() is the identity.

    **Verify the inverse.** Multiply `A @ A.inverse_2x2()` and confirm you get
    the identity matrix. Try it with three different 2x2 matrices. What happens
    when the determinant is zero?

Reading of the exercise: "confirm you get the identity" is a floating-point
claim, so it is checked as a residual against a tolerance, worst value reported.
The last sentence is the real question — it asks what *happens*, so the answer
has to exercise the singular case and state the observed behaviour rather than
predict it. Check 4 records what the lesson's code actually does there.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "02-vectors-matrices-operations"
TOL = 1e-12

INVERTIBLE = {
    "generic": [[4, 7], [2, 6]],
    "negative determinant": [[1, 2], [3, 4]],
    "near-singular (det 1e-8)": [[1, 1], [1, 1 + 1e-8]],
}
SINGULAR = [[2, 4], [1, 2]]           # det == 0: row 2 is row 1 halved


def _residual(product) -> float:
    """max |A A⁻¹ − I| over the 4 entries."""
    identity = [[1, 0], [0, 1]]
    return max(abs(product.data[i][j] - identity[i][j]) for i in range(2) for j in range(2))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "matrices")
    rows = {}
    for label, data in INVERTIBLE.items():
        matrix = ref.Matrix(data)
        rows[label] = {"det": matrix.determinant(),
                       "residual": _residual(matrix @ matrix.inverse_2x2())}
    singular = ref.Matrix(SINGULAR)
    try:
        singular.inverse_2x2()
        outcome = "returned a matrix — no error raised"
    except Exception as exc:
        outcome = f"{type(exc).__name__}: {exc}"
    return {"rows": rows, "singular_det": singular.determinant(), "singular": outcome}


def verify(result):
    checks = []
    for label, row in result["rows"].items():
        checks.append(practice.Check(
            f"{label}: A A⁻¹ == I", row["residual"] <= TOL,
            f"det {row['det']:g}, worst |entry − I| {row['residual']:.3g} (tol {TOL:g})"))
    checks.append(practice.Check(
        "a zero determinant is refused, not silently wrong",
        "Error" in result["singular"] or "error" in result["singular"],
        f"det {result['singular_det']:g} -> {result['singular']}"))
    numpy = parity.try_numpy()
    if numpy is not None:
        worst = max(
            float(abs(numpy.array(data) @ numpy.linalg.inv(numpy.array(data, dtype=float))
                      - numpy.eye(2)).max())
            for data in INVERTIBLE.values())
        checks.append(practice.Check("numpy's inverse is no better conditioned",
                                     worst < 1e-6, f"numpy worst residual {worst:.3g}"))
    return checks


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
