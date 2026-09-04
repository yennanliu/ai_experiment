"""Exercise 2 — eigenvalues of [[4,2],[1,3]] by hand, then two ways in code.

    Find the eigenvalues of the matrix [[4, 2], [1, 3]] by hand using the
    characteristic equation. Then verify with your from-scratch function and
    with NumPy.

Reading of the exercise: "by hand" cannot be a printed comment — it has to be a
*separate derivation the code does not share* with the function under test, or
it verifies nothing. So the hand result is entered as the closed-form roots of
the characteristic polynomial, computed here from the trace and determinant, and
the three routes are compared pairwise.

By hand: det(A − λI) = (4−λ)(3−λ) − 2·1 = λ² − 7λ + 10 = (λ−5)(λ−2), so λ = 5, 2.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "03-matrix-transformations"
MATRIX = [[4, 2], [1, 3]]
BY_HAND = (5.0, 2.0)
TOL = 1e-12


def characteristic_roots(matrix):
    """λ² − (trace)λ + det = 0, solved directly — the by-hand route."""
    trace = matrix[0][0] + matrix[1][1]
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    root = (trace * trace - 4 * det) ** 0.5
    return ((trace + root) / 2, (trace - root) / 2)


def _worst(a, b) -> float:
    return max(abs(x - y) for x, y in zip(sorted(a, reverse=True), sorted(b, reverse=True)))


def _residuals_ok(residuals) -> bool:
    return all(r <= TOL for r in residuals.values())


def _fmt(values) -> str:
    return str([round(float(v), 12) for v in values])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "transformations")
    theirs = ref.eigenvalues_2x2(MATRIX)
    mine = characteristic_roots(MATRIX)
    vectors = {value: ref.eigenvector_2x2(MATRIX, value) for value in theirs}
    residuals = {}
    for value, vector in vectors.items():
        av = ref.mat_vec_mul(MATRIX, vector)
        residuals[value] = max(abs(a - value * v) for a, v in zip(av, vector))
    return {"mine": mine, "theirs": theirs, "vectors": vectors, "residuals": residuals}


def verify(result):
    mine, theirs = result["mine"], result["theirs"]
    total, product = sum(theirs), theirs[0] * theirs[1]
    checks = [
        practice.Check("by hand: λ² − 7λ + 10 = 0 gives λ = 5, 2",
                       _worst(mine, BY_HAND) <= TOL, f"derived {_fmt(mine)}"),
        practice.Check("the lesson's from-scratch function agrees",
                       _worst(mine, theirs) <= TOL, f"eigenvalues_2x2 -> {_fmt(theirs)}"),
        practice.Check("each eigenvector satisfies A v = λ v",
                       _residuals_ok(result["residuals"]),
                       "; ".join(f"λ={k:g}: worst |Av − λv| {v:.3g}"
                                 for k, v in result["residuals"].items())),
        practice.Check("trace == Σλ and det == Πλ (the invariants that pin them down)",
                       abs(total - 7) <= TOL and abs(product - 10) <= TOL,
                       f"Σλ {total:g} == trace 7, Πλ {product:g} == det 10"),
    ]
    numpy = parity.try_numpy()
    if numpy is None:
        checks.append(practice.Check("numpy comparison", False,
                                     "the exercise names NumPy; uv sync --extra math"))
        return checks
    theirs_np = numpy.linalg.eigvals(numpy.array(MATRIX, dtype=float)).real
    gap = _worst(mine, theirs_np)
    checks.append(practice.Check("numpy.linalg.eigvals agrees to 1e-12", gap <= TOL,
                                 f"numpy -> {_fmt(theirs_np)}, worst gap {gap:.3g}"))
    return checks


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
