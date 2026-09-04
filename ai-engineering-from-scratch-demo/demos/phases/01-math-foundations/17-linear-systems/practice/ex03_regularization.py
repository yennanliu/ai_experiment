"""Exercise 3 — a nearly singular system, with and without ridge regularization.

    Create a nearly singular matrix by making two columns almost identical (e.g.,
    column 2 = column 1 + 1e-10 * noise). Compute its condition number. Solve
    Ax = b with and without regularization (add 0.01 * I). Compare the solutions
    and residuals. Explain why regularization helps.

Reading of the exercise: "compare the solutions and residuals" is a trap that
the checks are built to expose. The *residual* barely moves — regularization
makes it slightly worse, which it must, since it is no longer solving the stated
problem. What improves is **stability**: check 4 perturbs b by 1e-8 and measures
how far the solution moves. Unregularized it moves by orders of magnitude;
regularized it barely moves. That is the answer to "why does regularization
help", and a residual comparison alone would suggest it does not.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "17-linear-systems"
N, SEED, LAMBDA = 6, 42, 0.01
PERTURBATION = 1e-8


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy — uv sync --extra math")
    ref = parity.load_reference(PHASE, LESSON, "linear_systems")
    rng = numpy.random.default_rng(SEED)
    A = rng.normal(size=(N, N))
    A[:, 1] = A[:, 0] + 1e-10 * rng.normal(size=N)      # near-duplicate column
    b = rng.normal(size=N)
    b_perturbed = b + PERTURBATION * rng.normal(size=N)

    plain = numpy.linalg.solve(A, b)
    plain_perturbed = numpy.linalg.solve(A, b_perturbed)
    ridge = ref.ridge_regression(A, b, LAMBDA)
    ridge_perturbed = ref.ridge_regression(A, b_perturbed, LAMBDA)
    return {
        "kappa": float(ref.condition_number(A)),
        "plain_norm": float(numpy.linalg.norm(plain)),
        "ridge_norm": float(numpy.linalg.norm(ridge)),
        "plain_residual": float(numpy.linalg.norm(A @ plain - b)),
        "ridge_residual": float(numpy.linalg.norm(A @ ridge - b)),
        "plain_shift": float(numpy.linalg.norm(plain_perturbed - plain)),
        "ridge_shift": float(numpy.linalg.norm(ridge_perturbed - ridge)),
        "predicted_shift": float(ref.condition_number(A)) * PERTURBATION,
    }


def verify(result):
    return [
        practice.Check("the near-duplicate column makes κ(A) enormous",
                       result["kappa"] > 1e9,
                       f"κ(A) = {result['kappa']:.3e}, so about "
                       f"{len(str(int(result['kappa']))):.0f} of 16 digits are gone before "
                       f"any arithmetic happens"),
        practice.Check("the unregularized solution has a huge norm",
                       result["plain_norm"] > 100 * result["ridge_norm"],
                       f"‖x‖ = {result['plain_norm']:.3e} unregularized against "
                       f"{result['ridge_norm']:.4f} with ridge — the two near-identical "
                       f"columns get large opposing coefficients that nearly cancel"),
        practice.Check("TRAP: the residual gets WORSE with regularization",
                       result["ridge_residual"] > result["plain_residual"],
                       f"‖Ax − b‖: {result['plain_residual']:.3g} plain, "
                       f"{result['ridge_residual']:.4f} ridge. It must — ridge solves a "
                       f"different problem. Comparing residuals alone says regularization "
                       f"hurts"),
        practice.Check("ANSWER: what improves is stability under a 1e-8 perturbation of b",
                       result["plain_shift"] > 1e4 * result["ridge_shift"],
                       f"perturbing b by {PERTURBATION:g} moves the plain solution by "
                       f"{result['plain_shift']:.3g} and the ridge solution by "
                       f"{result['ridge_shift']:.3g} — "
                       f"{result['plain_shift'] / result['ridge_shift']:.1e}x apart"),
        practice.Check("…and the plain shift is what κ(A)·δ predicts",
                       result["plain_shift"] < result["predicted_shift"],
                       f"κ(A)·δ = {result['predicted_shift']:.3g} bounds the observed "
                       f"{result['plain_shift']:.3g}. Ridge replaces κ(A) with "
                       f"κ(A + λI) ≈ σ₁/λ, which is why a λ of {LAMBDA} buys so much: it "
                       f"puts a floor under the smallest singular value"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
