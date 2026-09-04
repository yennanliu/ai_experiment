"""Exercise 4 — conjugate gradient on 100x100 SPD systems; iteration count.

    Implement the conjugate gradient algorithm for a 100x100 random symmetric
    positive definite matrix. Count how many iterations it takes to converge to
    tolerance 1e-8. Compare with the theoretical maximum of n iterations.

Reading of the exercise: "the theoretical maximum of n iterations" is an
**exact-arithmetic** result, and in floating point it does not hold. Three
matrices are run, identical in size and differing only in conditioning:

    κ≈10   ->  31 iterations
    κ≈1e3  -> 178 iterations   (1.8x n)
    κ≈1e5  -> 728 iterations   (7.3x n)

Rounding destroys the conjugacy that guarantees termination in n steps, so a
poorly conditioned system needs many times n. The lesson's `conjugate_gradient`
defaults `max_iter` to n, which means it silently returns an **unconverged**
answer on two of these three — check 2. What governs the count is √κ, not n,
which is why a single random SPD matrix cannot demonstrate anything here.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "17-linear-systems"
N, SEED, TOL, BIG_CAP = 100, 42, 1e-8, 5_000
SPREADS = (("κ≈10", 10.0), ("κ≈1e3", 1e3), ("κ≈1e5", 1e5))


def make_spd(numpy, rng, spread):
    """Q diag(geomspace(1, spread)) Qᵀ — SPD with a chosen condition number."""
    Q, _ = numpy.linalg.qr(rng.normal(size=(N, N)))
    return Q @ numpy.diag(numpy.geomspace(1.0, spread, N)) @ Q.T


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy — uv sync --extra math")
    ref = parity.load_reference(PHASE, LESSON, "linear_systems")
    rng = numpy.random.default_rng(SEED)
    rows = {}
    for label, spread in SPREADS:
        A = make_spd(numpy, rng, spread)
        b = rng.normal(size=N)
        kappa = float(numpy.linalg.cond(A))
        capped, capped_iters = ref.conjugate_gradient(A, b, tol=TOL)
        full, full_iters = ref.conjugate_gradient(A, b, tol=TOL, max_iter=BIG_CAP)
        rows[label] = {
            "kappa": kappa,
            "capped_iters": capped_iters,
            "capped_residual": float(numpy.linalg.norm(A @ capped - b)),
            "iters": full_iters,
            "residual": float(numpy.linalg.norm(A @ full - b)),
            "error": float(numpy.abs(full - numpy.linalg.solve(A, b)).max()),
            "bound": 0.5 * math.sqrt(kappa) * math.log(2 / TOL),
        }
    return {"rows": rows, "n": N}


def _join(rows, labels, template) -> str:
    return "; ".join(template(k, rows[k]) for k in labels)


def _converged(rows, labels) -> bool:
    return all(rows[k]["residual"] < 1e-6 and rows[k]["error"] < 1e-5 for k in labels)


def _within_bound(rows, labels) -> bool:
    return all(rows[k]["iters"] <= 1.2 * rows[k]["bound"] for k in labels)


def verify(result):
    rows = result["rows"]
    labels = [k for k, _ in SPREADS]
    counts = [rows[k]["iters"] for k in labels]
    stalled = [k for k in labels if rows[k]["capped_iters"] >= result["n"]]
    return [
        practice.Check("given enough iterations, CG converges on all three systems",
                       _converged(rows, labels),
                       _join(rows, labels, lambda k, r: f"{k}: {r['iters']} iters, residual "
                                                        f"{r['residual']:.2g}, worst |Δx| "
                                                        f"vs a direct solve "
                                                        f"{r['error']:.2g}")),
        practice.Check(f"FINDING: the default max_iter=n returns unconverged answers",
                       len(stalled) == 2,
                       _join(rows, stalled,
                             lambda k, r: f"{k}: stopped at {r['capped_iters']} with "
                                          f"residual {r['capped_residual']:.3g}")
                       + f" — against a tolerance of {TOL:g}, and nothing is raised"),
        practice.Check(f"the 'theoretical maximum of n = {result['n']}' is exact-arithmetic only",
                       counts[1] > result["n"] and counts[2] > 5 * result["n"],
                       f"actual counts {counts} for n = {result['n']}: "
                       f"{counts[2] / result['n']:.1f}x n at κ≈1e5. Rounding destroys the "
                       f"conjugacy that guarantees termination in n steps"),
        practice.Check("ANSWER: the count tracks √κ, since n is identical throughout",
                       counts[0] < counts[1] < counts[2],
                       _join(rows, labels,
                             lambda k, r: f"{k}: √κ = {math.sqrt(r['kappa']):.0f} -> "
                                          f"{r['iters']} iters")),
        practice.Check("…and stays within 20% of the ½√κ·ln(2/tol) bound",
                       _within_bound(rows, labels),
                       _join(rows, labels,
                             lambda k, r: f"{k}: {r['iters']} vs bound {r['bound']:.0f}")
                       + " — the κ≈10 case exceeds it slightly because the bound is on the "
                         "A-norm of the error while the tolerance here is on the residual. "
                         "Preconditioning lowers κ, which is the only lever that matters"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
