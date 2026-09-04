"""Exercise 5 — Rosenbrock's Hessian eigenvalues at (1,1) and (−1,1).

    **Hessian eigenvalue analysis.** Compute the Hessian of the Rosenbrock
    function at (1,1) and at (-1,1). Compute eigenvalues at both points. What do
    the eigenvalues tell you about the curvature at the minimum versus far from
    it?

Reading of the exercise: the two requested points have **identical eigenvalue
spectra**, which is not what the question implies. At both (1,1) and (−1,1) the
Hessian is [[802, ∓400], [∓400, 200]], and flipping the off-diagonal sign leaves
trace and determinant unchanged — so both give {1001.60, 0.3994}. The eigenvalues
cannot distinguish the global minimum from a point four units above it.

What they *do* say is that the valley is narrow: κ = 2508, which is why every
optimiser in this phase struggles here. What distinguishes the two points is the
**gradient**, zero at (1,1) and not at (−1,1) — the check a second-order
condition cannot replace. Check 5 exhibits a point where the Hessian genuinely is
indefinite, since neither of the two requested ones is.

Hessian of 100(y−x²)² + (1−x)²:
    ∂²/∂x² = 1200x² − 400y + 2,  ∂²/∂x∂y = −400x,  ∂²/∂y² = 200.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "18-convex-optimization"
POINTS = {"(1, 1) — the global minimum": (1.0, 1.0),
          "(-1, 1) — far from it": (-1.0, 1.0),
          "(0, 1) — genuinely indefinite": (0.0, 1.0)}


def rosenbrock(x, y):
    return 100 * (y - x * x) ** 2 + (1 - x) ** 2


def hessian(x, y):
    return [[1200 * x * x - 400 * y + 2, -400 * x], [-400 * x, 200.0]]


def gradient(x, y):
    return [-400 * x * (y - x * x) - 2 * (1 - x), 200 * (y - x * x)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "convex")
    rows = {}
    for label, (x, y) in POINTS.items():
        H = hessian(x, y)
        eigen = sorted(ref.hessian_eigenvalues_2d(H), reverse=True)
        rows[label] = {"hessian": H, "eigen": list(eigen),
                       "psd": bool(ref.is_positive_semidefinite_2d(H)),
                       "value": rosenbrock(x, y),
                       "gradient": gradient(x, y),
                       "kappa": abs(eigen[0] / eigen[1]) if eigen[1] else float("inf")}
    return {"rows": rows}


def _fmt(values, places=4) -> str:
    return str([round(float(v), places) for v in values])


def _same_spectrum(a, b) -> bool:
    return max(abs(x - y) for x, y in zip(a["eigen"], b["eigen"])) < 1e-9


def _indefinite(entry) -> bool:
    return min(entry["eigen"]) < 0 < max(entry["eigen"]) and not entry["psd"]


def verify(result):
    rows = result["rows"]
    minimum = rows["(1, 1) — the global minimum"]
    far = rows["(-1, 1) — far from it"]
    saddle = rows["(0, 1) — genuinely indefinite"]
    return [
        practice.Check("at (1,1) the gradient vanishes and the value is 0",
                       max(abs(g) for g in minimum["gradient"]) < 1e-12
                       and minimum["value"] == 0.0,
                       f"∇f = {minimum['gradient']}, f = {minimum['value']} — the global "
                       f"minimum, as it must be for a sum of two squares both zeroed"),
        practice.Check("…and both eigenvalues are positive, so the Hessian is PSD",
                       minimum["psd"] and min(minimum["eigen"]) > 0,
                       f"H = {minimum['hessian']}, eigenvalues "
                       f"{_fmt(minimum['eigen'])}"),
        practice.Check("ANSWER: but their ratio is 2508 — the minimum is a narrow valley",
                       minimum["kappa"] > 1000,
                       f"κ = {minimum['kappa']:.0f}: curvature along one direction is "
                       f"{minimum['kappa']:.0f}x the other. That is why gradient descent "
                       f"needs O(κ) steps here (lesson 18 exercise 2) and why Rosenbrock is "
                       f"the standard optimiser benchmark"),
        practice.Check("ANSWER: (−1,1) has the SAME eigenvalues as the minimum",
                       _same_spectrum(minimum, far) and far["psd"] and far["value"] > 0,
                       f"both give {_fmt(far['eigen'])} — the off-diagonal "
                       f"sign flips but trace and determinant do not, so the spectrum is "
                       f"identical. f = {far['value']:.0f} there against 0 at the minimum, "
                       f"and ∇f = {_fmt(far['gradient'], 1)} against zero. "
                       f"Eigenvalues alone cannot locate a minimum"),
        practice.Check("…and Rosenbrock IS indefinite elsewhere, just not at either point asked",
                       _indefinite(saddle),
                       f"at (0, 1): H = {saddle['hessian']}, eigenvalues "
                       f"{_fmt(saddle['eigen'], 1)}, det = −79600. Negative "
                       f"curvature along x, so −H⁻¹∇f there points *toward* higher "
                       f"function values — Newton needs a PSD Hessian and does not get one"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
