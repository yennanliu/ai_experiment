"""Exercise 2 — least squares four ways, and which to trust.

    Generate a 50x5 random matrix X and target y = X @ w_true + noise. Solve for
    w using normal equations, QR (via `np.linalg.qr`), SVD (via
    `np.linalg.svd`), and `np.linalg.lstsq`. Compare all four solutions. Measure
    the condition number of X^T X and explain how it affects which method you
    trust.

Reading of the exercise: on a *random* 50x5 matrix all four agree to machine
precision, so the exercise as written cannot distinguish them — κ(XᵀX) is about
1e2 and everything works. The comparison only becomes informative when the design
matrix is ill-conditioned, so the solution runs both: a well-conditioned case
where the four agree (check 2), and one with κ(XᵀX) ≈ 1e14 where the normal
equations separate from the other three (check 4). κ(XᵀX) = κ(X)², which is the
whole reason to avoid forming it.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "17-linear-systems"
N, P, SEED = 50, 5, 42
NOISE = 0.1


def four_ways(numpy, X, y):
    normal = numpy.linalg.solve(X.T @ X, X.T @ y)
    Q, R = numpy.linalg.qr(X)
    qr = numpy.linalg.solve(R, Q.T @ y)
    U, S, Vt = numpy.linalg.svd(X, full_matrices=False)
    svd = Vt.T @ ((U.T @ y) / S)
    lstsq = numpy.linalg.lstsq(X, y, rcond=None)[0]
    return {"normal equations": normal, "QR": qr, "SVD": svd, "np.linalg.lstsq": lstsq}


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy — uv sync --extra math")
    ref = parity.load_reference(PHASE, LESSON, "linear_systems")
    rng = numpy.random.default_rng(SEED)
    w_true = rng.normal(size=P)
    designs = {}
    well = rng.normal(size=(N, P))
    designs["well-conditioned"] = well
    # ill-conditioned: one column nearly a copy of another
    ill = rng.normal(size=(N, P))
    ill[:, 1] = ill[:, 0] + 1e-7 * rng.normal(size=N)
    designs["ill-conditioned"] = ill

    rows = {}
    for label, X in designs.items():
        y = X @ w_true + NOISE * rng.normal(size=N)
        solutions = four_ways(numpy, X, y)
        best = solutions["np.linalg.lstsq"]
        rows[label] = {
            "kappa_x": float(numpy.linalg.cond(X)),
            "kappa_gram": float(ref.condition_number(X.T @ X)),
            "gaps": {k: float(numpy.abs(v - best).max()) for k, v in solutions.items()},
            "residuals": {k: float(numpy.linalg.norm(X @ v - y))
                          for k, v in solutions.items()},
        }
    return {"rows": rows}


def _log_ratio(row) -> float:
    """log κ(XᵀX) / (2 log κ(X)) — the squaring claim, stated where it survives."""
    return math.log10(row["kappa_gram"]) / (2 * math.log10(row["kappa_x"]))


def verify(result):
    well = result["rows"]["well-conditioned"]
    ill = result["rows"]["ill-conditioned"]
    ill_gaps = ill["gaps"]
    return [
        practice.Check("κ(XᵀX) = κ(X)² — the reason forming the Gram matrix costs you",
                       abs(well["kappa_gram"] / well["kappa_x"] ** 2 - 1) < 1e-6
                       and abs(_log_ratio(ill) - 1) < 0.02,
                       f"well-conditioned: κ(X) {well['kappa_x']:.4f} -> κ(XᵀX) "
                       f"{well['kappa_gram']:.4f}, the identity holding to "
                       f"{abs(well['kappa_gram'] / well['kappa_x'] ** 2 - 1):.1e}. "
                       f"Ill-conditioned: {ill['kappa_x']:.3e} -> {ill['kappa_gram']:.3e}, "
                       f"where the *ratio* is only good to "
                       f"{abs(ill['kappa_gram'] / ill['kappa_x'] ** 2 - 1):.1%} — κ(XᵀX) at "
                       f"1e14 is itself computed from singular values at the double-"
                       f"precision floor, so the identity is checked in log space "
                       f"({_log_ratio(ill):.4f}) where the estimate is reliable"),
        practice.Check("on a random 50x5 matrix all four methods agree to 1e-12",
                       max(well["gaps"].values()) < 1e-12,
                       f"worst gap {max(well['gaps'].values()):.3g} at κ(XᵀX) = "
                       f"{well['kappa_gram']:.1f} — the exercise as written cannot tell the "
                       f"four apart, because there is nothing to tell apart"),
        practice.Check("…and every residual is identical there too",
                       max(well["residuals"].values()) - min(well["residuals"].values())
                       < 1e-12,
                       f"‖Xw − y‖ ≈ {min(well['residuals'].values()):.6f} for all four"),
        practice.Check("ANSWER: at κ(XᵀX) ≈ 1e14 the normal equations separate from the rest",
                       ill_gaps["normal equations"] > 100 * max(
                           ill_gaps["QR"], ill_gaps["SVD"]),
                       f"normal equations differ from lstsq by "
                       f"{ill_gaps['normal equations']:.3g}, against QR {ill_gaps['QR']:.3g} "
                       f"and SVD {ill_gaps['SVD']:.3g} — QR and SVD never form XᵀX, so they "
                       f"work at κ(X) rather than κ(X)²"),
        practice.Check("…yet all four still fit the data, which is why residuals mislead",
                       max(ill["residuals"].values()) / min(ill["residuals"].values()) < 1.5,
                       ", ".join(f"{k}: {v:.4f}" for k, v in ill["residuals"].items())
                       + " — a near-duplicate column means many w fit almost equally well, "
                         "so a small residual is no evidence the coefficients are right. "
                         "Trust QR or SVD, and check κ before trusting any of them"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
