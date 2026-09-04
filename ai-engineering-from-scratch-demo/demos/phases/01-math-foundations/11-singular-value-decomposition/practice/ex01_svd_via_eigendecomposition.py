"""Exercise 1 — SVD from the eigendecomposition of AᵀA, three ways compared.

    Implement the full SVD from scratch without using power iteration. Instead,
    compute the eigendecomposition of A^T A to get V and the singular values,
    then compute U = A V Sigma^{-1}. Compare numerical accuracy with your power
    iteration version and with NumPy.

Reading of the exercise: the prescribed route is the numerically *worse* one, and
the exercise asks for the accuracy comparison itself — so the comparison is the
answer. Forming AᵀA squares the condition number, so singular values below
√ε·σ₁ ≈ 1e-8·σ₁ cannot be recovered. Both findings below come from running the
three routes on a matrix built to have σ spanning 1e-9.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "11-singular-value-decomposition"
SEED = 42
SIGMAS = (1.0, 1e-3, 1e-6, 1e-9)


def svd_via_gram(numpy, A):
    """V and Σ from eigh(AᵀA), then U = A V Σ⁻¹ — exactly as prescribed."""
    eigenvalues, eigenvectors = numpy.linalg.eigh(A.T @ A)
    order = numpy.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    singular = numpy.sqrt(numpy.clip(eigenvalues, 0.0, None))
    keep = singular > singular[0] * 1e-15
    singular, V = singular[keep], eigenvectors[:, keep]
    return A @ V / singular, singular, V.T


def _ill_conditioned(numpy, rng):
    Q, _ = numpy.linalg.qr(rng.normal(size=(6, 4)))
    R, _ = numpy.linalg.qr(rng.normal(size=(4, 4)))
    return Q @ numpy.diag(SIGMAS) @ R.T


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy — uv sync --extra math")
    ref = parity.load_reference(PHASE, LESSON, "svd")
    rng = numpy.random.default_rng(SEED)
    matrices = {"well-conditioned": rng.normal(size=(6, 4)),
                "ill-conditioned (κ=1e9)": _ill_conditioned(numpy, rng)}
    rows = {}
    for label, matrix in matrices.items():
        truth = numpy.linalg.svd(matrix, compute_uv=False)
        U, S, Vt = svd_via_gram(numpy, matrix)
        power = ref.svd_from_scratch(matrix)
        rows[label] = {
            "truth": truth.tolist(),
            "gram_err": float(numpy.abs(S - truth[: len(S)]).max() / truth[0]),
            "recon": float(numpy.abs(U * S @ Vt - matrix).max()),
            "orthogonality": float(numpy.abs(U.T @ U - numpy.eye(U.shape[1])).max()),
            "n_truth": len(truth),
            "n_power": len(numpy.asarray(power[1])),
        }
    return {"rows": rows}


def verify(result):
    well = result["rows"]["well-conditioned"]
    hard = result["rows"]["ill-conditioned (κ=1e9)"]
    return [
        practice.Check("the AᵀA route reproduces A = UΣVᵀ",
                       well["recon"] < 1e-12 and hard["recon"] < 1e-12,
                       f"worst |UΣVᵀ − A|: well-conditioned {well['recon']:.3g}, "
                       f"ill-conditioned {hard['recon']:.3g}"),
        practice.Check("U comes out orthonormal, as U = AVΣ⁻¹ requires",
                       well["orthogonality"] < 1e-12,
                       f"worst |UᵀU − I| = {well['orthogonality']:.3g}"),
        practice.Check("on a well-conditioned matrix it matches NumPy to machine precision",
                       well["gram_err"] < 1e-14,
                       f"worst relative error {well['gram_err']:.3g} against "
                       f"numpy.linalg.svd"),
        practice.Check("FINDING: on an ill-conditioned one it loses half the digits",
                       hard["gram_err"] > 1e6 * max(well["gram_err"], 1e-16),
                       f"relative error {hard['gram_err']:.3g} against "
                       f"{well['gram_err']:.3g} — forming AᵀA squares the condition number, "
                       f"so σ below √ε·σ₁ ≈ 1e-8·σ₁ is unrecoverable. True σ = "
                       f"{[f'{v:.0e}' for v in hard['truth']]}. NumPy bidiagonalises A "
                       f"directly and keeps full precision on the same matrix"),
        practice.Check("FINDING: the lesson's svd_from_scratch silently returns fewer σ",
                       hard["n_power"] < hard["n_truth"] == 4,
                       f"svd_from_scratch gives {hard['n_power']} of {hard['n_truth']} "
                       f"singular values, with no error: it breaks out when an AᵀA "
                       f"eigenvalue drops below 1e-10, i.e. whenever σ < 1e-5. On the "
                       f"well-conditioned matrix it returns all {well['n_power']}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
