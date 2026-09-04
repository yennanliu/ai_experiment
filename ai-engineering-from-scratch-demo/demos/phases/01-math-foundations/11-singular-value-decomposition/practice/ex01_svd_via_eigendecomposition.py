"""Exercise 1 — SVD from the eigendecomposition of AᵀA, three ways compared.

    Implement the full SVD from scratch without using power iteration. Instead,
    compute the eigendecomposition of A^T A to get V and the singular values,
    then compute U = A V Sigma^{-1}. Compare numerical accuracy with your power
    iteration version and with NumPy.

Reading of the exercise: the prescribed route is the numerically *worse* one and
the exercise asks for the accuracy comparison, so the comparison is the answer.
Forming AᵀA squares the condition number, so σ below √ε·σ₁ ≈ 1e-8·σ₁ cannot be
recovered. Both findings come from running the three routes on a matrix with σ
spanning 1e-9.

That matrix comes from Householder reflections, not `numpy.linalg.qr` of random
draws. `qr` returns *a* valid factorisation and LAPACK builds differ on which, so
one seed gave a different A on macOS than on Linux CI — and the AᵀA route's
relative error read 8.7e-09 on one, 1.9e-11 on the other. The bands below stay
loose anyway and assert claims, not magnitudes: `eigh` on a Gram matrix with
κ ≈ 1e18 is where implementations are entitled to differ.
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


def _orthonormal(numpy, rows, cols, vectors):
    """`cols` orthonormal columns in R^rows, as a product of Householder
    reflections I − 2vvᵀ/vᵀv — elementary arithmetic, so it is portable."""
    matrix = numpy.eye(rows)
    for raw in vectors:
        v = numpy.asarray(raw, dtype=float).reshape(-1, 1)
        matrix = matrix @ (numpy.eye(rows) - 2.0 * (v @ v.T) / float((v.T @ v)[0, 0]))
    return matrix[:, :cols]


def _ill_conditioned(numpy):
    """σ spanning 1e-9, from a platform-independent construction."""
    U = _orthonormal(numpy, 6, 4, ([1, 2, 3, 4, 5, 6], [1, -1, 2, -2, 3, -3]))
    V = _orthonormal(numpy, 4, 4, ([2, 1, 3, 1], [1, -3, 1, 2]))
    return U @ numpy.diag(SIGMAS) @ V.T


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy — uv sync --extra math")
    ref = parity.load_reference(PHASE, LESSON, "svd")
    rng = numpy.random.default_rng(SEED)
    matrices = {"well-conditioned": rng.normal(size=(6, 4)),
                "ill-conditioned (κ=1e9)": _ill_conditioned(numpy)}
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
        practice.Check("the AᵀA route reproduces A = UΣVᵀ, relative to σ₁",
                       well["recon"] < 1e-12 and hard["recon"] / hard["truth"][0] < 1e-7,
                       f"worst |UΣVᵀ − A|: well-conditioned {well['recon']:.3g}, "
                       f"ill-conditioned {hard['recon']:.3g} against σ₁ = "
                       f"{hard['truth'][0]:.3g}. The ill-conditioned bound is relative and "
                       f"loose because the residual there is LAPACK-dependent"),
        practice.Check("U comes out orthonormal, as U = AVΣ⁻¹ requires",
                       well["orthogonality"] < 1e-12,
                       f"worst |UᵀU − I| = {well['orthogonality']:.3g}"),
        practice.Check("on a well-conditioned matrix it matches NumPy to machine precision",
                       well["gram_err"] < 1e-14,
                       f"worst relative error {well['gram_err']:.3g} against "
                       f"numpy.linalg.svd"),
        practice.Check("FINDING: on an ill-conditioned one it loses most of the digits",
                       hard["gram_err"] > 1e-12 and well["gram_err"] < 1e-13,
                       f"relative error {hard['gram_err']:.3g} against "
                       f"{well['gram_err']:.3g} — {hard['gram_err'] / well['gram_err']:.0g}x "
                       f"worse. Forming AᵀA squares the condition number, so σ below "
                       f"√ε·σ₁ ≈ 1e-8·σ₁ is unrecoverable. True σ = "
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
