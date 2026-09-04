"""Exercise 5 — how the optimal truncation rank moves with noise level.

    Generate a clean low-rank matrix (rank 3, size 50x40) and add Gaussian noise
    at different levels (sigma = 0.1, 0.5, 1.0, 2.0). For each noise level, find
    the optimal truncation rank by sweeping k from 1 to 40 and measuring
    reconstruction error against the clean matrix. Plot how the optimal k
    changes with noise level.

Reading of the exercise: "plot" is printed as a table, since a chart is not
assertable. The interesting part is what the sweep shows: the optimal k is **3 at
every noise level tested**, not a decreasing function of sigma. That is the right
answer and worth stating plainly — measuring error against the *clean* matrix
means extra components only ever add noise, so the optimum sits at the true rank
regardless of sigma. Check 5 shows the version of the question that does have a
sigma-dependent answer, which is the one people usually mean.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "11-singular-value-decomposition"
ROWS, COLS, TRUE_RANK = 50, 40, 3
SIGMAS = (0.1, 0.5, 1.0, 2.0)
SEED = 42


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy — uv sync --extra math")
    ref = parity.load_reference(PHASE, LESSON, "svd")
    rng = numpy.random.default_rng(SEED)
    clean = rng.normal(size=(ROWS, TRUE_RANK)) @ rng.normal(size=(TRUE_RANK, COLS))
    clean_norm = float(numpy.linalg.norm(clean))
    rows = {}
    for sigma in SIGMAS:
        noisy = clean + rng.normal(scale=sigma, size=(ROWS, COLS))
        errors = []
        for k in range(1, COLS + 1):
            U, S, Vt = ref.truncated_svd(noisy, k)
            errors.append(float(numpy.linalg.norm(clean - ref.reconstruct(U, S, Vt))
                                / clean_norm))
        best = min(range(len(errors)), key=lambda i: errors[i]) + 1
        spectrum = numpy.linalg.svd(noisy, compute_uv=False)
        # the Gavish-Donoho hard threshold for a square-ish matrix
        threshold = 2.858 * numpy.median(spectrum) if sigma else 0.0
        rows[sigma] = {"best_k": best, "best_error": errors[best - 1],
                       "error_at_true": errors[TRUE_RANK - 1],
                       "error_at_full": errors[-1],
                       "gap": float(spectrum[TRUE_RANK - 1] / spectrum[TRUE_RANK]),
                       "above_threshold": int((spectrum > threshold).sum())}
    return {"rows": rows, "clean_norm": clean_norm}


def _join(rows, template) -> str:
    return "; ".join(template(s, rows[s]) for s in SIGMAS)


def verify(result):
    rows = result["rows"]
    best_ks = [rows[s]["best_k"] for s in SIGMAS]
    ratios = {s: rows[s]["error_at_full"] / rows[s]["error_at_true"] for s in SIGMAS}
    return [
        practice.Check(f"k swept 1..{COLS} at all {len(SIGMAS)} noise levels",
                       len(rows) == len(SIGMAS),
                       _join(rows, lambda s, r: f"σ={s}: best k={r['best_k']}, "
                                                 f"error {r['best_error']:.4f}")),
        practice.Check(f"ANSWER: the optimal rank is {TRUE_RANK} at every noise level",
                       set(best_ks) == {TRUE_RANK},
                       f"best k = {best_ks} for σ = {list(SIGMAS)} — flat, not decreasing. "
                       f"Measuring against the clean matrix means components past the true "
                       f"rank can only add noise, whatever σ is"),
        practice.Check("truncating at the true rank always beats keeping everything",
                       all(rows[s]["error_at_true"] < rows[s]["error_at_full"]
                           for s in SIGMAS),
                       _join(rows, lambda s, r: f"σ={s}: k=3 → {r['error_at_true']:.4f} vs "
                                                 f"k=40 → {r['error_at_full']:.4f}")
                       + " — this gap *is* the denoising, and it widens with σ"),
        practice.Check("…and the *relative* benefit is nearly constant — it is scale-free",
                       all(2.4 < ratios[s] < 3.1 for s in SIGMAS),
                       "error ratio full/true: "
                       + _join(rows, lambda s, r: f"σ={s} → {ratios[s]:.2f}x")
                       + ". Both errors scale linearly in σ, so the ratio cancels it — the "
                         "absolute gain from truncating grows with noise, the proportional "
                         "one does not"),
        practice.Check("what *does* move with σ is how visible the true rank is",
                       rows[SIGMAS[0]]["gap"] > rows[SIGMAS[-1]]["gap"] * 3,
                       "spectral gap σ₃/σ₄: "
                       + _join(rows, lambda s, r: f"σ={s} → {r['gap']:.1f}")
                       + f". The sweep can always find k=3 because it peeks at the clean "
                         f"matrix; a practitioner cannot, and has to read the rank off the "
                         f"spectrum — which gets harder as σ grows. Gavish-Donoho "
                         f"thresholding recovers "
                       + _join(rows, lambda s, r: f"{r['above_threshold']}")
                       + " components respectively"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
