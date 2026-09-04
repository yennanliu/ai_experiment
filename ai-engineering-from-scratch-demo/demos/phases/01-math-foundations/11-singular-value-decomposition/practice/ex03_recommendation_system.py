"""Exercise 3 — a 10x8 ratings matrix, row-mean filled, rank-3 reconstructed.

    Build a tiny recommendation system. Create a 10x8 user-movie ratings matrix
    with some known entries. Fill missing entries with row means. Compute SVD and
    reconstruct a rank-3 approximation. Use the reconstructed matrix to predict
    the missing ratings. Verify that the predictions are reasonable.

Reading of the exercise: "verify that the predictions are reasonable" is the only
assertable clause and it needs a definition. Two are used, because the weak one
alone is worthless: predictions must land in the valid 1–5 range (check 3), and
they must beat the row-mean baseline they were *initialised from* on a held-out
set (check 4). The second is the real test — a rank-3 reconstruction of
row-mean-filled data can easily just reproduce the row means, in which case the
SVD contributed nothing.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "11-singular-value-decomposition"
RANK, SEED = 3, 42
USERS, MOVIES = 10, 8
LOW, HIGH = 1.0, 5.0


def _latent_ratings(numpy, rng):
    """Two taste groups over two genre blocks, so a low-rank signal exists."""
    taste = rng.normal(size=(USERS, 2))
    genre = rng.normal(size=(2, MOVIES))
    raw = taste @ genre
    scaled = 3.0 + 1.2 * (raw - raw.mean()) / raw.std()
    return numpy.clip(scaled, LOW, HIGH)


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy — uv sync --extra math")
    ref = parity.load_reference(PHASE, LESSON, "svd")
    rng = numpy.random.default_rng(SEED)
    truth = _latent_ratings(numpy, rng)
    observed = rng.random((USERS, MOVIES)) > 0.35        # ~65% known
    for i in range(USERS):                               # every user rates something
        observed[i, rng.integers(MOVIES)] = True
    filled = truth.copy()
    row_means = numpy.array([truth[i][observed[i]].mean() for i in range(USERS)])
    for i in range(USERS):
        filled[i, ~observed[i]] = row_means[i]
    U, S, Vt = ref.truncated_svd(filled, RANK)
    predicted = numpy.clip(ref.reconstruct(U, S, Vt), LOW, HIGH)
    hidden = ~observed
    svd_error = float(numpy.abs(predicted[hidden] - truth[hidden]).mean())
    baseline = float(numpy.abs(
        numpy.repeat(row_means[:, None], MOVIES, axis=1)[hidden] - truth[hidden]).mean())
    spectrum = numpy.linalg.svd(filled, compute_uv=False)
    return {"n_hidden": int(hidden.sum()), "n_total": USERS * MOVIES,
            "svd_error": svd_error, "baseline": baseline,
            "min": float(predicted.min()), "max": float(predicted.max()),
            "observed_error": float(numpy.abs(
                predicted[observed] - truth[observed]).mean()),
            "energy": float((spectrum[:RANK] ** 2).sum() / (spectrum ** 2).sum())}


def verify(result):
    return [
        practice.Check(f"{result['n_hidden']} of {result['n_total']} ratings hidden, "
                       f"filled with row means",
                       0 < result["n_hidden"] < result["n_total"] // 2,
                       f"{100 * result['n_hidden'] / result['n_total']:.0f}% missing; "
                       f"rank-{RANK} factors hold {result['energy']:.1%} of the "
                       f"filled matrix's energy"),
        practice.Check("the reconstruction stays close on the ratings it was given",
                       result["observed_error"] < 0.5,
                       f"mean absolute error {result['observed_error']:.4f} on observed "
                       f"entries — a rank-3 fit that missed these would predict nothing"),
        practice.Check(f"every prediction lands inside the valid {LOW:g}–{HIGH:g} range",
                       LOW <= result["min"] and result["max"] <= HIGH,
                       f"predictions span {result['min']:.3f} to {result['max']:.3f}"),
        practice.Check("predictions beat the row-mean baseline they were initialised from",
                       result["svd_error"] < result["baseline"],
                       f"MAE on hidden entries: SVD {result['svd_error']:.4f} vs row-mean "
                       f"baseline {result['baseline']:.4f} — "
                       f"{100 * (1 - result['svd_error'] / result['baseline']):.0f}% better"),
        practice.Check("…which is the check that matters, since the fill *was* the baseline",
                       result["svd_error"] < 0.95 * result["baseline"],
                       "a rank-3 reconstruction of row-mean-filled data can simply reproduce "
                       "those row means and look fine on every other measure. Beating them "
                       "is the only evidence the factorisation learned cross-user structure"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
