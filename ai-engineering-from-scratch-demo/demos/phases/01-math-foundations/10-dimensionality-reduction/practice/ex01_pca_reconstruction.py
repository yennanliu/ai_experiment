"""Exercise 1 — reconstruct digits from 10, 50 and 200 components.

    Modify the PCA class to support `inverse_transform`. Reconstruct MNIST
    digits from 10, 50, and 200 components. Print the reconstruction error
    (mean squared difference from the original) for each.

Reading of the exercise: two adjustments, both forced by the data rather than by
preference. `inverse_transform` already exists on the lesson's PCA, so this
verifies it instead of rewriting it (as in lessons 01 and 05). And "MNIST" is
read as sklearn's bundled `load_digits` — 8x8, 64 features, no download, so the
exercise stays T0. That caps the component count at 64, which makes the third
requested value, 200, impossible; check 5 records what the lesson's PCA does when
asked for more components than the data has dimensions.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "10-dimensionality-reduction"
REQUESTED = (10, 50, 200)


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy and scikit-learn — uv sync --extra math")
    try:
        from sklearn.datasets import load_digits
    except ImportError:
        raise practice.Skip("needs scikit-learn — uv sync --extra math") from None
    ref = parity.load_reference(PHASE, LESSON, "dim_reduction")
    X = load_digits().data
    rows = {}
    for k in REQUESTED:
        model = ref.PCA(n_components=k)
        reduced = model.fit_transform(X)
        rebuilt = model.inverse_transform(reduced)
        rows[k] = {"error": float(ref.reconstruction_error(X, rebuilt)),
                   "shape": tuple(reduced.shape),
                   "kept": len(model.explained_variance_ratio_),
                   "variance": float(model.explained_variance_ratio_.sum())}
    full = ref.PCA(n_components=X.shape[1])
    exact = ref.reconstruction_error(X, full.inverse_transform(full.fit_transform(X)))
    return {"rows": rows, "n_features": X.shape[1], "n_samples": X.shape[0],
            "exact_error": float(exact), "spread": float(X.var())}


def verify(result):
    rows = result["rows"]
    errors = [rows[k]["error"] for k in REQUESTED]
    return [
        practice.Check(f"{result['n_samples']} digits, {result['n_features']} features each",
                       result["n_features"] == 64,
                       "sklearn's bundled load_digits: 8x8 images, no download, so T0"),
        practice.Check("error falls as components are added",
                       errors[0] > errors[1] >= errors[2],
                       ", ".join(f"k={k}: MSE {rows[k]['error']:.4f}" for k in REQUESTED)
                       + f" against a pixel variance of {result['spread']:.1f}"),
        practice.Check("10 components already capture most of the variance",
                       rows[10]["variance"] > 0.7,
                       ", ".join(f"k={k}: {rows[k]['variance']:.1%} of variance"
                                 for k in REQUESTED)),
        practice.Check("all 64 components reconstruct exactly — PCA is a rotation",
                       result["exact_error"] < 1e-25,
                       f"MSE {result['exact_error']:.3g} with k = {result['n_features']}: "
                       f"keeping every component loses nothing, so the error at smaller k "
                       f"is entirely the discarded variance"),
        practice.Check("FINDING: asking for 200 of 64 components silently returns 64",
                       rows[200]["kept"] == result["n_features"]
                       and rows[200]["shape"][1] == result["n_features"],
                       f"n_components=200 yields a {rows[200]['shape']} projection and "
                       f"{rows[200]['kept']} components, not 200 — the slice "
                       f"eigenvectors[:, :200] just runs out, with no error. The exercise's "
                       f"third value is unreachable on 8x8 digits, and nothing says so"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
