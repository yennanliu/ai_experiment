"""Rank statistics for solutions that compare two orderings.

Zero-dependency by the same rule as the rest of `harness` (DESIGN §4): these are
a dozen lines of arithmetic, and a solution should not need numpy or scipy
installed to ask whether two criteria rank the same candidates the same way.

Lives here rather than in a solution because "do these two scores order the same
items identically?" recurs — split criteria, distance metrics, feature
importances — and D14's per-solution ceilings are for the *answer*, not for a
measurement utility copied into every file that needs it.
"""

from __future__ import annotations


def kendall_tau(a, b) -> float:
    """Concordant pairs minus discordant, over the pairs where both rank strictly.

    Tau-b's tie handling is deliberately not implemented: ties in either input
    are dropped from the denominator rather than penalised, which is what a
    comparison of two continuous score vectors wants. `a` and `b` must be equal
    in length and at least 2 long; a pair of constant vectors has no ordered
    pairs at all and raises rather than returning a meaningless 0.
    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} against {len(b)}")
    net = ordered = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            sign = (a[i] - a[j]) * (b[i] - b[j])
            net += (sign > 0) - (sign < 0)
            ordered += sign != 0
    if not ordered:
        raise ValueError("no strictly ordered pairs: tau is undefined")
    return net / ordered


def fit_line(xs, ys) -> tuple:
    """Least-squares slope and intercept for a single feature.

    The degenerate case is explicit: a constant `xs` has no slope, and returning
    (0, mean(ys)) — the best constant predictor — is right for the callers here,
    which compare a feature against doing nothing. A caller that needs to *know*
    the feature was constant should check the spread itself.
    """
    n = len(xs)
    if n != len(ys):
        raise ValueError(f"length mismatch: {n} against {len(ys)}")
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    spread = sum((v - mean_x) ** 2 for v in xs)
    if spread < 1e-12:
        return 0.0, mean_y
    slope = sum((a - mean_x) * (b - mean_y) for a, b in zip(xs, ys)) / spread
    return slope, mean_y - slope * mean_x


def rmse(xs, ys, slope: float, intercept: float) -> float:
    """Root mean squared error of `slope * x + intercept` against `ys`."""
    return (sum((slope * x + intercept - y) ** 2 for x, y in zip(xs, ys)) / len(ys)) ** 0.5


def least_squares(np, matrix, target, n_train: int) -> dict:
    """Fit `matrix` (plus an intercept) on the first `n_train` rows; score both halves.

    `np` is passed in rather than imported: `harness` must import with nothing
    installed (DESIGN §4), and a solution that needs least squares has already
    obtained numpy through `parity.try_numpy()` and skipped if it is absent.

    `cond` is the condition number of the *training* design. It is reported
    because `lstsq` returns a minimum-norm solution for a singular design without
    complaining, so a fit can look fine while its coefficients mean nothing.
    """
    design = np.array([[1.0] + list(row) for row in matrix])
    weights = np.linalg.lstsq(design[:n_train], np.array(target[:n_train]), rcond=None)[0]
    error = design @ weights - np.array(target)
    return {"train": float(np.sqrt(np.mean(error[:n_train] ** 2))),
            "test": float(np.sqrt(np.mean(error[n_train:] ** 2))),
            "cond": float(np.linalg.cond(design[:n_train])), "k": len(matrix[0])}
