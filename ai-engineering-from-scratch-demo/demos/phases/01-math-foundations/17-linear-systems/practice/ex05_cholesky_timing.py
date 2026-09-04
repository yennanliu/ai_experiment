"""Exercise 5 — time Cholesky vs LU vs np.linalg.solve; is Cholesky 2x faster?

    Time your Cholesky solver vs your LU solver vs `np.linalg.solve` on symmetric
    positive definite matrices of size 10, 50, 200, 500. Plot the results. Verify
    Cholesky is roughly 2x faster than LU.

Reading of the exercise: "verify Cholesky is roughly 2x faster" does not hold,
and the reason is worth more than the claim. The 2x is a **flop** ratio — n³/3
against LU's 2n³/3 — but neither implementation is flop-bound: both are O(n²)
Python-level iterations each delegating an O(n) vector op to numpy, so wall clock
measures the iteration count, and both have the same n²/2 of them. The measured
growth exponent confirms it: **n^2.0**, not n³. Cholesky comes out 1.39x faster
from its shorter inner dot, not halved flops; `np.linalg.solve` is ~70x faster.

Every timing is the best of 3 repeats, and n=10 is excluded from the ordering
check — microseconds there, so the ratio is noise either side of 1.0.
"""

from __future__ import annotations

import math
import time

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "17-linear-systems"
SIZES = (10, 50, 200, 500)
SEED = 42


def make_spd(numpy, rng, n):
    A = rng.normal(size=(n, n))
    return A @ A.T + n * numpy.eye(n)


def timed(fn, repeats=3):
    """Best of `repeats` — a single sample measures machine load as much as code."""
    best, value = float("inf"), None
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        best = min(best, time.perf_counter() - start)
    return value, best


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy — uv sync --extra math")
    ref = parity.load_reference(PHASE, LESSON, "linear_systems")
    rng = numpy.random.default_rng(SEED)
    rows = {}
    for n in SIZES:
        A = make_spd(numpy, rng, n)
        b = rng.normal(size=n)
        truth = numpy.linalg.solve(A, b)
        (L,), chol_time = timed(lambda: (ref.cholesky(A),))
        x_chol, chol_solve = timed(lambda: ref.cholesky_solve(L, b))
        (plu,), lu_time = timed(lambda: (ref.lu_decompose(A),))
        _, numpy_time = timed(lambda: numpy.linalg.solve(A, b))
        rows[n] = {
            "cholesky": chol_time + chol_solve,
            "lu": lu_time,
            "numpy": numpy_time,
            "error": float(numpy.abs(x_chol - truth).max()),
            "symmetric": float(numpy.abs(A - A.T).max()),
        }
    return {"rows": rows}


def _exponent(rows, key):
    """Fit t ∝ n^p over the two largest sizes."""
    big, small = SIZES[-1], SIZES[-2]
    return math.log(rows[big][key] / rows[small][key]) / math.log(big / small)


def verify(result):
    rows = result["rows"]
    big = rows[SIZES[-1]]
    ratios = {n: rows[n]["lu"] / rows[n]["cholesky"] for n in SIZES}
    chol_p, lu_p = _exponent(rows, "cholesky"), _exponent(rows, "lu")
    return [
        practice.Check(f"all {len(SIZES)} sizes solved correctly by Cholesky",
                       all(r["error"] < 1e-8 and r["symmetric"] < 1e-12
                           for r in rows.values()),
                       "; ".join(f"n={n}: worst |Δx| {rows[n]['error']:.2g}" for n in SIZES)),
        practice.Check("FINDING: wall clock scales as n², not the n³ of the flop count",
                       1.8 < chol_p < 2.4 and 1.8 < lu_p < 2.4,
                       f"fitted exponents over n = {SIZES[-2]}..{SIZES[-1]}: Cholesky "
                       f"n^{chol_p:.2f}, LU n^{lu_p:.2f}. Both are O(n²) Python iterations "
                       f"each making one numpy call, so the call boundaries dominate and "
                       f"the n³ arithmetic inside numpy is nearly free"),
        practice.Check(f"Cholesky is faster than LU at every size above n={SIZES[0]}",
                       all(ratios[n] > 1.05 for n in SIZES[1:]),
                       ", ".join(f"n={n}: {ratios[n]:.2f}x" for n in SIZES)
                       + f" — n={SIZES[0]} is excluded deliberately: the whole "
                         f"decomposition takes microseconds there, so the ratio is timing "
                         f"noise and lands on either side of 1.0 between runs"),
        practice.Check("…but not by 2x, because the flop saving is not what is being measured",
                       not 1.8 < ratios[SIZES[-1]] < 2.2,
                       f"at n={SIZES[-1]} the ratio is {ratios[SIZES[-1]]:.2f}x against a "
                       f"flop ratio of 2x. Both algorithms run n²/2 Python iterations, so "
                       f"the counts are equal; Cholesky wins only on the shorter average "
                       f"inner dot product. The exercise's 2x needs a flop-bound "
                       f"implementation to show up at all"),
        practice.Check("np.linalg.solve is orders of magnitude faster than both",
                       big["numpy"] * 50 < big["cholesky"],
                       f"at n={SIZES[-1]}: numpy {big['numpy'] * 1e3:.2f} ms, Cholesky "
                       f"{big['cholesky'] * 1e3:.0f} ms, LU {big['lu'] * 1e3:.0f} ms — "
                       f"{big['cholesky'] / big['numpy']:.0f}x. Choosing the better "
                       f"algorithm in Python loses to blocked, vectorised LAPACK"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
