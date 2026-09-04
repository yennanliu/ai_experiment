"""Exercise 3 — logsumexp_stable on three edge cases the naive version fails.

    **Log-sum-exp edge cases.** Test your `logsumexp_stable` function with:
    (a) all values equal, (b) one value much larger than the rest, (c) all values
    very negative (-1000). Verify it gives correct results where the naive
    version fails.

Reading of the exercise: each case has a closed form, so "correct" means matching
that rather than matching the stable function to itself. All-equal gives
log(n) + v. One dominant value gives that value plus a vanishing correction. All
at −1000 gives −1000 + log(n). Case (c) is the interesting one and not for the
reason the exercise implies: the naive version does not overflow there, it
**underflows**: every exp(−1000) is 0, the sum is 0, and `math.log(0)` raises
`ValueError` rather than returning −inf. Two different failure modes from one
function, and only one of them is an overflow.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "13-numerical-stability"
CASES = {
    "(a) all equal": [3.0] * 5,
    "(b) one dominant": [1000.0, 1.0, 2.0, 3.0],
    "(c) all very negative": [-1000.0] * 4,
    "(d) mixed extremes": [-1000.0, 1000.0, 0.0],
}


def closed_form(values):
    """log Σ eˣ, computed by hand for each case's known structure."""
    peak = max(values)
    return peak + math.log(sum(math.exp(v - peak) for v in values))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "numerical")
    rows = {}
    for label, values in CASES.items():
        try:
            naive = ref.logsumexp_naive(values)
        except (OverflowError, ValueError) as exc:
            naive = f"{type(exc).__name__}: {exc}"
        rows[label] = {"stable": ref.logsumexp_stable(values),
                       "naive": naive, "exact": closed_form(values)}
    return {"rows": rows, "n": {k: len(v) for k, v in CASES.items()}}


def _broken(value):
    return isinstance(value, str) or not math.isfinite(value)


def verify(result):
    rows = result["rows"]
    equal, dominant, negative, mixed = (rows[k] for k in CASES)
    worst = max(abs(r["stable"] - r["exact"]) for r in rows.values())
    return [
        practice.Check("(a) all equal: log(n) + v, to machine precision",
                       abs(equal["stable"] - (math.log(5) + 3.0)) < 1e-12,
                       f"5 copies of 3.0 -> {equal['stable']:.12f} = log(5) + 3 = "
                       f"{math.log(5) + 3.0:.12f}"),
        practice.Check("(b) one dominant value: the naive version overflows",
                       _broken(dominant["naive"])
                       and abs(dominant["stable"] - dominant["exact"]) < 1e-12,
                       f"stable {dominant['stable']:.9f}, naive -> {dominant['naive']}; "
                       f"exp(1000) is beyond float64's 1.8e308 ceiling"),
        practice.Check("(c) all at −1000: the naive version UNDERFLOWS, it does not overflow",
                       dominant["naive"] != negative["naive"]
                       and _broken(negative["naive"]),
                       f"stable {negative['stable']:.9f} = −1000 + log(4); naive -> "
                       f"{negative['naive']} — every exp(−1000) is 0, the sum is 0, and "
                       f"math.log refuses it. A different failure from (b)'s overflow, from "
                       f"the same function"),
        practice.Check("(d) mixed extremes: both failures at once, still exact",
                       abs(mixed["stable"] - 1000.0) < 1e-12,
                       f"[−1000, 1000, 0] -> {mixed['stable']:.9f} — the shift makes the "
                       f"large term 1 and the small one 0, which is the right answer to "
                       f"12 places"),
        practice.Check(f"the stable version matches the closed form on all "
                       f"{len(CASES)} cases",
                       worst < 1e-12,
                       f"worst deviation {worst:.3g}; subtracting max(values) is exact in "
                       f"floating point, so the shift introduces no error of its own"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
