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
