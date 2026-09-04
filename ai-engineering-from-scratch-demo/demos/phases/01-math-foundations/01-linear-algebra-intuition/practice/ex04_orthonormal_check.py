"""Exercise 4 — is the Gram-Schmidt output really orthonormal?

    Verify that the Gram-Schmidt output is truly orthonormal: check that every
    pair has dot product 0 and every vector has magnitude 1

Reading of the exercise: "every pair has dot product 0" cannot be taken
literally in floating point — nothing is exactly 0. So the answer reports the
**worst measured deviation** against a stated tolerance (D5) instead of printing
"orthonormal ✓", and it checks the property where it is actually at risk: on an
ill-conditioned basis, where classical Gram-Schmidt is known to lose
orthogonality. Checks 7 and 8 are the two things a run surfaced that reading the
code did not.
"""

from __future__ import annotations

import fractions
import itertools

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "01-linear-algebra-intuition"
TOL = 1e-12

BASES = {
    "canonical-ish": [[1, 1, 0], [1, 0, 1], [0, 1, 1]],
    "skewed": [[3, 1, 1], [1, 4, 0], [2, 2, 5]],
    # a Hilbert-like basis: independent, but severely ill-conditioned
    "ill-conditioned": [[1, 0.5, 1 / 3], [0.5, 1 / 3, 0.25], [1 / 3, 0.25, 0.2]],
    # Läuchli: independent in exact arithmetic, but the residuals fall under the
    # lesson's own 1e-10 discard threshold
    "lauchli": [[1, 1, 1], [1e-10, 0, 0], [0, 1e-10, 0]],
}


def orthonormality(vectors):
    """Worst |dot| over distinct pairs, and worst |‖v‖ − 1| over the set."""
    worst_dot = max((abs(a.dot(b)) for a, b in itertools.combinations(vectors, 2)),
                    default=0.0)
    worst_norm = max((abs(v.magnitude() - 1.0) for v in vectors), default=0.0)
    return worst_dot, worst_norm


def _det3(rows):
    """Exact 3x3 determinant over Fractions — no rounding, so 'independent in
    exact arithmetic' is a fact here rather than another float comparison."""
    m = [[fractions.Fraction(x).limit_denominator(10 ** 15) for x in row] for row in rows]
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "vectors")
    out = {}
    for label, rows in BASES.items():
        basis = ref.gram_schmidt([ref.Vector(r) for r in rows])
        worst_dot, worst_norm = orthonormality(basis)
        out[label] = {"n_in": len(rows), "n_out": len(basis),
                      "worst_dot": worst_dot, "worst_norm": worst_norm,
                      "exact_det": _det3(rows),
                      "says_independent": ref.is_independent([ref.Vector(r) for r in rows])}
    return out


def verify(result):
    checks = []
    for label in ("canonical-ish", "skewed"):
        row = result[label]
        checks.append(practice.Check(
            f"{label}: every pair orthogonal", row["worst_dot"] <= TOL,
            f"worst |dot| {row['worst_dot']:.3g} (tol {TOL:g})"))
        checks.append(practice.Check(
            f"{label}: every vector is unit length", row["worst_norm"] <= TOL,
            f"worst |‖v‖−1| {row['worst_norm']:.3g} (tol {TOL:g})"))
        checks.append(practice.Check(
            f"{label}: 3 vectors in, 3 out", row["n_out"] == 3, f"{row['n_out']} returned"))
    ill, well = result["ill-conditioned"], result["skewed"]
    checks.append(practice.Check(
        "ill-conditioned basis still passes the 1e-12 bar",
        ill["worst_dot"] <= TOL, f"worst |dot| {ill['worst_dot']:.3g}"))
    checks.append(practice.Check(
        "…but orthogonality is measurably worse there — classical G-S loses it first",
        ill["worst_dot"] > well["worst_dot"] * 10,
        f"{ill['worst_dot']:.3g} vs {well['worst_dot']:.3g}, "
        f"{ill['worst_dot'] / well['worst_dot']:.0f}x worse"))
    lauchli = result["lauchli"]
    checks.append(practice.Check(
        "FINDING: gram_schmidt returns fewer vectors than it was given, and raises nothing",
        lauchli["n_out"] < lauchli["n_in"],
        f"{lauchli['n_in']} vectors in, {lauchli['n_out']} out, no error and no warning"))
    checks.append(practice.Check(
        "…on vectors that are independent in exact arithmetic",
        lauchli["exact_det"] != 0 and not lauchli["says_independent"],
        f"exact det = {lauchli['exact_det']} ≠ 0, but is_independent() says False — both "
        f"functions share the 1e-10 threshold, so they agree with each other, not with algebra"))
    checks.append(practice.Check(
        "…which makes 'every pair is orthogonal' vacuous for the survivors",
        lauchli["worst_dot"] == 0.0 and lauchli["n_out"] == 1,
        "a 1-vector basis has no pairs at all — which is why check 5 has to count them too"))
    return checks


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
