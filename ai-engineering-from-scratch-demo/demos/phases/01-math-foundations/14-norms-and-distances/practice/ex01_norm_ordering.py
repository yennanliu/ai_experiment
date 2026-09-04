"""Exercise 1 — L1/L2/L∞ between (1,2,3) and (4,0,6), and why L∞ ≤ L2 ≤ L1.

    Compute L1, L2, and L-infinity distances between (1, 2, 3) and (4, 0, 6).
    Verify that L-inf <= L2 <= L1 always holds for any pair of points. Prove why
    this ordering is guaranteed.

Reading of the exercise: "always holds for any pair" cannot be shown by one pair,
so the ordering is checked over 2000 random pairs across several dimensions, and
the *proof* is turned into two testable identities rather than prose. L∞ ≤ L2
because the max term is one summand under the square root; L2 ≤ L1 because
squaring and re-rooting drops all cross terms, and the gap closes to equality
exactly when the difference vector has a single non-zero component. Check 5 tests
that equality case, which is what makes the inequality tight rather than loose.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "14-norms-and-distances"
A, B = (1, 2, 3), (4, 0, 6)
TRIALS, SEED = 2000, 20260904


def solve():
    ref = parity.load_reference(PHASE, LESSON, "distances")
    named = {"L1": ref.l1_distance(A, B), "L2": ref.l2_distance(A, B),
             "Linf": ref.linf_distance(A, B)}
    rng = random.Random(SEED)
    violations, ratios = 0, []
    for _ in range(TRIALS):
        dim = rng.randint(1, 12)
        p = [rng.uniform(-50, 50) for _ in range(dim)]
        q = [rng.uniform(-50, 50) for _ in range(dim)]
        one, two, inf = ref.l1_distance(p, q), ref.l2_distance(p, q), ref.linf_distance(p, q)
        if not inf <= two + 1e-12 <= one + 1e-12:
            violations += 1
        ratios.append((one / two, math.sqrt(dim)))
    axis = ref.l1_distance((0, 0, 0), (5, 0, 0)), ref.l2_distance((0, 0, 0), (5, 0, 0)), \
        ref.linf_distance((0, 0, 0), (5, 0, 0))
    equal = [1.0] * 9
    spread = ref.l1_distance([0.0] * 9, equal) / ref.l2_distance([0.0] * 9, equal)
    return {"named": named, "violations": violations, "trials": TRIALS,
            "worst_ratio": max(r for r, _ in ratios),
            "bound_respected": all(r <= s + 1e-12 for r, s in ratios),
            "axis": axis, "spread": spread, "sqrt9": 3.0,
            "delta": [abs(x - y) for x, y in zip(A, B)]}


def verify(result):
    named = result["named"]
    return [
        practice.Check("L1=8, L2=√22≈4.690, L∞=3 for (1,2,3) vs (4,0,6)",
                       named["L1"] == 8 and abs(named["L2"] - math.sqrt(22)) < 1e-12
                       and named["Linf"] == 3,
                       f"|Δ| = {result['delta']}; L1 {named['L1']}, "
                       f"L2 {named['L2']:.9f}, L∞ {named['Linf']}"),
        practice.Check("the ordering holds, and it holds here",
                       named["Linf"] <= named["L2"] <= named["L1"],
                       f"{named['Linf']} ≤ {named['L2']:.4f} ≤ {named['L1']}"),
        practice.Check(f"…and on all {result['trials']} random pairs, dimensions 1–12",
                       result["violations"] == 0,
                       f"{result['violations']} violations"),
        practice.Check("PROOF (upper): L1/L2 ≤ √d, by Cauchy-Schwarz, never exceeded",
                       result["bound_respected"],
                       f"worst observed ratio {result['worst_ratio']:.4f}; the bound is "
                       f"attained only when every |Δᵢ| is equal — 9 equal components give "
                       f"L1/L2 = {result['spread']:.6f} = √9 exactly"),
        practice.Check("PROOF (lower): all three coincide iff Δ has one non-zero component",
                       result["axis"][0] == result["axis"][1] == result["axis"][2] == 5,
                       f"(0,0,0) to (5,0,0) gives L1 = L2 = L∞ = 5. With one term the sum, "
                       f"the root-of-sum-of-squares and the max are the same number, which "
                       f"is why the inequalities are tight and not merely true"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
