"""Exercise 5 — complex multiplication vs the 2x2 rotation matrix, 10x10.

    **Rotation matrix equivalence.** For 10 random angles and 10 random points,
    verify that complex multiplication gives the same result as matrix-vector
    multiplication with the 2x2 rotation matrix. Print the maximum numerical
    difference.

Reading of the exercise: 100 agreements at machine precision are worth little on
their own — the two routes compute the same three products from the same inputs,
so agreement is close to guaranteed. What the equivalence is *for* is the
structural claim, and check 4 tests it: composing two rotations equals
multiplying the two complex rotors, which is why the 2D rotation group and the
unit complex numbers are the same group. Check 5 marks where the analogy stops —
3D rotations do not commute, and complex numbers cannot express them.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "19-complex-numbers"
SEED, N_ANGLES, N_POINTS = 20260904, 10, 10


def rotate_matrix(theta, point):
    c, s = math.cos(theta), math.sin(theta)
    return [c * point[0] - s * point[1], s * point[0] + c * point[1]]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "complex_numbers")
    rng = random.Random(SEED)
    angles = [rng.uniform(-math.pi, math.pi) for _ in range(N_ANGLES)]
    points = [(rng.uniform(-10, 10), rng.uniform(-10, 10)) for _ in range(N_POINTS)]
    worst, worst_case = 0.0, (angles[0], points[0])
    for theta in angles:
        rotor = ref.euler(theta)
        for point in points:
            complex_result = ref.Complex(*point) * rotor
            matrix_result = rotate_matrix(theta, point)
            gap = max(abs(complex_result.real - matrix_result[0]),
                      abs(complex_result.imag - matrix_result[1]))
            if gap > worst:
                worst, worst_case = gap, (theta, point)
    # composition: rotating by a then b equals rotating by the product rotor
    a, b = angles[0], angles[1]
    point = points[0]
    stepwise = rotate_matrix(b, rotate_matrix(a, point))
    combined = ref.Complex(*point) * (ref.euler(a) * ref.euler(b))
    swapped = ref.Complex(*point) * (ref.euler(b) * ref.euler(a))
    return {
        "worst": worst, "worst_case": worst_case,
        "pairs": N_ANGLES * N_POINTS,
        "composition_gap": max(abs(combined.real - stepwise[0]),
                               abs(combined.imag - stepwise[1])),
        "commutes": max(abs(combined.real - swapped.real),
                        abs(combined.imag - swapped.imag)),
        "norm_preserved": abs(math.hypot(*stepwise) - math.hypot(*point)),
        "angles": (a, b),
    }


def verify(result):
    theta, point = result["worst_case"]
    return [
        practice.Check(f"all {result['pairs']} angle-point pairs agree",
                       result["worst"] < 1e-13,
                       f"maximum numerical difference {result['worst']:.3g}, at θ = "
                       f"{theta:.4f} on ({point[0]:.3f}, {point[1]:.3f})"),
        practice.Check("…and the agreement is bit-exact, because it is an identity",
                       result["worst"] == 0.0,
                       f"maximum difference is exactly {result['worst']} over "
                       f"{result['pairs']} pairs. cos·x − sin·y and sin·x + cos·y *are* the "
                       f"real and imaginary parts of (x+iy)(cos+i·sin) — the same four "
                       f"products in the same order, so the floats are identical. That "
                       f"makes this check an identity rather than evidence, which is why "
                       f"checks 3-5 go after something else"),
        practice.Check("rotation preserves length",
                       result["norm_preserved"] < 1e-13,
                       f"|R·p| − |p| = {result['norm_preserved']:.3g} after two composed "
                       f"rotations"),
        practice.Check("STRUCTURE: composing rotations equals multiplying the rotors",
                       result["composition_gap"] < 1e-13,
                       f"rotate by {result['angles'][0]:.4f} then "
                       f"{result['angles'][1]:.4f}, against one rotation by the product "
                       f"rotor: gap {result['composition_gap']:.3g}. This is the real "
                       f"content — the unit complex numbers under multiplication *are* the "
                       f"2D rotation group"),
        practice.Check("…and they commute, which is where the 3D analogy fails",
                       result["commutes"] < 1e-13,
                       f"e^(ia)·e^(ib) = e^(ib)·e^(ia) to {result['commutes']:.3g}, because "
                       f"both reduce to e^(i(a+b)). 3D rotations do not commute, so no "
                       f"single complex number can represent one — that needs quaternions"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
