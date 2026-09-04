"""Exercise 1 — (2+3i)(4−i) and (5+2i)/(1−3i), by hand and in polar form.

    **Complex arithmetic by hand.** Compute (2 + 3i) * (4 - i) and verify with
    the code. Then compute (5 + 2i) / (1 - 3i). Draw both results on the complex
    plane and check that multiplication rotated and scaled the first number.

Reading of the exercise: "draw on the complex plane" is not assertable, but the
claim behind it is — multiplication rotates by the second argument's phase and
scales by its magnitude. Check 4 tests that as an identity on the polar
representation, which is the content the drawing was meant to convey.

By hand: (2+3i)(4−i) = 8 − 2i + 12i − 3i² = 8 + 10i + 3 = **11 + 10i**.
And (5+2i)/(1−3i) = (5+2i)(1+3i)/((1)²+(3)²) = (5 + 15i + 2i − 6)/10
= (−1 + 17i)/10 = **−0.1 + 1.7i**.
"""

from __future__ import annotations

import cmath
import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "19-complex-numbers"
TOL = 1e-12
PRODUCT = (11.0, 10.0)
QUOTIENT = (-0.1, 1.7)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "complex_numbers")
    a, b = ref.Complex(2, 3), ref.Complex(4, -1)
    c, d = ref.Complex(5, 2), ref.Complex(1, -3)
    product, quotient = a * b, c / d
    r_a, theta_a = ref.to_polar(a)
    r_b, theta_b = ref.to_polar(b)
    r_p, theta_p = ref.to_polar(product)
    # python's own complex type, as a second independent oracle
    native_product = complex(2, 3) * complex(4, -1)
    native_quotient = complex(5, 2) / complex(1, -3)
    return {
        "product": (product.real, product.imag),
        "quotient": (quotient.real, quotient.imag),
        "native": ((native_product.real, native_product.imag),
                   (native_quotient.real, native_quotient.imag)),
        "polar": {"a": (r_a, theta_a), "b": (r_b, theta_b), "product": (r_p, theta_p)},
        "magnitude_gap": abs(r_p - r_a * r_b),
        "phase_gap": abs(cmath.phase(cmath.exp(1j * (theta_p - theta_a - theta_b)))),
        "conjugate_identity": abs((d * d.conjugate()).real - d.magnitude() ** 2),
    }


def verify(result):
    return [
        practice.Check("by hand: (2+3i)(4−i) = 11 + 10i",
                       max(abs(a - b) for a, b in zip(result["product"], PRODUCT)) < TOL,
                       f"the lesson's Complex gives {result['product']}, matching the "
                       f"hand expansion 8 + 10i − 3i²"),
        practice.Check("by hand: (5+2i)/(1−3i) = −0.1 + 1.7i",
                       max(abs(a - b) for a, b in zip(result["quotient"], QUOTIENT)) < TOL,
                       f"got {result['quotient']} — multiply by the conjugate (1+3i) over "
                       f"|1−3i|² = 10"),
        practice.Check("Python's built-in complex agrees with both",
                       max(abs(a - b) for pair, truth in
                           zip(result["native"], (PRODUCT, QUOTIENT))
                           for a, b in zip(pair, truth)) < TOL,
                       f"native complex -> {result['native'][0]} and "
                       f"{result['native'][1]}, an oracle written by someone else"),
        practice.Check("ANSWER: multiplication scaled by |b| and rotated by arg(b)",
                       result["magnitude_gap"] < TOL and result["phase_gap"] < TOL,
                       f"|a| = {result['polar']['a'][0]:.4f} × |b| = "
                       f"{result['polar']['b'][0]:.4f} gives "
                       f"{result['polar']['a'][0] * result['polar']['b'][0]:.4f} = "
                       f"|ab| = {result['polar']['product'][0]:.4f}; and arg(a) + arg(b) = "
                       f"{result['polar']['a'][1] + result['polar']['b'][1]:.6f} matches "
                       f"arg(ab) = {result['polar']['product'][1]:.6f} mod 2π"),
        practice.Check("…and z·z̄ = |z|², the identity the division relied on",
                       result["conjugate_identity"] < TOL,
                       f"(1−3i)(1+3i) = 10 = |1−3i|², gap "
                       f"{result['conjugate_identity']:.3g} — which is why dividing by the "
                       f"conjugate turns a complex denominator into a real one"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
