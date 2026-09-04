"""Exercise 4 — the 8th roots of unity: they sum to zero and cycle.

    **Roots of unity visualization.** Compute the 8th roots of unity. Verify that
    they sum to zero. Verify that multiplying any root by the primitive root
    e^(2*pi*i/8) gives the next root.

Reading of the exercise: both claims hold, and both are worth stating precisely.
"Sum to zero" is exact algebraically but only ~1e-16 numerically, so the measured
residual is reported. "Multiplying any root gives the next" needs the wrap-around
case — the last root times the primitive returns the *first* — which check 3
includes deliberately, since a loop over pairs that stops early would miss it and
still pass.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "19-complex-numbers"
N = 8
TOL = 1e-12


def solve():
    ref = parity.load_reference(PHASE, LESSON, "complex_numbers")
    roots = list(ref.roots_of_unity(N))
    primitive = ref.euler(2 * math.pi / N)
    total = ref.Complex(0.0, 0.0)
    for root in roots:
        total = total + root
    cycle_gaps = []
    for i, root in enumerate(roots):
        expected = roots[(i + 1) % N]
        stepped = root * primitive
        cycle_gaps.append(math.hypot(stepped.real - expected.real,
                                     stepped.imag - expected.imag))
    nth_gaps = []
    for root in roots:
        acc = ref.Complex(1.0, 0.0)
        for _ in range(N):
            acc = acc * root
        nth_gaps.append(math.hypot(acc.real - 1.0, acc.imag))
    return {
        "count": len(roots),
        "sum": (total.real, total.imag),
        "sum_residual": math.hypot(total.real, total.imag),
        "radii": [r.magnitude() for r in roots],
        "cycle_worst": max(cycle_gaps),
        "wrap_gap": cycle_gaps[-1],
        "nth_worst": max(nth_gaps),
        "first_two": [(r.real, r.imag) for r in roots[:2]],
        "distinct_phases": len({round(math.atan2(r.imag, r.real) % (2 * math.pi), 9)
                                for r in roots}),
    }


def verify(result):
    return [
        practice.Check(f"all {N} roots computed, each on the unit circle",
                       result["count"] == N
                       and max(abs(r - 1.0) for r in result["radii"]) < TOL,
                       f"worst |r − 1| = {max(abs(r - 1.0) for r in result['radii']):.3g}; "
                       f"first two roots "
                       + ", ".join(f"({x:+.6f}, {y:+.6f})" for x, y in result["first_two"])),
        practice.Check("they sum to zero",
                       result["sum_residual"] < TOL,
                       f"Σ = ({result['sum'][0]:.3g}, {result['sum'][1]:.3g}), residual "
                       f"{result['sum_residual']:.3g} — exactly 0 algebraically, since the "
                       f"roots are the vertices of a regular polygon centred on the origin"),
        practice.Check("multiplying any root by the primitive gives the next — including "
                       "the wrap",
                       result["cycle_worst"] < TOL,
                       f"worst gap over all {N} pairs {result['cycle_worst']:.3g}, of which "
                       f"the wrap-around (last × primitive -> first) is "
                       f"{result['wrap_gap']:.3g}. A loop that stopped at N−1 pairs would "
                       f"miss the only interesting one"),
        practice.Check(f"every root raised to the {N}th power returns 1",
                       result["nth_worst"] < 1e-14,
                       f"worst |ζᴺ − 1| = {result['nth_worst']:.3g} — which is the "
                       f"definition the name 'roots of unity' asserts, and is not implied "
                       f"by either of the two properties the exercise asks for"),
        practice.Check("…and all N are distinct, so the primitive visits each exactly once",
                       result["distinct_phases"] == N,
                       f"{result['distinct_phases']} distinct phases at radius 1. That is "
                       f"what makes e^(2πi/8) *primitive* rather than merely a root: a "
                       f"non-primitive 8th root such as −1 would cycle through 2 of them "
                       f"and never reach the other 6"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
