"""Exercise 2 — multiply (1,0) by e^(iπ/6) twelve times; trace a 12-gon.

    **Rotation sequence.** Start with the point (1, 0). Multiply by e^(i*pi/6)
    twelve times. Verify that you return to (1, 0) after 12 multiplications.
    Print the coordinates at each step and confirm they trace a regular 12-gon.

Reading of the exercise: "confirm they trace a regular 12-gon" needs the two
properties that define regularity — every vertex at the same radius, and every
consecutive pair separated by the same angle. Both are measured (checks 3, 4).
"Return to (1,0)" is true only up to floating-point: twelve multiplications
accumulate about 2.6e-16 of error, and check 2 reports the measured drift rather
than asserting exact equality, which would fail.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "19-complex-numbers"
STEPS = 12
ANGLE = math.pi / 6


def solve():
    ref = parity.load_reference(PHASE, LESSON, "complex_numbers")
    rotor = ref.euler(ANGLE)
    point = ref.Complex(1.0, 0.0)
    trail = [(point.real, point.imag)]
    for _ in range(STEPS):
        point = point * rotor
        trail.append((point.real, point.imag))
    radii = [math.hypot(*p) for p in trail]
    gaps = []
    for i in range(len(trail) - 1):
        a = math.atan2(trail[i][1], trail[i][0])
        b = math.atan2(trail[i + 1][1], trail[i + 1][0])
        gaps.append((b - a) % (2 * math.pi))
    vertices = {(round(x, 9), round(y, 9)) for x, y in trail[:STEPS]}
    return {"trail": trail, "radii": radii, "gaps": gaps,
            "drift": math.dist(trail[-1], trail[0]),
            "distinct_vertices": len(vertices),
            "rotor": (rotor.real, rotor.imag),
            "half_turn": trail[STEPS // 2]}


def verify(result):
    trail, radii, gaps = result["trail"], result["radii"], result["gaps"]
    return [
        practice.Check(f"{STEPS} multiplications by e^(iπ/6) recorded",
                       len(trail) == STEPS + 1,
                       "first four: " + ", ".join(f"({x:+.4f}, {y:+.4f})"
                                                  for x, y in trail[:4])
                       + f"; rotor = ({result['rotor'][0]:.6f}, "
                         f"{result['rotor'][1]:.6f}) = cos30° + i·sin30°"),
        practice.Check("it returns to (1, 0) — to within accumulated rounding",
                       result["drift"] < 1e-14,
                       f"final point ({trail[-1][0]:.15f}, {trail[-1][1]:.15f}), drift "
                       f"{result['drift']:.3g} after {STEPS} multiplications. Exact "
                       f"equality would fail: each step rounds"),
        practice.Check("every vertex is at radius 1 — the rotor is a pure rotation",
                       max(abs(r - 1.0) for r in radii) < 1e-15,
                       f"worst |r − 1| = {max(abs(r - 1.0) for r in radii):.3g} over "
                       f"{len(radii)} points"),
        practice.Check("…and every consecutive gap is exactly 30°",
                       max(abs(g - ANGLE) for g in gaps) < 1e-15,
                       f"worst |Δθ − π/6| = {max(abs(g - ANGLE) for g in gaps):.3g}; "
                       f"equal radii plus equal angles is what *regular* means"),
        practice.Check(f"the {STEPS} vertices are distinct, and step 6 is the antipode",
                       result["distinct_vertices"] == STEPS
                       and abs(result["half_turn"][0] + 1.0) < 1e-14,
                       f"{result['distinct_vertices']} distinct vertices, so the path is a "
                       f"12-gon and not a shorter polygon retraced — π/6 generates the full "
                       f"cyclic group of order 12. Step 6 lands at "
                       f"({result['half_turn'][0]:+.6f}, {result['half_turn'][1]:+.6f})"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
