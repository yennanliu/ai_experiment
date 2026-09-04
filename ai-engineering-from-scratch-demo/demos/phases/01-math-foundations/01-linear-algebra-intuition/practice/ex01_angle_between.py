"""Exercise 1 — angle between two vectors, in degrees.

    Implement `Vector.angle_between(other)` that returns the angle in degrees
    between two vectors

Reading of the exercise: the lesson's `code/vectors.py` *already ships*
`angle_between`, so "implement" cannot mean "write it again and check it against
itself" — that would assert nothing. It is read as: write an **independent**
implementation, then use the lesson's as the oracle (D5). The independent one
uses Kahan's formula, `2·atan2(‖â−b̂‖, ‖â+b̂‖)`, rather than `acos(cosθ)`; the two
agree away from the ends and, as check 3 shows, measurably disagree near them,
which is the whole reason the stable form exists.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "01-linear-algebra-intuition"
TOL_DEG = 1e-9

PAIRS = [
    ([1, 0], [0, 1], "perpendicular"),
    ([1, 2, 3], [1, 1, 1], "oblique"),
    ([3, 4], [3, 4], "identical (0°)"),
    ([1, 1, 1], [-1, -1, -1], "antiparallel (180°)"),
    ([2, 0, 0], [5, 0, 0], "parallel, different lengths"),
    ([1, -2, 0.5], [-0.25, 1, 3], "mixed signs"),
]


def _norm(v) -> float:
    return math.sqrt(sum(x * x for x in v))


def _unit(v):
    length = _norm(v)
    if length == 0.0:
        raise ValueError("angle with the zero vector is undefined")
    return [x / length for x in v]


def angle_between(a, b) -> float:
    """Kahan's stable angle: no `acos`, so no cancellation as θ → 0 or π."""
    ua, ub = _unit(a), _unit(b)
    minus = _norm([x - y for x, y in zip(ua, ub)])
    plus = _norm([x + y for x, y in zip(ua, ub)])
    return math.degrees(2.0 * math.atan2(minus, plus))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "vectors")
    rows = []
    for a, b, label in PAIRS:
        theirs = ref.Vector(a).angle_between(ref.Vector(b))
        rows.append((label, angle_between(a, b), theirs))
    # the near-parallel case the exercise does not ask for, but that decides
    # which implementation is the better answer
    tiny = [1.0, 0.0, 0.0]
    near = [1.0, 1e-8, 0.0]
    return {
        "pairs": rows,
        "near": (angle_between(tiny, near), ref.Vector(tiny).angle_between(ref.Vector(near))),
    }


def verify(result):
    rows = result["pairs"]
    worst = max(abs(mine - theirs) for _, mine, theirs in rows)
    checks = [
        practice.Check("all 6 pairs scored", len(rows) == 6, f"{len(rows)} pairs"),
        practice.Check(
            f"matches the lesson's angle_between within {TOL_DEG:g}°",
            worst <= TOL_DEG, f"worst deviation {worst:.3g}°"),
    ]
    named = dict((label, (mine, theirs)) for label, mine, theirs in rows)
    zero = named["identical (0°)"][0]
    flat = named["antiparallel (180°)"][0]
    checks.append(practice.Check("degenerate ends are exact",
                                 abs(zero) < 1e-12 and abs(flat - 180.0) < 1e-12,
                                 f"0° case -> {zero:.3g}, 180° case -> {flat:.12g}"))
    mine, theirs = result["near"]
    exact = math.degrees(1e-8)                       # atan(1e-8) to working precision
    checks.append(practice.Check(
        "near-parallel: Kahan beats acos (why the reimplementation is worth it)",
        abs(mine - exact) < abs(theirs - exact),
        f"exact {exact:.6e}°, Kahan err {abs(mine - exact):.2e}, acos err {abs(theirs - exact):.2e}"))
    return checks


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
