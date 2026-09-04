"""Exercise 2 — a 2D scaling matrix, applied to [1, 1].

    Create a 2D scaling matrix that doubles the x-coordinate and triples the
    y-coordinate, then apply it to the vector [1, 1]

Reading of the exercise: the arithmetic is one line, so the answer is not the
number [2, 3] — it is the evidence that the matrix *is* the transformation.
Checks 3 and 4 assert the two properties that make it one: linearity, and that
the determinant is the area scale factor.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "01-linear-algebra-intuition"
SX, SY = 2, 3


def scaling_matrix(ref, sx=SX, sy=SY):
    return ref.Matrix([[sx, 0], [0, sy]])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "vectors")
    S = scaling_matrix(ref)
    applied = S @ ref.Vector([1, 1])
    probes = [([1, 0], [3, 5]), ([2, -1], [0, 4]), ([-1.5, 2.5], [1, 1])]
    linearity = []
    for a, b in probes:
        va, vb = ref.Vector(a), ref.Vector(b)
        combined = (S @ (va + vb)).components
        separate = ((S @ va) + (S @ vb)).components
        linearity.append((a, b, combined, separate))
    unit_square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    image = [(S @ ref.Vector(p)).components for p in unit_square]
    return {"matrix": S, "applied": applied.components,
            "linearity": linearity, "image": image, "ref": ref}


def _shoelace(points) -> float:
    n = len(points)
    return abs(sum(points[i][0] * points[(i + 1) % n][1]
                   - points[(i + 1) % n][0] * points[i][1] for i in range(n))) / 2.0


def verify(result):
    rows = result["matrix"].rows
    off_diagonal = rows[0][1] == 0 and rows[1][0] == 0
    worst_linear = max(
        max(abs(c - s) for c, s in zip(combined, separate))
        for _, _, combined, separate in result["linearity"])
    area = _shoelace(result["image"])
    checks = [
        practice.Check("S @ [1,1] == [2,3]", result["applied"] == [2, 3],
                       f"got {result['applied']}"),
        practice.Check("S is diagonal — axes scale independently",
                       off_diagonal and rows[0][0] == SX and rows[1][1] == SY,
                       f"rows {rows}"),
        practice.Check("the map is linear: S(a+b) == Sa + Sb",
                       worst_linear < 1e-12,
                       f"worst deviation {worst_linear:.3g} over 3 probes"),
        practice.Check("det(S) == 6 is the area scale factor",
                       abs(area - SX * SY) < 1e-12,
                       f"unit square -> area {area:g}, det {SX * SY}"),
    ]
    numpy = parity.try_numpy()
    if numpy is not None:
        theirs = numpy.diag([SX, SY]) @ numpy.array([1, 1])
        checks.append(practice.Check("matches numpy", list(theirs) == [2, 3],
                                     f"numpy -> {list(theirs)}"))
    return checks


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
