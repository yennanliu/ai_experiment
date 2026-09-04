"""Exercise 5 — a 3x3 matrix of rank 2, and what its columns span.

    Create a 3x3 matrix with rank 2. Verify using the `rank()` method. Then
    explain what geometric object the columns span.

Reading of the exercise: it ends in a question, so a passing `rank() == 2` is
only half an answer. The geometric claim — *a plane through the origin* — is
itself checkable, and check 4 does it the only way that means anything: exhibit
a point that provably cannot be reached by any combination of the columns. The
prose answer is in the module docstring below and in practice/README.md.

Answer: the three columns span a **plane through the origin** in R^3 — a
2-dimensional subspace. Rank 2 means exactly two columns are linearly
independent; the third is a combination of them and so adds no new direction.
The span is closed under scaling and addition, so it must contain the origin: it
is a plane, not an arbitrary flat sheet floating in space. Its normal is the
cross product of any two independent columns, and every vector off that plane —
such as the normal itself — is unreachable.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "01-linear-algebra-intuition"

# c3 = 2*c1 - c2, so the third column adds no direction. Rank is therefore 2.
ROWS = [[1, 0, 2],
        [2, 1, 3],
        [3, 1, 5]]


def columns(rows):
    return [[row[j] for row in rows] for j in range(len(rows[0]))]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "vectors")
    matrix = ref.Matrix(ROWS)
    cols = columns(ROWS)
    c1, c2, c3 = cols
    combination = [2 * x - y for x, y in zip(c1, c2)]
    normal = cross(c1, c2)
    # a point off the plane: the normal itself. Reachable only if some
    # combination of the columns equals it, which would make normal.normal == 0.
    reachable = sum(n * n for n in normal)
    independent_pair = ref.Matrix([[c1[i], c2[i]] for i in range(3)])
    return {"rank": matrix.rank(), "cols": cols, "c3": c3, "combination": combination,
            "normal": normal, "off_plane_residual": reachable,
            "pair_rank": independent_pair.rank(),
            "dots": [sum(n * c for n, c in zip(normal, col)) for col in cols]}


def verify(result):
    return [
        practice.Check("rank() returns 2", result["rank"] == 2, f"got {result['rank']}"),
        practice.Check("the first two columns are independent",
                       result["pair_rank"] == 2, f"rank of [c1 c2] = {result['pair_rank']}"),
        practice.Check("c3 == 2*c1 - c2, so it adds no direction",
                       result["c3"] == result["combination"],
                       f"c3 {result['c3']} == {result['combination']}"),
        practice.Check("every column lies in the plane (normal ⊥ all three)",
                       all(abs(d) < 1e-12 for d in result["dots"]),
                       f"normal·columns = {result['dots']}"),
        practice.Check("a point off the plane is unreachable — so the span is a plane, not R^3",
                       result["off_plane_residual"] > 0,
                       f"normal {result['normal']} has |n|^2 = {result['off_plane_residual']}, "
                       f"but n·c = 0 for every column"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
