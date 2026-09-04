"""Exercise 1 — rotate, scale and shear a unit square; check rotation is rigid.

    Apply rotation, scaling, and shearing to a unit square (corners at [0,0],
    [1,0], [1,1], [0,1]). Print the transformed corners for each. Verify that
    rotation preserves distances between corners.

Reading of the exercise: "verify that rotation preserves distances" is the only
assertable clause, and it is stronger than it looks — preserving all 6 pairwise
distances between 4 points is exactly what makes a map an isometry. Scaling and
shearing are included as controls: a check that only ever sees rotation cannot
tell "distances are preserved" from "my distance function is broken". The
scaling factors are picked so that all three maps have determinant 1, which
makes check 6's point: preserving area is not preserving shape.
"""

from __future__ import annotations

import itertools
import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "03-matrix-transformations"
TOL = 1e-12
SQUARE = [[0, 0], [1, 0], [1, 1], [0, 1]]
THETA = math.pi / 6


def distances(points):
    return [math.dist(a, b) for a, b in itertools.combinations(points, 2)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "transformations")
    maps = {
        "rotation 30°": ref.rotation_2d(THETA),
        "scaling (2, 0.5)": ref.scaling_2d(2, 0.5),
        "shearing (0.5, 0)": ref.shearing_2d(0.5, 0),
    }
    before = distances(SQUARE)
    out = {}
    for label, matrix in maps.items():
        corners = [ref.mat_vec_mul(matrix, p) for p in SQUARE]
        after = distances(corners)
        out[label] = {"corners": corners, "worst_change": max(
            abs(a - b) for a, b in zip(before, after)), "det": ref.det_2x2(matrix)}
    return {"before": before, "maps": out}


def verify(result):
    rotation = result["maps"]["rotation 30°"]
    scaling = result["maps"]["scaling (2, 0.5)"]
    shearing = result["maps"]["shearing (0.5, 0)"]
    checks = [
        practice.Check("all 4 corners transformed by all 3 maps",
                       all(len(m["corners"]) == 4 for m in result["maps"].values()),
                       "; ".join(f"{k}: {[[round(c, 4) for c in p] for p in v['corners']]}"
                                 for k, v in result["maps"].items())),
        practice.Check("rotation preserves all 6 pairwise distances",
                       rotation["worst_change"] <= TOL,
                       f"worst change {rotation['worst_change']:.3g} (tol {TOL:g})"),
        practice.Check("…and preserves area: det == 1",
                       abs(rotation["det"] - 1.0) <= TOL, f"det {rotation['det']:.12g}"),
        practice.Check("control: scaling does NOT preserve distances",
                       scaling["worst_change"] > 0.1,
                       f"worst change {scaling['worst_change']:.4g}"),
        practice.Check("control: shearing does not either",
                       shearing["worst_change"] > 0.1,
                       f"worst change {shearing['worst_change']:.4g}"),
        practice.Check("all three have det == 1, so area alone cannot tell them apart",
                       all(abs(m["det"] - 1.0) <= TOL for m in result["maps"].values()),
                       "rotation, scaling(2, 0.5) and shearing(0.5, 0) all preserve area; "
                       "only rotation preserves distance — that is the difference between "
                       "det == 1 and being an isometry"),
    ]
    return checks


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
