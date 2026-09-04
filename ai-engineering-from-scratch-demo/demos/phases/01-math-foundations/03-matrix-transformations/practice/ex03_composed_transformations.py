"""Exercise 3 — compose rotate ∘ scale ∘ shear over 8 points on a circle.

    Create a composition of three transformations (rotate 30 degrees, scale by
    [1.5, 0.8], shear with kx=0.3) and apply it to 8 points arranged in a
    circle. Print before and after coordinates. Compute the determinant of the
    composed matrix and verify it equals the product of the individual
    determinants.

Reading of the exercise: det(ABC) == det(A)det(B)det(C) is the assertable claim,
and it holds for *any* composition order — so on its own it would still pass if
the matrices were multiplied in the wrong order. Check 3 pins the order down
separately, by composing the same three maps the other way and showing the
points land somewhere else while the determinant does not move.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "03-matrix-transformations"
TOL = 1e-12
N_POINTS = 8


def circle(n=N_POINTS):
    return [[math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)] for i in range(n)]


def _apply(ref, matrix, points):
    return [ref.mat_vec_mul(matrix, p) for p in points]


def _worst_gap(a, b) -> float:
    return max(abs(x - y) for pa, pb in zip(a, b) for x, y in zip(pa, pb))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "transformations")
    rotate = ref.rotation_2d(math.radians(30))
    scale = ref.scaling_2d(1.5, 0.8)
    shear = ref.shearing_2d(0.3, 0)
    composed = ref.mat_mul(ref.mat_mul(rotate, scale), shear)
    reversed_order = ref.mat_mul(ref.mat_mul(shear, scale), rotate)
    points = circle()
    after = _apply(ref, composed, points)
    # applying the three one at a time must equal applying the product once
    stepwise = _apply(ref, rotate, _apply(ref, scale, _apply(ref, shear, points)))
    return {"points": points, "after": after,
            "det_composed": ref.det_2x2(composed), "det_reversed": ref.det_2x2(reversed_order),
            "parts": [ref.det_2x2(m) for m in (rotate, scale, shear)],
            "stepwise_gap": _worst_gap(after, stepwise),
            "order_gap": _worst_gap(after, _apply(ref, reversed_order, points))}


def verify(result):
    product = result["parts"][0] * result["parts"][1] * result["parts"][2]
    return [
        practice.Check(f"{N_POINTS} circle points transformed",
                       len(result["after"]) == N_POINTS,
                       "before -> after: " + "; ".join(
                           f"({b[0]:.3f},{b[1]:.3f})->({a[0]:.3f},{a[1]:.3f})"
                           for b, a in list(zip(result["points"], result["after"]))[:3]) + " …"),
        practice.Check("det(ABC) == det(A)·det(B)·det(C)",
                       abs(result["det_composed"] - product) <= TOL,
                       f"composed {result['det_composed']:.12g} vs product {product:.12g} "
                       f"(dets {[round(d, 6) for d in result['parts']]})"),
        practice.Check("…but that identity is order-blind, so order is pinned separately",
                       abs(result["det_reversed"] - product) <= TOL
                       and result["order_gap"] > 0.1,
                       f"reversed order has the same det {result['det_reversed']:.6g} "
                       f"yet moves points by up to {result['order_gap']:.4g}"),
        practice.Check("composing once equals applying the three in sequence",
                       result["stepwise_gap"] <= TOL,
                       f"worst gap {result['stepwise_gap']:.3g}"),
        practice.Check("area scales by 1.2 — the circle becomes an ellipse of that area ratio",
                       abs(result["det_composed"] - 1.2) <= 1e-12,
                       f"det {result['det_composed']:.12g}; shear contributes "
                       f"{result['parts'][2]:g}, rotation {result['parts'][0]:g}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
