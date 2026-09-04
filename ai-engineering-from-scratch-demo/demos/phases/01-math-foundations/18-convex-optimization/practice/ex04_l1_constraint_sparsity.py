"""Exercise 4 — minimize (x−3)²+(y−2)² on |x|+|y| ≤ 1; sparsity from the diamond.

    **Regularization constraint.** Implement L1-constrained optimization:
    minimize (x-3)^2 + (y-2)^2 subject to |x| + |y| <= 1. Show that the solution
    has one coordinate equal to zero (sparsity from the diamond constraint).

Reading of the exercise: the claim is that a coordinate lands exactly at zero,
and that is only true because the target's coordinates differ — the closest point
on the diamond to (3, 2) is the **vertex (1, 0)**, since the target lies outside
and the x-component dominates. Projected gradient descent finds it, and check 4
checks the claim the exercise is really making: the same problem with an L2 ball
constraint produces *no* zero coordinate. Sparsity comes from the diamond's
corners, not from constraining the norm.
"""

from __future__ import annotations

import math

from harness import practice

TARGET = (3.0, 2.0)
BUDGET, STEPS, LR = 1.0, 20_000, 0.01
EXACT_L1 = (1.0, 0.0)


def objective(v):
    return (v[0] - TARGET[0]) ** 2 + (v[1] - TARGET[1]) ** 2


def project_l1(v, budget=BUDGET):
    """Euclidean projection onto {‖v‖₁ ≤ budget} — soft-threshold by θ (Duchi et al.).

    θ is chosen so the shrunk magnitudes sum to exactly `budget`: sort
    descending, and take the largest prefix whose implied θ still lies below the
    next magnitude.
    """
    if sum(abs(c) for c in v) <= budget:
        return list(v)
    magnitudes = sorted((abs(c) for c in v), reverse=True)
    theta, running = 0.0, 0.0
    for i, value in enumerate(magnitudes, 1):
        running += value
        candidate = (running - budget) / i
        if i == len(magnitudes) or candidate < magnitudes[i]:
            theta = candidate
            break
    return [math.copysign(max(abs(c) - theta, 0.0), c) for c in v]


def project_l2(v, budget=BUDGET):
    norm = math.hypot(*v)
    return list(v) if norm <= budget else [c * budget / norm for c in v]


def descend(projection):
    point = [0.0, 0.0]
    for _ in range(STEPS):
        gradient = [2 * (point[0] - TARGET[0]), 2 * (point[1] - TARGET[1])]
        point = projection([p - LR * g for p, g in zip(point, gradient)])
    return point


def solve():
    l1 = descend(project_l1)
    l2 = descend(project_l2)
    vertices = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
    return {"l1": l1, "l2": l2,
            "l1_norm": sum(abs(c) for c in l1), "l2_norm": math.hypot(*l2),
            "l1_value": objective(l1), "l2_value": objective(l2),
            "best_vertex": min(vertices, key=objective),
            "vertex_values": {v: objective(v) for v in vertices},
            "error": math.dist(l1, EXACT_L1)}


def verify(result):
    l1, l2 = result["l1"], result["l2"]
    zeros_l1 = sum(1 for c in l1 if abs(c) < 1e-8)
    zeros_l2 = sum(1 for c in l2 if abs(c) < 1e-8)
    return [
        practice.Check("the L1 solution respects the budget",
                       abs(result["l1_norm"] - BUDGET) < 1e-6,
                       f"‖x‖₁ = {result['l1_norm']:.9f} = {BUDGET:g}, so the constraint is "
                       f"active — the target (3, 2) is far outside the diamond"),
        practice.Check("ANSWER: exactly one coordinate is zero",
                       zeros_l1 == 1,
                       f"solution ({l1[0]:.9f}, {l1[1]:.9f}) — y is exactly 0"),
        practice.Check("…and it is the diamond vertex nearest the target",
                       result["error"] < 1e-6
                       and tuple(result["best_vertex"]) == EXACT_L1,
                       f"converged to {EXACT_L1}, distance {result['error']:.3g}; vertex "
                       f"objectives "
                       + ", ".join(f"{v}: {c:.1f}"
                                   for v, c in result["vertex_values"].items())),
        practice.Check("CAUSE: the same problem on an L2 ball has NO zero coordinate",
                       zeros_l2 == 0,
                       f"L2-constrained solution is ({l2[0]:.6f}, {l2[1]:.6f}) with "
                       f"‖x‖₂ = {result['l2_norm']:.6f} — the sphere has no corners, so the "
                       f"nearest point keeps both coordinates. Sparsity is the diamond's "
                       f"geometry, not the act of constraining a norm"),
        practice.Check("…and the L1 solution pays for that sparsity in objective value",
                       result["l1_value"] > result["l2_value"],
                       f"f = {result['l1_value']:.4f} under L1 against "
                       f"{result['l2_value']:.4f} under L2, at the same budget of 1. The "
                       f"diamond is contained in the ball, so it can only ever do worse — "
                       f"sparsity is bought, not free"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
