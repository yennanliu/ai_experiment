"""Exercise 3 — minimize (x−3)²+(y−3)² on x+2y=4; gradients parallel.

    **Lagrange multiplier geometry.** Minimize f(x,y) = (x-3)^2 + (y-3)^2 subject
    to x + 2y = 4. Verify the solution by checking that the gradient of f is
    parallel to the gradient of g at the solution.

Reading of the exercise: the parallel-gradient condition is necessary but not
sufficient — it holds at every stationary point of the Lagrangian, including
maxima. So the solution is also checked against a closed form (check 2) and
against the constraint itself (check 3), and check 5 confirms it is a minimum by
sampling the constraint line rather than assuming it.

By hand: the constrained minimum is the foot of the perpendicular from (3,3) to
x+2y=4. With g = x+2y−4, g(3,3) = 5 and ‖∇g‖² = 5, so the projection subtracts
(5/5)·(1,2) = (1,2), giving **(2, 1)** and λ = −2.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "18-convex-optimization"
CENTRE = (3.0, 3.0)
EXACT = (2.0, 1.0)


def f(v):
    return (v[0] - CENTRE[0]) ** 2 + (v[1] - CENTRE[1]) ** 2


def f_grad(v):
    return [2 * (v[0] - CENTRE[0]), 2 * (v[1] - CENTRE[1])]


def g_val(v):
    return v[0] + 2 * v[1] - 4


def g_grad(v):
    return [1.0, 2.0]


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "convex")
    history = ref.lagrange_solve(f_grad, g_val, g_grad, [0.0, 0.0],
                                 lr=0.05, lr_lambda=0.05, steps=20_000)
    point = list(history[-1][0]) if isinstance(history[-1], (tuple, list)) \
        and isinstance(history[-1][0], (tuple, list)) else list(history[-1])[:2]
    gf, gg = f_grad(point), g_grad(point)
    # sample the constraint line to confirm this is a minimum, not a maximum
    line = []
    for i in range(-40, 41):
        t = i / 4.0
        candidate = [EXACT[0] + 2 * t, EXACT[1] - t]      # direction (2,−1) ⟂ (1,2)
        line.append((t, f(candidate), g_val(candidate)))
    return {"point": point, "f_grad": gf, "g_grad": gg,
            "cross": abs(cross(gf, gg)),
            "ratio": gf[0] / gg[0] if gg[0] else float("nan"),
            "constraint": g_val(point), "value": f(point),
            "error": math.dist(point, EXACT),
            "line_min": min(line, key=lambda row: row[1]),
            "line_on_constraint": max(abs(row[2]) for row in line)}


def verify(result):
    gf = result["f_grad"]
    return [
        practice.Check("∇f is parallel to ∇g at the solution",
                       result["cross"] < 1e-6,
                       f"∇f = {[round(v, 6) for v in gf]}, ∇g = {result['g_grad']}, "
                       f"cross product {result['cross']:.3g}; the ratio is "
                       f"{result['ratio']:.6f} = λ"),
        practice.Check("the solution matches the closed form (2, 1)",
                       result["error"] < 1e-4,
                       f"converged to ({result['point'][0]:.6f}, "
                       f"{result['point'][1]:.6f}), distance {result['error']:.3g} from the "
                       f"perpendicular foot of (3,3) on x+2y=4"),
        practice.Check("…and satisfies the constraint",
                       abs(result["constraint"]) < 1e-5,
                       f"x + 2y − 4 = {result['constraint']:.3g}"),
        practice.Check("λ = −2, matching −g(centre)·2/‖∇g‖²",
                       abs(result["ratio"] + 2.0) < 1e-4,
                       f"measured {result['ratio']:.6f}; ∇f = λ∇g with λ = −2, which is "
                       f"the sign convention for a constraint written g = 0 and a point "
                       f"pulled *toward* the line"),
        practice.Check("PARALLEL IS NOT ENOUGH: sampling the line confirms it is a minimum",
                       abs(result["line_min"][0]) < 1e-9
                       and result["line_on_constraint"] < 1e-9,
                       f"81 points along the constraint direction (2,−1), all satisfying "
                       f"x+2y=4 to {result['line_on_constraint']:.1g}; f is smallest at "
                       f"t={result['line_min'][0]:g} with value {result['line_min'][1]:.4f}. "
                       f"The parallel condition holds at maxima too, so it identifies "
                       f"stationary points and nothing more"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
