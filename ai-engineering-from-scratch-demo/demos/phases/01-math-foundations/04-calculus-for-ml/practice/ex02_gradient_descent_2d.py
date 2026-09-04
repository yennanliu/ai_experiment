"""Exercise 2 — gradient descent on f(x,y) = (x−3)² + (y+1)², from (0,0).

    Use gradient descent to find the minimum of f(x, y) = (x - 3)^2 + (y + 1)^2.
    Start from (0, 0). The answer should converge to (3, -1).

Reading of the exercise: "should converge to (3, −1)" is the claim, so the check
is a distance to the known optimum, not "the loss went down". For this bowl the
exact solution is available, which makes two stronger checks possible: the error
must contract by a constant factor every step (check 3), and that factor must be
the theoretical |1 − 2·lr| (check 4). A run that merely ends near (3, −1) would
pass check 2 even if the descent were doing something else entirely.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "04-calculus-for-ml"
OPTIMUM = (3.0, -1.0)
LR, STEPS = 0.1, 100


def bowl(point):
    x, y = point
    return (x - 3) ** 2 + (y + 1) ** 2


def solve():
    ref = parity.load_reference(PHASE, LESSON, "derivatives")
    final, history = ref.gradient_descent_nd(bowl, [0.0, 0.0], lr=LR, steps=STEPS)
    # the lesson's own history is the trail — no need to re-run the loop here
    trail = [math.dist([0.0, 0.0], OPTIMUM)] + [
        math.dist(point, OPTIMUM) for _, point, _ in history[:11]]
    ratios = [trail[i + 1] / trail[i] for i in range(len(trail) - 1)]
    return {"final": list(final), "distance": math.dist(final, OPTIMUM),
            "trail": trail, "ratios": ratios,
            "grad_at_optimum": ref.numerical_gradient(bowl, list(OPTIMUM))}


def verify(result):
    ratios = result["ratios"]
    theoretical = abs(1 - 2 * LR)
    spread = max(ratios) - min(ratios)
    predicted = result["trail"][0] * theoretical ** STEPS
    return [
        practice.Check("converged to (3, −1)", result["distance"] < 1e-6,
                       f"final ({result['final'][0]:.9f}, {result['final'][1]:.9f}), "
                       f"distance {result['distance']:.3g}"),
        practice.Check("the gradient vanishes there — it is a stationary point",
                       max(abs(g) for g in result["grad_at_optimum"]) < 1e-6,
                       f"∇f(3,−1) = {[round(g, 9) for g in result['grad_at_optimum']]}"),
        practice.Check("error contracts by a constant factor each step (linear convergence)",
                       spread < 1e-6,
                       f"ratios over 11 steps span {spread:.3g}, mean "
                       f"{sum(ratios) / len(ratios):.6f}"),
        practice.Check(f"…and that factor is the theoretical |1 − 2·lr| = {theoretical}",
                       abs(sum(ratios) / len(ratios) - theoretical) < 1e-5,
                       f"measured {sum(ratios) / len(ratios):.6f} vs predicted {theoretical}"),
        practice.Check(f"the whole {STEPS}-step run is predicted in closed form",
                       abs(result["distance"] - predicted) / predicted < 1e-6,
                       f"d0·|1−2·lr|^{STEPS} = {result['trail'][0]:.5f}·{theoretical}^{STEPS} "
                       f"= {predicted:.4e}, measured {result['distance']:.4e}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
