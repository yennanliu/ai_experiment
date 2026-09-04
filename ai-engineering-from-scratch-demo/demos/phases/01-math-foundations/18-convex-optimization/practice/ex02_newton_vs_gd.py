"""Exercise 2 — Newton vs gradient descent on 50x² + y², and the κ effect.

    **Newton vs gradient descent race.** Run both methods on f(x,y) = 50*x^2 +
    y^2 from the starting point (10, 10). How many steps does each need to reach
    loss < 1e-10? What happens to gradient descent when the condition number
    (ratio of largest to smallest Hessian eigenvalue) increases?

Reading of the exercise: Newton's count is not a race result, it is a
consequence — on a quadratic the Newton step lands on the exact minimum, so the
answer is **1 step**, at any condition number (check 3). Gradient descent's count
grows like κ, and the second question is answered by sweeping κ rather than
asserting the trend (check 4). The learning rate has to be tied to κ or the
sweep measures divergence instead: lr = 1/λ_max throughout.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "18-convex-optimization"
START, TARGET, MAX_STEPS = (10.0, 10.0), 1e-10, 200_000
KAPPAS = (1, 10, 50, 500)


def make(kappa):
    """f = κx² + y²: Hessian diag(2κ, 2), condition number κ."""
    def f(v):
        return kappa * v[0] ** 2 + v[1] ** 2

    def grad(v):
        return [2 * kappa * v[0], 2 * v[1]]

    def hessian(v):
        return [[2.0 * kappa, 0.0], [0.0, 2.0]]

    return f, grad, hessian


def count_to_target(f, history):
    for step, point in enumerate(history):
        if f(point) < TARGET:
            return step
    return None


def solve():
    ref = parity.load_reference(PHASE, LESSON, "convex")
    rows = {}
    for kappa in KAPPAS:
        f, grad, hessian = make(kappa)
        lr = 1.0 / (2.0 * kappa)                      # 1/λ_max, the stable choice
        # both return the history directly, not (x, history)
        gd_history = ref.optimize_gd(grad, START, lr=lr, steps=MAX_STEPS, tol=0.0)
        newton_history = ref.newtons_method(grad, hessian, START, steps=50, tol=0.0)
        rows[kappa] = {"gd": count_to_target(f, gd_history),
                       "newton": count_to_target(f, newton_history),
                       "lr": lr,
                       "newton_first": list(newton_history[1])}
    return {"rows": rows}


def verify(result):
    rows = result["rows"]
    asked = rows[50]
    gd_counts = [rows[k]["gd"] for k in KAPPAS]
    # κ=1 is degenerate — GD converges in one step there, so the growth ratio is
    # measured across the non-trivial end of the sweep
    growth = gd_counts[-1] / gd_counts[1]
    kappa_growth = KAPPAS[-1] / KAPPAS[1]
    return [
        practice.Check("ANSWER: on f = 50x² + y² from (10,10), GD needs "
                       f"{asked['gd']} steps and Newton needs {asked['newton']}",
                       asked["newton"] == 1 and asked["gd"] > 100,
                       f"loss < {TARGET:g} after {asked['gd']} GD steps at lr = "
                       f"{asked['lr']:g}, against {asked['newton']} Newton step"),
        practice.Check("Newton lands exactly on the minimum in one step",
                       all(abs(v) < 1e-12 for v in asked["newton_first"]),
                       f"first Newton iterate is {asked['newton_first']} — on a quadratic "
                       f"the Hessian is exact, so −H⁻¹∇f *is* the vector to the minimum"),
        practice.Check("…and that holds at every condition number",
                       all(rows[k]["newton"] == 1 for k in KAPPAS),
                       f"Newton steps: {[rows[k]['newton'] for k in KAPPAS]} for κ = "
                       f"{list(KAPPAS)} — conditioning is what a Hessian corrects for, so "
                       f"Newton is blind to it here"),
        practice.Check("ANSWER: gradient descent's step count grows with κ",
                       gd_counts[0] < gd_counts[1] < gd_counts[2] < gd_counts[3],
                       ", ".join(f"κ={k}: {rows[k]['gd']} steps" for k in KAPPAS)
                       + f" — κ=1 is degenerate (a perfectly round bowl, one step), and "
                         f"across the rest a {kappa_growth:.0f}x rise in κ costs "
                         f"{growth:.0f}x the steps"),
        practice.Check("…linearly, as the (κ−1)/(κ+1) contraction predicts",
                       0.8 < growth / kappa_growth < 1.25,
                       f"steps scale {growth:.1f}x against κ scaling "
                       f"{kappa_growth:.0f}x — within "
                       f"{abs(growth / kappa_growth - 1):.0%}. Each GD step shrinks the error by "
                       f"(κ−1)/(κ+1), which tends to 1 as κ grows, so the step count is "
                       f"O(κ·log(1/ε)) — the same reason ill-conditioned training needs "
                       f"either a preconditioner or an adaptive optimiser"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
