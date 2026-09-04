"""Exercise 2 — momentum sweep on Rosenbrock: which is fastest, which overshoots.

    **Momentum comparison.** Run SGD with momentum values [0.0, 0.5, 0.9, 0.99]
    on the Rosenbrock function. Track the loss at every step. Which momentum
    value converges fastest? Which overshoots?

Reading of the exercise: two questions, needing two different measurements, and
they do not have the same answer. "Fastest" is steps to cross the lesson's own
`find_convergence_step` threshold. "Overshoots" is positional: does the path go
*past* the minimum at (1,1)? β=0.99 reaches the lowest final loss of the four,
so a loss-only reading would call it the winner — while its trajectory travels to
x=2.04 and its loss peaks above where it started.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "08-optimization"
MOMENTA = [0.0, 0.5, 0.9, 0.99]
START, LR, STEPS = (-1.2, 1.0), 0.001, 5000


def solve():
    ref = parity.load_reference(PHASE, LESSON, "optimizers")
    rows = {}
    for beta in MOMENTA:
        history = ref.optimize(ref.SGDMomentum(lr=LR, momentum=beta), ref.rosenbrock,
                               ref.rosenbrock_gradient, START, steps=STEPS)
        losses = [ref.rosenbrock(p) for p in history]
        rows[beta] = {
            "converged_at": ref.find_convergence_step(history, ref.rosenbrock),
            "loss": losses[-1],
            "rises": sum(1 for a, b in zip(losses, losses[1:]) if b > a),
            "max_x": max(p[0] for p in history),
            "peak_loss": max(losses[5:]),
            "start_loss": losses[0],
        }
    return {"rows": rows}


def verify(result):
    rows = result["rows"]
    fastest = min(MOMENTA, key=lambda b: rows[b]["converged_at"])
    lowest = min(MOMENTA, key=lambda b: rows[b]["loss"])
    overshooting = [b for b in MOMENTA if rows[b]["max_x"] > 1.0]
    return [
        practice.Check(f"all {len(MOMENTA)} momentum values complete {STEPS} steps",
                       all(r["converged_at"] <= STEPS + 1 for r in rows.values()),
                       "steps to threshold: " + ", ".join(
                           f"β={b}→{rows[b]['converged_at']}" for b in MOMENTA)),
        practice.Check("β=0.9 converges fastest", fastest == 0.9,
                       f"β=0.9 crosses the threshold at step {rows[0.9]['converged_at']}, "
                       f"against {rows[0.0]['converged_at']} for plain SGD — "
                       f"{rows[0.0]['converged_at'] / rows[0.9]['converged_at']:.1f}x"),
        practice.Check("β=0.99 overshoots — it travels past the minimum at x=1",
                       overshooting == [0.99],
                       f"max x reached: " + ", ".join(f"β={b}→{rows[b]['max_x']:.4f}"
                                                      for b in MOMENTA)
                       + f"; only β=0.99 exceeds 1, by {rows[0.99]['max_x'] - 1:.4f}"),
        practice.Check("…and its loss climbs above where it started",
                       rows[0.99]["peak_loss"] > rows[0.99]["start_loss"],
                       f"peak loss {rows[0.99]['peak_loss']:.4g} vs start "
                       f"{rows[0.99]['start_loss']:.4g}, with "
                       f"{rows[0.99]['rises']} of {STEPS} steps increasing it — "
                       f"against {rows[0.9]['rises']} for β=0.9"),
        practice.Check("the two questions have different answers",
                       lowest == 0.99 and fastest == 0.9,
                       f"β=0.99 reaches the lowest final loss ({rows[0.99]['loss']:.3g} vs "
                       f"{rows[0.9]['loss']:.3g}) yet is slower to the threshold "
                       f"({rows[0.99]['converged_at']} vs {rows[0.9]['converged_at']}) — "
                       f"'fastest' and 'best' are not the same measurement"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
