"""Exercise 1 — learning-rate sweep on Rosenbrock; find the largest that works.

    **Learning rate sweep.** Run vanilla gradient descent on the Rosenbrock
    function with learning rates [0.0001, 0.0005, 0.001, 0.005, 0.01]. Plot or
    print the final loss after 5000 steps for each. Find the largest learning
    rate that still converges.

Reading of the exercise: "still converges" needs a definition. Diverging here is
not "ends with a high loss" — the lesson's `optimize` *aborts* on overflow, so a
diverged run returns a short history and its loss is 1e35. Converged is therefore
defined as: completed all 5000 steps, and final loss below the starting loss.

The exercise asks for the largest converging rate, which is 0.005. Worth noticing
that this is *not* the best one: 0.001 reaches loss 0.0038 while 0.005 reaches
0.79, 200x worse, despite both "converging". The loss is U-shaped in the learning
rate, so the largest safe rate and the best rate are different questions, and the
exercise only asks the first. Check 4 records the gap.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "08-optimization"
RATES = [0.0001, 0.0005, 0.001, 0.005, 0.01]
START, STEPS = (-1.2, 1.0), 5000


def solve():
    ref = parity.load_reference(PHASE, LESSON, "optimizers")
    start_loss = ref.rosenbrock(list(START))
    rows = {}
    for rate in RATES:
        history = ref.optimize(ref.GradientDescent(lr=rate), ref.rosenbrock,
                               ref.rosenbrock_gradient, START, steps=STEPS)
        completed = len(history) == STEPS + 1
        final = ref.rosenbrock(history[-1]) if completed else float("inf")
        rows[rate] = {"steps_run": len(history) - 1, "completed": completed,
                      "loss": final, "params": history[-1],
                      "distance": ref.distance_to_minimum(history[-1]),
                      "converged": completed and final < start_loss}
    largest = max((r for r in RATES if rows[r]["converged"]), default=None)
    return {"rows": rows, "largest": largest, "start_loss": start_loss}


def verify(result):
    rows = result["rows"]
    diverged = [r for r in RATES if not rows[r]["completed"]]
    finished = [r for r in RATES if rows[r]["completed"]]
    best = min(finished, key=lambda r: rows[r]["loss"])
    return [
        practice.Check(f"all {len(RATES)} rates run", len(rows) == len(RATES),
                       "final loss: " + ", ".join(
                           f"{r:g}→{rows[r]['loss']:.4g}" if rows[r]["completed"]
                           else f"{r:g}→diverged@{rows[r]['steps_run']}" for r in RATES)),
        practice.Check("the largest converging rate is 0.005",
                       result["largest"] == 0.005,
                       f"lr={result['largest']:g} reaches loss "
                       f"{rows[result['largest']]['loss']:.4g}, distance "
                       f"{rows[result['largest']]['distance']:.4g} from (1,1)"),
        practice.Check("the next rate up diverges, and aborts rather than finishing",
                       0.01 in diverged,
                       f"lr=0.01 stopped after {rows[0.01]['steps_run']} of {STEPS} steps — "
                       f"optimize() breaks on overflow, so a diverged run is detectable by "
                       f"its history length, not by its loss"),
        practice.Check("FINDING: the largest converging rate is not the best rate",
                       best != result["largest"]
                       and rows[best]["loss"] < rows[result["largest"]]["loss"] / 10,
                       f"lr={best:g} reaches {rows[best]['loss']:.4g} while the largest "
                       f"converging lr={result['largest']:g} reaches "
                       f"{rows[result['largest']]['loss']:.4g} — "
                       f"{rows[result['largest']]['loss'] / rows[best]['loss']:.0f}x worse, "
                       f"yet both satisfy 'still converges'"),
        practice.Check("loss is U-shaped in the learning rate, not monotone",
                       rows[RATES[0]]["loss"] > rows[best]["loss"]
                       and rows[best]["loss"] < rows[finished[-1]]["loss"],
                       "loss by lr: " + ", ".join(
                           f"{r:g}→{rows[r]['loss']:.3g}" for r in finished)
                       + f"; too small underfits in 5000 steps, too large oscillates in the "
                         f"Rosenbrock valley, and the cliff is just past {finished[-1]:g}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
