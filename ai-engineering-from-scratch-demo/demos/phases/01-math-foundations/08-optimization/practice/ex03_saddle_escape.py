"""Exercise 3 — who escapes the saddle of f(x,y) = x² − y², from (0.01, 0.01)?

    **Saddle point escape.** Define the function `f(x, y) = x^2 - y^2` (a saddle
    point at the origin). Start at (0.01, 0.01). Compare how vanilla GD, SGD
    with momentum, and Adam behave. Which escapes the saddle point?

Reading of the exercise: "which escapes" has to be split, because the ranking
flips depending on when you look. All three escape — the start is off the saddle
in y, so any descent direction grows. Leaving the *flat neighbourhood* is one
measurement and travelling far is another:

  - to |y| > 1: momentum 49 steps, **Adam 76**, plain GD 233.
  - after 2000 steps: GD and momentum are at |y| ~ 5e14, Adam at 30.

So Adam is 3x faster than plain GD out of the flat region — the property it is
recommended for — and then far slower once the gradient is large. Both follow
from one mechanism: dividing by the RMS gradient makes each step ~lr regardless
of slope, which is large when the gradient is tiny and small when it is huge.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "08-optimization"
START, LR, STEPS = (0.01, 0.01), 0.01, 2000
BLOWUP = 1e15                             # optimize() aborts here


def saddle(params):
    return params[0] ** 2 - params[1] ** 2


def saddle_gradient(params):
    return [2 * params[0], -2 * params[1]]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "optimizers")
    optimizers = {
        "vanilla GD": ref.GradientDescent(lr=LR),
        "SGD + momentum": ref.SGDMomentum(lr=LR, momentum=0.9),
        "Adam": ref.Adam(lr=LR),
    }
    rows = {}
    for label, optimizer in optimizers.items():
        history = ref.optimize(optimizer, saddle, saddle_gradient, START, steps=STEPS)
        final = history[-1]
        escaped = next((i for i, p in enumerate(history) if abs(p[1]) > 1.0), None)
        rows[label] = {"steps": len(history) - 1, "final": final,
                       "abs_y": abs(final[1]), "abs_x": abs(final[0]),
                       "escape_step": escaped, "aborted": len(history) - 1 < STEPS}
    return {"rows": rows}


def verify(result):
    rows = result["rows"]
    fastest = min(rows, key=lambda k: rows[k]["escape_step"])
    slowest = max(rows, key=lambda k: rows[k]["escape_step"])
    adam = rows["Adam"]
    return [
        practice.Check("all three escape — |y| grows without bound in every case",
                       all(r["abs_y"] > 1.0 for r in rows.values()),
                       ", ".join(f"{k}: |y|={v['abs_y']:.3g}" for k, v in rows.items())),
        practice.Check("all three drive x to ~0 — the stable direction of the saddle",
                       all(r["abs_x"] < 1e-10 for r in rows.values()),
                       ", ".join(f"{k}: |x|={v['abs_x']:.2g}" for k, v in rows.items())
                       + " — x is a minimum of f and y a maximum, so escape is along y"),
        practice.Check("momentum leaves the flat neighbourhood first",
                       fastest == "SGD + momentum",
                       ", ".join(f"{k}: |y|>1 at step {v['escape_step']}"
                                 for k, v in rows.items())),
        practice.Check("Adam beats plain GD out of the flat region, 3x",
                       adam["escape_step"] < rows["vanilla GD"]["escape_step"] / 2,
                       f"Adam {adam['escape_step']} steps vs GD "
                       f"{rows['vanilla GD']['escape_step']} — near a saddle the gradient is "
                       f"tiny and Adam's normalisation ignores that, which is the property "
                       f"it is recommended for"),
        practice.Check("FINDING: …and then far slower, because that same cap still applies",
                       adam["abs_y"] < rows["vanilla GD"]["abs_y"] / 1e6,
                       f"after {STEPS} steps Adam is at |y| = {adam['abs_y']:.4g} while GD and "
                       f"momentum hit the {BLOWUP:g} abort guard early "
                       f"({rows['vanilla GD']['steps']} and "
                       f"{rows['SGD + momentum']['steps']} steps)"),
        practice.Check("…because Adam's normalisation makes escape linear, not geometric",
                       abs(adam["abs_y"] - STEPS * LR) < 0.7 * STEPS * LR,
                       f"|y| ≈ {adam['abs_y']:.2f} against steps×lr = {STEPS * LR:g}: "
                       f"dividing by √v̂ caps each step near lr however steep the slope, "
                       f"so Adam walks away from a saddle while GD accelerates away from it"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
