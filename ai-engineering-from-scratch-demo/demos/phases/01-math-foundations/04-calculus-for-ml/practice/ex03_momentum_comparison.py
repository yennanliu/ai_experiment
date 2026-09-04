"""Exercise 3 — add momentum, and compare convergence on f(x) = x⁴ − 3x².

    Add momentum to the gradient descent loop: maintain a velocity vector that
    accumulates past gradients. Compare convergence speed with and without
    momentum on f(x) = x^4 - 3x^2.

Reading of the exercise: "compare convergence speed" needs a definition or the
comparison is a vibe. Speed here is **steps to reach |x| − √1.5 < 1e-6** from a
fixed start at a fixed learning rate, changing only β. The exercise does not say
momentum wins, and on this function it does not: it is 2.3x *slower*. Reporting
that is the answer; checks 3–5 establish why, so the result is a mechanism and
not an anecdote.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "04-calculus-for-ml"
LR, BETA, MAX_STEPS, TARGET = 0.01, 0.9, 4000, 1e-6
MINIMUM = math.sqrt(1.5)                  # f'(x) = 4x³ − 6x = 0 -> x = 0, ±√1.5
CURVATURE = 12.0                          # f''(√1.5) = 12x² − 6


def well(x):
    return x ** 4 - 3 * x ** 2


def descend(ref, x0, beta, lr=LR):
    """beta=0 is plain descent; beta>0 adds a velocity term. Counts reversals."""
    x, velocity, reversals, previous = x0, 0.0, 0, None
    for step in range(1, MAX_STEPS + 1):
        velocity = beta * velocity + ref.numerical_derivative(well, x)
        move = -lr * velocity
        if previous is not None and move * previous < 0:
            reversals += 1
        previous, x = move, x + move
        if abs(abs(x) - MINIMUM) < TARGET:
            return {"steps": step, "reversals": reversals, "x": x}
    return {"steps": None, "reversals": reversals, "x": x}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "derivatives")
    return {
        "plain": descend(ref, 2.5, 0.0),
        "momentum": descend(ref, 2.5, BETA),
        "matched": descend(ref, 2.5, 0.0, lr=LR / (1 - BETA)),
        "critical": [ref.numerical_derivative(well, x) for x in (0.0, MINIMUM, -MINIMUM)],
    }


def verify(result):
    plain, momentum, matched = result["plain"], result["momentum"], result["matched"]
    effective = LR / (1 - BETA)
    return [
        practice.Check("all three variants reach a true minimum at ±√1.5",
                       all(r["steps"] and abs(abs(r["x"]) - MINIMUM) < TARGET
                           for r in (plain, momentum, matched)),
                       f"√1.5 = {MINIMUM:.6f}; f'(0), f'(±√1.5) = "
                       f"{[round(g, 9) for g in result['critical']]} — 0 is the local maximum"),
        practice.Check("FINDING: momentum is slower here, not faster",
                       momentum["steps"] > plain["steps"],
                       f"plain {plain['steps']} steps vs momentum {momentum['steps']} — "
                       f"{momentum['steps'] / plain['steps']:.1f}x slower"),
        practice.Check("because it oscillates: plain descent never reverses, momentum does",
                       plain["reversals"] == 0 and momentum["reversals"] > 10,
                       f"direction changes: plain {plain['reversals']}, "
                       f"momentum {momentum['reversals']} — underdamped around a sharp minimum"),
        practice.Check("…and it overshoots the ridge into the *other* well",
                       plain["x"] > 0 and momentum["x"] < 0,
                       f"both start at x₀=+2.5; plain lands at {plain['x']:+.6f}, "
                       f"momentum at {momentum['x']:+.6f} — accumulated velocity carried it "
                       f"across the local maximum at x=0"),
        practice.Check("the speedup momentum is *meant* to give is just lr/(1−β), taken directly",
                       matched["steps"] < plain["steps"] and matched["steps"] < momentum["steps"],
                       f"plain descent at the matched effective rate {effective:g} takes "
                       f"{matched['steps']} steps — {plain['steps'] / matched['steps']:.0f}x on "
                       f"plain, {momentum['steps'] / matched['steps']:.0f}x on momentum. "
                       f"Stability allows it: 2/f''(√1.5) = {2 / CURVATURE:.3f} > {effective:g}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
