"""Exercise 2 — tanh as an activation, and its derivative at 0 and 2.

    Add `tanh` as an activation function. Verify that `tanh'(0) = 1` and
    `tanh'(2) = 0.0707` (approx).

Reading of the exercise: `tanh` already exists on the lesson's `Value`, so as in
exercise 1 this verifies the shipped implementation against independent oracles
rather than reimplementing it. The two stated numbers are checked at the
precision they are stated to — `tanh'(0) = 1` exactly, `tanh'(2) ≈ 0.0707` to 4
decimals — and check 4 adds the saturation behaviour that makes tanh worth
knowing about at all.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "05-chain-rule-and-autodiff"
EXPECTED_AT_2 = 0.0707


def _grad_at(ref, x):
    v = ref.Value(x)
    v.tanh().backward()
    return v.grad


def solve():
    ref = parity.load_reference(PHASE, LESSON, "autodiff")
    grads = {x: _grad_at(ref, x) for x in (0.0, 2.0, 5.0, 10.0, 20.0)}
    analytic = {x: 1 - math.tanh(x) ** 2 for x in grads}
    checked = ref.gradient_check(lambda t: t.tanh(), 2.0)
    return {"grads": grads, "analytic": analytic, "gradient_check": checked,
            "value_at_2": math.tanh(2.0)}


def verify(result):
    grads, analytic = result["grads"], result["analytic"]
    worst = max(abs(grads[x] - analytic[x]) for x in grads)
    return [
        practice.Check("tanh'(0) = 1 exactly", grads[0.0] == 1.0,
                       f"got {grads[0.0]!r} — at 0, tanh is steepest and 1 − tanh²(0) = 1"),
        practice.Check("tanh'(2) ≈ 0.0707 to 4 decimals",
                       abs(grads[2.0] - EXPECTED_AT_2) < 5e-5,
                       f"got {grads[2.0]:.7f}, exercise says {EXPECTED_AT_2}"),
        practice.Check("matches the analytic 1 − tanh²(x) everywhere tested",
                       worst < 1e-15,
                       f"worst deviation {worst:.3g} over x ∈ {sorted(grads)}"),
        practice.Check("the gradient saturates — this is why deep tanh stacks stop learning",
                       grads[10.0] < 1e-8 and grads[20.0] == 0.0,
                       f"tanh'(5) {grads[5.0]:.3e}, tanh'(10) {grads[10.0]:.3e}, "
                       f"tanh'(20) {grads[20.0]:.3e} — underflows to exactly 0 by x=20, "
                       f"so a gradient through it is not small, it is gone"),
        practice.Check("the lesson's finite-difference check agrees at x=2",
                       abs(result["gradient_check"][0] - result["gradient_check"][1]) < 1e-6,
                       f"autodiff {result['gradient_check'][0]:.9f} vs numerical "
                       f"{result['gradient_check'][1]:.9f}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
