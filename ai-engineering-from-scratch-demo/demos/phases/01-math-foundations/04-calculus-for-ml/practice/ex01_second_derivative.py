"""Exercise 1 — a second derivative built from numerical_derivative, twice.

    Implement `numerical_second_derivative(f, x)` using `numerical_derivative`
    called twice. Verify that the second derivative of x^3 at x=2 is 12.

Reading of the exercise: "called twice" constrains the *method*, not just the
result — differentiate the derivative, rather than use the central
second-difference formula. Answering it as posed exposes something the exercise
does not warn about: at the lesson's own default `h=1e-7`, the nested result is
**11.923796**, not 12. Rounding rescues it, but only to the nearest integer.

The cause is that nesting squares the step error. A central difference has error
O(h²) from truncation and O(ε/h) from cancellation; nest one inside another and
the cancellation term becomes O(ε/h²), which at h=1e-7 is ~1e-2. The optimum
moves from h≈1e-5 to h≈1e-2 — five orders of magnitude — and check 4 shows the
resulting U-curve, including h=1e-9 where the answer is off by 12, i.e. 100%.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "04-calculus-for-ml"
EXACT = 12.0                              # d²/dx²(x³) = 6x, at x=2


def cube(x):
    return x ** 3


def solve():
    ref = parity.load_reference(PHASE, LESSON, "derivatives")

    def numerical_second_derivative(f, x, h=1e-7):
        """As asked: differentiate the derivative."""
        return ref.numerical_derivative(lambda t: ref.numerical_derivative(f, t, h), x, h)

    def direct_second_derivative(f, x, h=1e-4):
        """The central second difference, for comparison only."""
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)

    nested = numerical_second_derivative(cube, 2.0)
    direct = direct_second_derivative(cube, 2.0)
    sweep = {h: numerical_second_derivative(cube, 2.0, h) for h in (1e-2, 1e-4, 1e-7, 1e-9)}
    first = ref.numerical_derivative(cube, 2.0)
    return {"nested": nested, "direct": direct, "sweep": sweep, "first": first}


def verify(result):
    nested_err = abs(result["nested"] - EXACT)
    direct_err = abs(result["direct"] - EXACT)
    best_h, best = min(((h, abs(v - EXACT)) for h, v in result["sweep"].items()),
                       key=lambda row: row[1])
    result["sweep"] = {h: abs(v - EXACT) for h, v in result["sweep"].items()}
    return [
        practice.Check("first derivative of x³ at 2 is 12", abs(result["first"] - EXACT) < 1e-4,
                       f"3x² = 12, got {result['first']:.9f}"),
        practice.Check("second derivative of x³ at 2 is 12, at the lesson's default h",
                       round(result["nested"]) == EXACT,
                       f"got {result['nested']:.6f} — rounds to 12, but only just: "
                       f"error {nested_err:.3g} at h=1e-7"),
        practice.Check("the method is sound; the default step is what costs the digits",
                       best < 1e-9,
                       f"same nested method at h={best_h:g} gives error {best:.3g}, "
                       f"{nested_err / max(best, 1e-18):.0g}x better than at h=1e-7"),
        practice.Check("the direct second difference beats nesting at its own best step",
                       direct_err < nested_err,
                       f"nested {nested_err:.3g} vs direct {direct_err:.3g} "
                       f"({nested_err / max(direct_err, 1e-18):.0f}x)"),
        practice.Check("error is U-shaped in h — too small is as bad as too large",
                       result["sweep"][1e-9] > result["sweep"][1e-2],
                       "error by h: " + ", ".join(
                           f"{h:g}->{v:.2e}"
                           for h, v in sorted(result["sweep"].items(), reverse=True))
                       + "; at h=1e-9 cancellation has destroyed the answer entirely"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
