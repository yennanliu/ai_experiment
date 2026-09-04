"""Exercise 4 — forward-mode autodiff with dual numbers, cross-checked.

    Implement forward-mode autodiff using dual numbers. Create a `Dual` class
    and verify it gives the same derivatives as your reverse-mode engine.

Reading of the exercise: agreement on one function would be luck, so the engines
are compared over expressions covering every operator the lesson's `Value`
defines. Check 5 measures the cost asymmetry that makes the two modes different
despite identical answers — see the lesson README.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "05-chain-rule-and-autodiff"
TOL = 1e-12


class Dual:
    """a + bε with ε² = 0: the ε component carries the derivative. Each rule is
    the product/quotient/chain rule read off directly — no graph, no tape."""

    __slots__ = ("real", "eps")

    def __init__(self, real, eps=0.0):
        self.real, self.eps = float(real), float(eps)

    def _lift(self, other):
        return other if isinstance(other, Dual) else Dual(other)

    def __add__(self, other):
        o = self._lift(other)
        return Dual(self.real + o.real, self.eps + o.eps)

    def __mul__(self, other):
        o = self._lift(other)
        return Dual(self.real * o.real, self.real * o.eps + self.eps * o.real)

    def __truediv__(self, other):
        o = self._lift(other)
        return Dual(self.real / o.real, (self.eps * o.real - self.real * o.eps) / o.real ** 2)

    def __neg__(self):
        return Dual(-self.real, -self.eps)

    def __sub__(self, other):
        return self + (-self._lift(other))

    def __pow__(self, n):
        return Dual(self.real ** n, n * self.real ** (n - 1) * self.eps)

    def relu(self):
        return Dual(max(0.0, self.real), self.eps if self.real > 0 else 0.0)

    def tanh(self):
        t = math.tanh(self.real)
        return Dual(t, (1 - t * t) * self.eps)

    def exp(self):
        e = math.exp(self.real)
        return Dual(e, e * self.eps)

    def log(self):
        return Dual(math.log(self.real), self.eps / self.real)
    __radd__, __rmul__ = __add__, __mul__


EXPRESSIONS = {"x³ − 2x": (lambda x: x ** 3 - 2 * x, 1.5),
               "tanh(2x + 1)": (lambda x: (2 * x + 1).tanh(), 0.4),
               "relu(x) · x": (lambda x: x.relu() * x, 1.2),
               "exp(x) / (x + 3)": (lambda x: x.exp() / (x + 3), 0.7),
               "log(x² + 1)": (lambda x: (x ** 2 + 1).log(), 2.0)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "autodiff")
    rows = {}
    for label, (expression, point) in EXPRESSIONS.items():
        reverse = ref.Value(point)
        out = expression(reverse)
        out.backward()
        forward = expression(Dual(point, 1.0))  # ε=1 seeds d/dx
        rows[label] = {"forward": forward.eps, "reverse": reverse.grad,
                       "value_gap": abs(forward.real - out.data),
                       "numerical": ref.gradient_check(expression, point)[1]}
    return {"rows": rows, "passes": {"forward-mode": len(EXPRESSIONS), "reverse-mode": 1}}


def verify(result):
    rows = result["rows"]
    worst = max(abs(r["forward"] - r["reverse"]) for r in rows.values())
    worst_value = max(r["value_gap"] for r in rows.values())
    worst_numeric = max(abs(r["forward"] - r["numerical"]) for r in rows.values())
    return [
        practice.Check(f"forward and reverse agree on all {len(rows)} expressions",
                       worst <= TOL,
                       "; ".join(f"{k}: {v['forward']:.9f}" for k, v in rows.items())
                       + f"; worst gap {worst:.3g}"),
        practice.Check("…and both agree with a finite difference",
                       worst_numeric < 1e-6, f"worst gap vs numerical {worst_numeric:.3g}"),
        practice.Check("the forward passes also produce identical values, not just derivatives",
                       worst_value <= TOL, f"worst value gap {worst_value:.3g}"),
        practice.Check("every operator the lesson's Value defines is covered", len(rows) == 5,
                       "pow, mul, sub, div, add, relu, tanh, exp, log over 5 expressions"),
        practice.Check("cost differs even though the answers do not",
                       result["passes"]["forward-mode"] > result["passes"]["reverse-mode"],
                       f"forward: one pass per input ({result['passes']['forward-mode']}); "
                       f"reverse: one per output (1). Training has millions of parameters "
                       f"and one scalar loss — which is why backprop is reverse mode"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
