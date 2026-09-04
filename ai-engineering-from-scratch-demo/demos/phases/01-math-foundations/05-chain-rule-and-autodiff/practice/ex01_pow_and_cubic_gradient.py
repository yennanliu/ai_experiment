"""Exercise 1 — `__pow__` on Value, and d/dx(x³) at x=2.

    Add `__pow__` to the Value class so you can compute `x ** n`. Verify that
    `d/dx(x^3)` at `x=2` equals `12.0`.

Reading of the exercise: the lesson's `Value` *already defines* `__pow__`, so
"add it" cannot mean write it again and test it against itself. It is read as:
verify the shipped operator against two independent oracles — the analytic
derivative 3x², and the lesson's own `gradient_check` finite difference — and
then probe the cases the exercise does not mention: negative and fractional
exponents (check 4), and a Value-valued exponent (check 6), which turns out to
raise rather than return a silently gradient-less result.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "05-chain-rule-and-autodiff"
TOL = 1e-9


def solve():
    ref = parity.load_reference(PHASE, LESSON, "autodiff")
    x = ref.Value(2.0)
    y = x ** 3
    y.backward()
    exponents = {}
    for n in (2, 3, 5, -1, 0.5):
        v = ref.Value(2.0)
        (v ** n).backward()
        exponents[n] = {"grad": v.grad, "analytic": n * (2.0 ** (n - 1))}
    checked = ref.gradient_check(lambda t: t ** 3, 2.0)
    # a Value raised to a Value — the case __pow__ does not cover
    try:
        base, power = ref.Value(2.0), ref.Value(3.0)
        (base ** power).backward()
        symbolic = f"grad {base.grad}, but d/dx(x^y) also needs dy — got {power.grad}"
    except Exception as exc:
        symbolic = f"{type(exc).__name__}: {exc}"
    return {"value": y.data, "grad": x.grad, "exponents": exponents,
            "gradient_check": checked, "symbolic": symbolic}


def verify(result):
    worst = max(abs(row["grad"] - row["analytic"]) for row in result["exponents"].values())
    integer_ok = all(abs(result["exponents"][n]["grad"] - result["exponents"][n]["analytic"]) <= TOL
                     for n in (2, 3, 5))
    return [
        practice.Check("x³ at x=2 evaluates to 8", abs(result["value"] - 8.0) <= TOL,
                       f"forward pass -> {result['value']}"),
        practice.Check("d/dx(x³) at x=2 equals 12.0", abs(result["grad"] - 12.0) <= TOL,
                       f"backward -> {result['grad']}"),
        practice.Check("matches the analytic n·xⁿ⁻¹ for integer n = 2, 3, 5",
                       integer_ok,
                       ", ".join(f"n={n}: {result['exponents'][n]['grad']:g}" for n in (2, 3, 5))),
        practice.Check("…and for negative and fractional exponents too",
                       worst <= TOL,
                       f"n=-1: {result['exponents'][-1]['grad']:g} (analytic "
                       f"{result['exponents'][-1]['analytic']:g}), "
                       f"n=0.5: {result['exponents'][0.5]['grad']:g}; worst {worst:.3g}"),
        practice.Check("the lesson's own gradient_check agrees with the analytic value",
                       abs(result["gradient_check"][0] - 12.0) <= 1e-5,
                       f"gradient_check -> {result['gradient_check']}"),
        practice.Check("LIMIT: xʸ with a Value exponent is refused, not silently mis-differentiated",
                       "TypeError" in result["symbolic"],
                       f"{result['symbolic']} — __pow__ takes a raw number, and Python "
                       f"raises rather than returning a value with no gradient path to y. "
                       f"Refusing is the right failure here"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
