"""Exercise 1 — five functions through the convexity checker, and why each.

    **Convexity gallery.** Test these functions for convexity using the checker:
    f(x) = x^4, f(x) = sin(x), f(x,y) = x^2 + y^2, f(x,y) = x*y, f(x) = max(x,
    0). Explain why each result makes sense.

Reading of the exercise: the checker samples random chords, so it can only ever
*fail to find* a violation — it is a one-sided test. Four of the five have
verdicts derivable from second derivatives, so those are asserted independently
(check 3) rather than trusted from the sampler. `max(x, 0)` is the interesting
one: it is convex but not differentiable at 0, so the sampler agrees while a
Hessian argument cannot be made, which is exactly why a sampling checker earns
its place.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "18-convex-optimization"
SEED, SAMPLES = 20260904, 4000

CASES = {
    "x^4": (lambda v: v[0] ** 4, 1, True),
    "sin(x)": (lambda v: math.sin(v[0]), 1, False),
    "x^2 + y^2": (lambda v: v[0] ** 2 + v[1] ** 2, 2, True),
    "x*y": (lambda v: v[0] * v[1], 2, False),
    "max(x, 0)": (lambda v: max(v[0], 0.0), 1, True),
}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "convex")
    rows = {}
    for label, (fn, dim, expected) in CASES.items():
        random.seed(SEED)
        result = ref.check_convexity(fn, dim, samples=SAMPLES, label=label)
        rows[label] = {"result": result, "expected": expected}
    # x*y has Hessian [[0,1],[1,0]], eigenvalues ±1 — indefinite, so a saddle
    xy_eigen = ref.hessian_eigenvalues_2d([[0.0, 1.0], [1.0, 0.0]])
    bowl_eigen = ref.hessian_eigenvalues_2d([[2.0, 0.0], [0.0, 2.0]])
    # a chord that violates convexity for sin, exhibited rather than sampled
    a, b = 0.0, 3 * math.pi / 2
    mid = (a + b) / 2
    sin_gap = math.sin(mid) - 0.5 * (math.sin(a) + math.sin(b))
    return {"rows": rows, "xy_eigen": list(xy_eigen), "bowl_eigen": list(bowl_eigen),
            "sin_gap": sin_gap, "sin_chord": (a, b)}


def _verdict(entry):
    """The checker returns either a bool or a (bool, ...) tuple."""
    value = entry["result"]
    return bool(value[0]) if isinstance(value, (tuple, list)) else bool(value)


def verify(result):
    rows = result["rows"]
    verdicts = {k: _verdict(v) for k, v in rows.items()}
    agree = {k: verdicts[k] == rows[k]["expected"] for k in rows}
    return [
        practice.Check(f"all {len(CASES)} functions tested with {SAMPLES:,} random chords",
                       len(verdicts) == len(CASES),
                       ", ".join(f"{k}: {'convex' if v else 'not convex'}"
                                 for k, v in verdicts.items())),
        practice.Check("every verdict matches the analytic answer",
                       all(agree.values()),
                       f"{sum(agree.values())}/{len(agree)} agree — x⁴ (f''=12x²≥0) and "
                       f"x²+y² (Hessian 2I) convex; sin and x·y not"),
        practice.Check("x·y is rejected because its Hessian is indefinite",
                       not verdicts["x*y"]
                       and min(result["xy_eigen"]) < 0 < max(result["xy_eigen"]),
                       f"Hessian [[0,1],[1,0]] has eigenvalues "
                       f"{[round(v, 6) for v in result['xy_eigen']]} — one negative, so a "
                       f"saddle, against {[round(v, 1) for v in result['bowl_eigen']]} for "
                       f"the bowl"),
        practice.Check("…and sin is rejected by an exhibited chord, not just a sampled one",
                       result["sin_gap"] > 0,
                       f"on [0, 3π/2] the midpoint value {math.sin(sum(result['sin_chord']) / 2):.4f} "
                       f"exceeds the chord average "
                       f"{0.5 * (math.sin(0) + math.sin(3 * math.pi / 2)):.4f} by "
                       f"{result['sin_gap']:.4f} — a specific counterexample, since a "
                       f"sampler finding none would prove nothing"),
        practice.Check("max(x, 0) is the case only a sampling checker can settle",
                       verdicts["max(x, 0)"],
                       "convex, but not differentiable at 0 — there is no Hessian to "
                       "examine, so the second-derivative argument used for the other four "
                       "is unavailable. This is what the chord test buys, and also why it "
                       "is one-sided: it can fail to find a violation, never prove absence"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
