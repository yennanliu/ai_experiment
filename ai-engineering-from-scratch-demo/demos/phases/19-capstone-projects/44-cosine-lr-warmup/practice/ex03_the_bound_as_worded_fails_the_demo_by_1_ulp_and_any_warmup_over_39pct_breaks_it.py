"""Exercise 3 -- the bound as worded fails the lesson's own demo by 1 ulp, and any warmup over 39% breaks it.

    Add a unit test that the schedule is continuous: for every step in
    `[0, total_steps]` the difference `|lr(step+1) - lr(step)|` is bounded by
    `lr_max / warmup_steps`.

Reading of the exercise: `ContinuityTest` is a `unittest.TestCase` that
checks the bound on the lesson's three shipped configurations: the demo
(warmup 4, total 20, 1e-2 -> 1e-4), the test suite's (10, 100, 1.0 -> 0.1)
and a 200-step run (20, 200, 1e-2 -> 1e-4). It uses a relative tolerance of
1e-12. It is run through `unittest` next to a literal variant with no
tolerance. Both bounds are then swept over every configuration the lesson's
validator accepts for total_steps 2-200: every warmup from 1 to total - 1,
with four (lr_max, lr_min) pairs, 79,600 schedules in all.

**ANSWER: the test passes on all three shipped configurations** once it
allows a 1e-12 relative tolerance. The largest step is exactly the warmup
slope, lr_max / warmup_steps, and it occurs inside the ramp (at steps 3, 7
and 19).

**FINDING: the bound as worded fails the lesson's own configurations.** With
no tolerance, `lr(4) - lr(3)` in the demo exceeds lr_max / 4 by one ulp,
because it is a difference of two rounded products. The literal test fails
on all three shipped configurations, by 1, 6 and 4 ulps. Over the sweep the literal test fails 76,059 of
79,600 schedules (95.6%). The continuity test needs a tolerance, or it tests
float rounding.

**FINDING: the bound is a property of the configuration, not of the code.**
With the tolerance, 47,637 of the 79,600 schedules still fail, all of them
in the cosine region. The cosine's steepest step is about
(lr_max - lr_min) * (pi / 2) / (total - warmup), which is larger than
lr_max / warmup once warmup is more than total / (1 + (pi / 2)(1 - lr_min /
lr_max)). That is 38.9% of the run for lr_min = 0; at total 200 the first
failure is at warmup 78. The closed form predicts 47,653. Every measured
failure is among them, and the 16 extra sit on the boundary, where no
integer step lands on the steepest point. The lesson's validator accepts
any warmup below total_steps. With warmup_steps = 0, which the lesson
explicitly supports, the bound divides by zero.

Structure: `worst()` is the largest step over the bound; `ContinuityTest` is
the unit test; `sweep()` runs the bound over the grid.
"""

from __future__ import annotations

import io
import math
import unittest

from harness import parity, practice

try:
    import torch  # noqa: F401  (main.py exits without it)
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "44-cosine-lr-warmup"
REF = parity.load_reference(PHASE, LESSON, "main")
SHIPPED = [(4, 20, 1e-2, 1e-4), (10, 100, 1.0, 0.1), (20, 200, 1e-2, 1e-4)]
PAIRS = [(1e-2, 1e-4), (1.0, 0.0), (3e-4, 3e-5), (1.0, 0.1)]
TOL = 1e-12


def worst(s):
    """(largest |lr(k+1) - lr(k)| / bound, the step k where it occurs)."""
    bound = s.lr_max / s.warmup_steps
    ratio, k = max((abs(s.lr(k + 1) - s.lr(k)) / bound, k) for k in range(s.total_steps + 1))
    return ratio, k


def excess_ulps(s):
    bound = s.lr_max / s.warmup_steps
    return max(abs(s.lr(k + 1) - s.lr(k)) - bound for k in range(s.total_steps + 1)) / math.ulp(bound)


class ContinuityTest(unittest.TestCase):
    tol = TOL

    def test_step_is_bounded_by_the_warmup_slope(self):
        for cfg in SHIPPED:
            with self.subTest(cfg=cfg):
                self.assertLessEqual(worst(REF.CosineWithWarmup(*cfg))[0], 1.0 + self.tol)


class LiteralTest(ContinuityTest):
    tol = 0.0


def outcome(case):
    res = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(case))
    return res.wasSuccessful(), len(res.failures)


def sweep():
    rows = [(t, w, hi, lo, *worst(REF.CosineWithWarmup(w, t, hi, lo)))
            for t in range(2, 201) for w in range(1, t) for hi, lo in PAIRS]
    over = {r[:4]: r[5] < r[1] for r in rows if r[4] > 1.0 + TOL}
    predicted = {r[:4] for r in rows if r[1] * math.pi / 2 * (1 - r[3] / r[2]) > r[0] - r[1]}
    return {"n": len(rows), "exact": sum(r[4] > 1.0 for r in rows), "tolerant": len(over),
            "in_warmup": sum(over.values()), "predicted": len(predicted),
            "unpredicted": len(over.keys() - predicted),
            "first_at_200": min(w for t, w, hi, lo in over if (t, hi, lo) == (200, 1.0, 0.0))}


def solve():
    try:
        zero = worst(REF.CosineWithWarmup(0, 100, 1.0))
    except ZeroDivisionError:
        zero = "ZeroDivisionError"
    shipped = [worst(REF.CosineWithWarmup(*c)) for c in SHIPPED]
    return {
        "tolerant": outcome(ContinuityTest), "literal": outcome(LiteralTest), "shipped": shipped,
        "at_bound": max(abs(ratio - 1) for ratio, _ in shipped) <= TOL, "at": [k for _, k in shipped],
        "ulps": [excess_ulps(REF.CosineWithWarmup(*c)) for c in SHIPPED],
        "zero_warmup": zero, "sweep": sweep(), "limit": round(1 / (1 + math.pi / 2), 3),
    }


def verify(result):
    r, s = result, result["sweep"]
    return [
        practice.Check(
            "ANSWER: the continuity test passes on all three shipped configurations",
            (r["tolerant"], r["at"], r["at_bound"]) == ((True, 0), [3, 7, 19], True),
            f"worst step / (lr_max / warmup) and its step per config: {r['shipped']}",
        ),
        practice.Check(
            "FINDING: the bound as worded fails the lesson's own configurations",
            (r["literal"], r["ulps"], s["exact"], s["n"]) == ((False, 3), [1.0, 6.0, 4.0], 76059, 79600),
            f"literal test (passed, failures): {r['literal']}; excess over lr_max/warmup per config "
            f"{r['ulps']} ulps; exact bound fails {s['exact']:,}/{s['n']:,} swept schedules",
        ),
        practice.Check(
            "FINDING: the bound is a property of the configuration, not of the code",
            (s["tolerant"], s["in_warmup"], s["predicted"], s["unpredicted"], s["first_at_200"],
             r["limit"], r["zero_warmup"]) == (47637, 0, 47653, 0, 78, 0.389, "ZeroDivisionError"),
            f"with tolerance {s['tolerant']:,} fail ({s['in_warmup']} in warmup); closed form "
            f"predicts {s['predicted']:,}, misses {s['unpredicted']}; limit {r['limit']:.1%} of "
            f"the run; first failure at total 200 is warmup {s['first_at_200']}; "
            f"warmup 0: {r['zero_warmup']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
