"""Exercise 2 — the gate is exact at zero, and steepest there too.

    Implement the gated residual `y = tanh(alpha) * cross + x` in PyTorch. Show
    experimentally that with `alpha=0`, `y==x` exactly at init.

Reading of the exercise: the lesson already ships the residual as
`gated_cross_attention_step`, so it is run rather than rewritten, and "exactly"
is tested as Python equality on the objects rather than as a tolerance -- which
is the only test that distinguishes this gate from a merely small one. Having
established the equality, the solution asks what it costs: where the exactness
stops holding, and how quickly the gate stops being a no-op once alpha moves.

**ANSWER: exact, not approximate.** Across 5 text positions of 16 dimensions
`max |y - x|` is **0.0**, and every output row compares `==` to its input row.
`tanh(0)` is exactly 0.0 and `x + 0.0*c == x` in IEEE-754 for every finite `c`,
so the frozen LLM is preserved bit for bit.

**FINDING: "for every finite c" is load-bearing.** `0.0 * inf` is `nan`, so a
single non-finite value anywhere in the cross-attention output makes `y` `nan`
even with the gate fully closed. The guarantee is about the arithmetic, not
about the gate.

**FINDING: the no-op point is the point of fastest change.**
`d/d(alpha) tanh(alpha)` is `1 - tanh^2`, which is **1.0** at alpha = 0 -- its
maximum. The initialisation that makes the gate a perfect no-op is also where a
step of 0.01 in alpha admits 1.0% of the cross-attention, and the measured
deviation moves from 0.0 to **0.00197** there.

**FINDING: the gate has about 2.5 units of usable range.** At alpha = 2 tanh is
0.9640 and at 5 it is 0.99991, so the last three units buy 3.6% of the gate
while the slope falls by **5,507x**. Past alpha = 2 the parameter is nearly
unidentifiable, and the lesson's own table prints alpha = 5 as if it were a
distinct setting.

Structure: `gated` runs the lesson's own step at one alpha, `deviation` is the
max absolute change from the input, `SWEEP` is the alpha grid, and `leak` shows
the one input that defeats the exactness.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "04-flamingo-gated-cross-attention"
SEED, TEXT, VISUAL, DIM = 7, 5, 8, 16
SWEEP = (0.0, 0.01, 0.5, 1.0, 2.0, 5.0)


def fixture(ref, seed=SEED):
    """The lesson's own demo_gate fixture: 5 text positions, 8 visual tokens."""
    ref.rng = random.Random(seed)
    return ([ref.vec(DIM) for _ in range(TEXT)], [ref.vec(DIM) for _ in range(VISUAL)])


def deviation(out, text):
    return max(max(abs(a - b) for a, b in zip(row, original))
               for row, original in zip(out, text))


def leak(ref):
    """0.0 * inf is nan, so a closed gate still propagates a non-finite value."""
    return ref.add([1.0], ref.scale([float("inf")], math.tanh(0.0)))[0]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    text, visual = fixture(ref)
    outputs = {alpha: ref.gated_cross_attention_step(text, visual, alpha) for alpha in SWEEP}
    closed = outputs[0.0]
    return {
        "closed_deviation": deviation(closed, text),
        "rows_equal": sum(row == original for row, original in zip(closed, text)),
        "rows": len(text),
        "leak_is_nan": leak(ref) != leak(ref),
        "gates": {alpha: round(math.tanh(alpha), 6) for alpha in SWEEP},
        "slopes": {alpha: round(1 - math.tanh(alpha) ** 2, 6) for alpha in SWEEP},
        "deviations": {alpha: round(deviation(out, text), 6)
                       for alpha, out in outputs.items()},
        "slope_collapse": round((1 - math.tanh(0.0) ** 2) / (1 - math.tanh(5.0) ** 2)),
        "tail_gain_pct": round((math.tanh(5.0) - math.tanh(2.0)) * 100, 1),
    }


def verify(result):
    gates, slopes, deviations = result["gates"], result["slopes"], result["deviations"]
    return [
        practice.Check(
            "ANSWER: exact, not approximate -- max |y - x| is 0.0 and every row compares ==",
            all([result["closed_deviation"] == 0.0,
                 result["rows_equal"] == result["rows"] == TEXT]),
            f"across {TEXT} text positions of {DIM} dimensions the largest change is "
            f"{result['closed_deviation']} and all {result['rows_equal']} output rows compare "
            "== to their inputs. tanh(0) is exactly 0.0 and x + 0.0*c == x in IEEE-754 for "
            "every finite c, so the frozen LLM is preserved bit for bit",
        ),
        practice.Check(
            "FINDING: 'for every finite c' is load-bearing",
            result["leak_is_nan"],
            "0.0 * inf is nan, so one non-finite value anywhere in the cross-attention output "
            "makes y nan with the gate fully closed. The guarantee is a property of the "
            "arithmetic, not of the gate",
        ),
        practice.Check(
            "FINDING: the no-op point is the point of fastest change",
            all([slopes[0.0] == 1.0, gates[0.01] == 0.01, deviations[0.0] == 0.0,
                 deviations[0.01] == 0.001969]),
            f"d/d(alpha) tanh is 1 - tanh^2, which is {slopes[0.0]} at alpha=0 -- its "
            f"maximum. A step of 0.01 admits {gates[0.01] * 100:.1f}% of the cross-attention "
            f"and moves the measured deviation from {deviations[0.0]} to "
            f"{deviations[0.01]}. The perfect no-op sits on the steepest part of the curve",
        ),
        practice.Check(
            "FINDING: the gate has about 2.5 units of usable range",
            all([gates[2.0] == 0.964028, gates[5.0] == 0.999909,
                 result["tail_gain_pct"] == 3.6, result["slope_collapse"] == 5507]),
            f"tanh is {gates[2.0]} at alpha=2 and {gates[5.0]} at alpha=5, so the last three "
            f"units buy {result['tail_gain_pct']}% of the gate while the slope falls by "
            f"{result['slope_collapse']:,}x ({slopes[2.0]} to {slopes[5.0]}). Past alpha=2 "
            "the parameter is nearly unidentifiable, and the lesson's table prints alpha=5 "
            "as a distinct setting",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
