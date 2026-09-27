"""Exercise 1 — no curve peaks: gold is a straight line in sqrt(KL) and the printed peak is the last grid point.

    Run `code/main.py`. Reproduce the gold-peak-then-collapse shape for proxies
    fit on 100, 300, 1000 samples. Where does each curve peak in KL units?

Reading of the exercise: "run code/main.py" is the shipped run, seed 42,
replayed through the reference's own `main()`; "where does each curve peak"
is read off its printed "gold peak at sqrt(KL)" line and then tested: is that
point a peak of the curve, or only the largest budget on the grid?

**ANSWER: none of the three curves peaks.** The run prints "gold peak at
sqrt(KL) = 2.828" for 100, 300 and 1000 samples alike (gold 6.333, 6.374,
6.377), and 2.828 is sqrt(8), the last of the nine budgets. Gold rises at a
constant slope of 2.239, 2.253 and 2.254 per unit of sqrt(KL) at every
budget (the ratio varies by under 1e-9), so the "peak" is wherever the grid
stops. Extending the grid to a budget of 1e6 moves it to sqrt(KL) = 1000.

**FINDING: the collapse cannot happen in this model.** The KL-constrained
optimum is mu = s * w_proxy, so gold = s * <w_gold, w_proxy>: a line through
the origin, whose slope is sqrt(2) |w_gold| cos(w_gold, w_proxy). The
cosines are 0.9914, 0.9978 and 0.9983. Gold can only fall if the proxy points
more than 90 degrees away from gold, and then it falls from the origin. The
printed TAKEAWAY, "gold peaks and falls", is contradicted by every table the
same run prints.

**FINDING: the proxy-gold gap does not even keep its sign.** At sqrt(KL) =
2.828 it is +0.100 and +0.203 for 100 and 300 samples and -0.108 for 1000:
the 1000-sample proxy under-rates the policy it produces. With a linear
proxy and a linear gold, proxy error is estimation error in the slope, not
Goodhart.

**FINDING: the best-of-N curve's x axis is not KL.** Its first row, n = 1,
sits at "sqrt(KL)" = 1.920, although best-of-1 is the initial policy (KL =
0). The column is the mean distance of the chosen sample from the origin.

Structure: `shipped()` runs `main()` with `ref.random` swapped for
`random.Random(42)` (restored after) and parses its output; `proxies()`
redraws the three proxies in the same order so their weights can be read.
"""

from __future__ import annotations

import contextlib
import io
import math
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "02-reward-hacking-goodhart"
SIZES = (100, 300, 1000)
BUDGETS = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]      # main()'s grid
PEAK = r"gold peak at sqrt\(KL\) = ([\d.]+), gold = ([-\d.]+), proxy = ([-\d.]+)"
ROW = r"(?m)^ +([\d.]+) +([-\d.]+) +([-\d.]+) +([-+\d.]+)$"


def shipped(ref):
    """main()'s printed output on its own seed."""
    saved, log = ref.random, io.StringIO()
    ref.random = random.Random(42)
    try:
        with contextlib.redirect_stdout(log):
            ref.main()
        return log.getvalue()
    finally:
        ref.random = saved


def proxies(ref):
    """The first three proxies main() fits, redrawn from the same seed."""
    saved, ref.random = ref.random, random.Random(42)
    try:
        return [ref.train_proxy(n) for n in SIZES]
    finally:
        ref.random = saved


def cosine(ref, w):
    return ref.dot(w, ref.GOLD_W) / math.sqrt(ref.dot(w, w) * ref.dot(ref.GOLD_W, ref.GOLD_W))


def parse(log):
    """(printed peaks, printed table rows) as floats."""
    return ([tuple(map(float, p)) for p in re.findall(PEAK, log)],
            [tuple(map(float, r)) for r in re.findall(ROW, log)])


def line(ref, rm):
    """One proxy's sweep: (gold slope, slope spread, cosine, closed-form slope, end gap)."""
    curve = ref.kl_constrained_policy_sweep(rm, BUDGETS)
    slopes = [g / d for d, _, g in curve[1:]]
    cos = cosine(ref, rm.w)
    closed = math.sqrt(2 * ref.dot(ref.GOLD_W, ref.GOLD_W)) * cos
    return (round(slopes[0], 3), max(slopes) - min(slopes), round(cos, 4), round(closed, 3),
            round(curve[-1][1] - curve[-1][2], 3))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    log = shipped(ref)
    peaks, rows = parse(log)
    rms = proxies(ref)
    slope, spread, cos, closed, gap = zip(*(line(ref, rm) for rm in rms))
    wide = ref.kl_constrained_policy_sweep(rms[0], BUDGETS + [1e2, 1e4, 1e6])
    return {
        "peaks": [p[:2] for p in peaks[:3]], "grid_end": round(math.sqrt(BUDGETS[-1]), 3),
        "slope": list(slope), "spread": max(spread), "cos": list(cos),
        "slope_formula": list(closed), "gap_end": list(gap),
        "wide_peak": round(max(wide, key=lambda r: r[2])[0], 3),
        "bon_first_x": rows[-8][0],
        "takeaway": "gold peaks and falls" in log,
    }


def verify(result):
    peaks = result["peaks"]
    return [
        practice.Check(
            "ANSWER: none of the three curves peaks; the printed peak is the last grid point",
            (peaks, result["grid_end"], result["wide_peak"], result["slope"])
            == ([(2.828, 6.333), (2.828, 6.374), (2.828, 6.377)], 2.828, 1000.0,
                [2.239, 2.253, 2.254]) and result["spread"] < 1e-9,
            f"printed peaks {peaks}; grid ends at {result['grid_end']}; gold/sqrt(KL) = "
            f"{result['slope']} at every budget (spread {result['spread']:.1e}); grid to 1e6 "
            f"-> peak at {result['wide_peak']}",
        ),
        practice.Check(
            "FINDING: the collapse cannot happen -- gold = sqrt(2)|w_gold| cos * sqrt(KL)",
            (result["slope_formula"], result["cos"], result["takeaway"])
            == (result["slope"], [0.9914, 0.9978, 0.9983], True),
            f"cos(w_gold, w_proxy) = {result['cos']}; closed-form slopes "
            f"{result['slope_formula']}; main() still prints 'gold peaks and falls'",
        ),
        practice.Check(
            "FINDING: the proxy-gold gap does not keep its sign",
            result["gap_end"] == [0.1, 0.203, -0.108],
            f"gap at sqrt(KL) = 2.828 for 100/300/1000 samples: {result['gap_end']}",
        ),
        practice.Check(
            "FINDING: the best-of-N x axis is not KL",
            result["bon_first_x"] == 1.92,
            f"best-of-1 is plotted at 'sqrt(KL)' = {result['bon_first_x']}, not 0",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
