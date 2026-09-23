"""Exercise 1 — the thirty-hour crossing lands between two rows of the table.

    Run the simulator. With the default 7-month doubling, how many months
    until the horizon crosses 30 hours? 168 hours? Plot the two crossings.

Reading of the exercise: "plot" in a stdlib lesson is an ASCII axis, not a
chart library. The lesson ships `assets/horizon-curve.svg` and its `code/`
produces no image, so the deliverable is a month axis with a caret under each
crossing -- printed by the same module that computes them.

**ANSWER: 30 hours at month 7.70, 168 hours at month 25.09.** Both fall out of
the lesson's own `months_to_cross` against the shipped baseline of **14.0** hr
at a **7.0**-month doubling: 30 hours is **1.10** doublings away and one week
of expert work is **3.58**. The axis marks them at months **8** and **25**.

**FINDING: the lesson prints one of the two numbers and not the other.**
`horizon_projection` loops over **4** crossing targets -- 24, 48, 168 and 720
hours -- so the 168-hour answer is already on screen and the 30-hour one is
not. The exercise asks for the crossing the table skips.

**FINDING: the projection grid straddles the crossing it was asked about.**
The table steps 6 months, so month 6 reads **25.4** hr and month 12 reads
**45.9** hr. **0** of its **7** rows lands within an hour of 30, and a reader
who trusts the table sees the horizon pass 30 hours somewhere in a six-month
gap it never prints.

**FINDING: the distance between the two crossings does not depend on the
baseline.** At baselines of 7, 14 and 28 hours the crossings move but the gap
is **17.40** months in all three, because it is `7 * log2(168 / 30)` and the
baseline cancels. The only thing a better baseline estimate buys is *when*
both crossings happen, never how far apart they are.

Structure: `crossings()` asks the lesson for the two months; `plot()` renders
them on one 36-month ASCII axis.
"""

from __future__ import annotations

import inspect
import math
import re

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "01-long-horizon-agents"

BASELINE_HOURS, DOUBLING_MONTHS = 14.0, 7.0
TARGETS = (30.0, 168.0)
SPAN = 36                       # the last month the shipped projection table prints
GRID = (0, 6, 12, 18, 24, 30, 36)


def crossings(ref, baseline=BASELINE_HOURS):
    cfg = ref.HorizonConfig(baseline, 0, DOUBLING_MONTHS)
    return [ref.months_to_cross(cfg, target) for target in TARGETS]


def plot(months):
    """One ASCII month axis with a caret under each crossing."""
    lane = ["."] * (SPAN + 1)
    for month in months:
        lane[round(month)] = "^"
    ticks = "".join("|" if i % 6 == 0 else " " for i in range(SPAN + 1))
    labels = "".join(f"{i:<6}" for i in range(0, SPAN + 1, 6))
    return ["".join(lane), ticks, labels.rstrip()]


def grid_hours(ref):
    """The horizon at each month the shipped projection table prints."""
    cfg = ref.HorizonConfig(BASELINE_HOURS, 0, DOUBLING_MONTHS)
    return [round(ref.horizon_at(cfg, month), 1) for month in GRID]


def printed_targets(ref):
    """The crossing targets `horizon_projection` puts on screen."""
    found = re.search(r"for target in \(([^)]*)\)",
                      inspect.getsource(ref.horizon_projection)).group(1)
    return [float(target) for target in found.split(",")]


def gaps(ref, baselines=(7.0, 14.0, 28.0)):
    return [round(pair[1] - pair[0], 2)
            for pair in (crossings(ref, base) for base in baselines)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    months = crossings(ref)
    lane = plot(months)[0]
    grid, printed = grid_hours(ref), printed_targets(ref)
    return {
        "months": [round(month, 2) for month in months],
        "doublings": [round(month / DOUBLING_MONTHS, 2) for month in months],
        "lane": lane,
        "marks": [index for index, char in enumerate(lane) if char == "^"],
        "printed_targets": printed,
        "prints_168": 168.0 in printed,
        "prints_30": 30.0 in printed,
        "grid": grid,
        "near_30": sum(abs(hours - 30.0) <= 1.0 for hours in grid),
        "straddle": [grid[1], grid[2]],
        "gaps": gaps(ref),
        "ratio_only": round(DOUBLING_MONTHS * math.log2(TARGETS[1] / TARGETS[0]), 2),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: month 7.70 for 30 hours, month 25.09 for one week",
            all([result["months"] == [7.7, 25.09], result["marks"] == [8, 25],
                 result["doublings"] == [1.1, 3.58]]),
            f"the two crossings are months {result['months']} -- "
            f"{result['doublings']} doublings -- plotted at {result['marks']} on "
            f"{result['lane']}",
        ),
        practice.Check(
            "FINDING: the lesson prints one of the two crossings and not the other",
            all([result["prints_168"], not result["prints_30"],
                 len(result["printed_targets"]) == 4]),
            f"horizon_projection prints crossings for {result['printed_targets']}, so "
            f"168 hours is already on screen and 30 is the one the exercise has to "
            "compute",
        ),
        practice.Check(
            "FINDING: the projection grid straddles the crossing it was asked about",
            all([result["near_30"] == 0, result["straddle"] == [25.4, 45.9],
                 len(result["grid"]) == 7]),
            f"the 6-month grid reads {result['straddle'][0]} hr at month 6 and "
            f"{result['straddle'][1]} hr at month 12, so {result['near_30']} of "
            f"{len(result['grid'])} rows lands within an hour of 30",
        ),
        practice.Check(
            "FINDING: the gap between crossings does not depend on the baseline",
            all([result["gaps"] == [17.4, 17.4, 17.4],
                 result["ratio_only"] == 17.4]),
            f"baselines of 7, 14 and 28 hours all give a gap of {result['gaps'][0]} "
            f"months, because it is 7 * log2(168/30) = {result['ratio_only']} and the "
            "baseline cancels",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
