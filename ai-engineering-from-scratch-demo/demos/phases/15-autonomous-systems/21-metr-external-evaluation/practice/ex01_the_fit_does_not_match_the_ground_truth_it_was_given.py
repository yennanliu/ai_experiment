"""Exercise 1 — the fit does not match the ground truth it was given.

    Run `code/main.py`. Confirm the fit's 50% horizon matches the synthetic
    ground truth. Now halve the task-time grid; does the horizon estimate
    change meaningfully?

Reading of the exercise: "confirm it matches" is the instruction and checking
it is the exercise. It does not match, so the second half is answered against
an estimate that was already wrong -- which changes what "meaningfully" has to
mean.

**ANSWER: 11.30 hr against a ground truth of 14.0, and halving gives
9.98.** At the shipped seed the clean fit underestimates by **19.3%**;
halving the sample to **80** tasks takes it to **9.98** hr, a **28.7%**
underestimate. The estimate changes by **1.31** hr, which is smaller than
either error -- so the sample size moves the number by less than the number is
already off by.

**FINDING: the estimator is noisy and the shipped seed is near the bottom of
its range.** Over **20** seeds at n=160 the median estimate is **13.27** hr
with a range of **11.06** to **35.49**. The distribution has a long right
tail and the demonstration's **11.30** sits in its lower quarter, so a reader
who runs the file once sees an unusually low draw presented as the answer.

**FINDING: the tail is a slope collapse.** Sorting those 20 fits by |w|, the
three flattest slopes -- **0.754**, **1.016**, **1.034** -- carry horizons of
**35.5**, **13.9** and **11.3** hr, while the three steepest all sit near
**11-16**. `horizon_at` divides by `w`, so an estimator that is uncertain
about *whether* success depends on task length reports a very long horizon
rather than a wide interval.

**FINDING: the 10% horizon is longer than the 50%.** The fit's slope is
negative, so `horizon_at(w, b, 0.10)` returns **55.09** hr against
**11.30** at 50% and **2.32** at 90%. The column labelled 10% is the
pessimistic bound in probability and the optimistic one in hours, which is a
different axis from the "horizons are upper bounds" framing and easy to read
past in a three-number row.

Structure: `estimate()` fits one seeded sample; `across()` repeats it to get
the spread the single run hides.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "21-metr-external-evaluation"

TRUE_HOURS, SHIPPED_SEED, SHIPPED_N = 14.0, 3, 160


def estimate(ref, seed, count, true_hours=TRUE_HOURS):
    random.seed(seed)
    tasks = ref.synth_tasks(true_hours, n=count)
    slope, intercept = ref.fit(tasks)
    return slope, intercept, ref.horizon_at(slope, intercept, 0.50)


def across(ref, seeds=range(20), count=SHIPPED_N):
    rows = [estimate(ref, seed, count) for seed in seeds]
    return sorted((abs(slope), horizon) for slope, _b, horizon in rows)


def bands(ref, seed=SHIPPED_SEED, count=SHIPPED_N):
    slope, intercept, _h = estimate(ref, seed, count)
    return [round(ref.horizon_at(slope, intercept, p), 2) for p in (0.50, 0.10, 0.90)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    _w, _b, shipped = estimate(ref, SHIPPED_SEED, SHIPPED_N)
    _w2, _b2, halved = estimate(ref, SHIPPED_SEED, SHIPPED_N // 2)
    rows = across(ref)
    horizons = sorted(horizon for _slope, horizon in rows)
    return {
        "truth": TRUE_HOURS,
        "shipped": round(shipped, 2),
        "halved": round(halved, 2),
        "shipped_error": round((shipped - TRUE_HOURS) / TRUE_HOURS, 3),
        "halved_error": round((halved - TRUE_HOURS) / TRUE_HOURS, 3),
        "change": round(abs(shipped - halved), 2),
        "matches": abs(shipped - TRUE_HOURS) < 0.5,
        "seeds": len(rows),
        "median": round(statistics.median(horizons), 2),
        "span": [round(min(horizons), 2), round(max(horizons), 2)],
        "shipped_in_lower_quarter": shipped <= statistics.quantiles(horizons, n=4)[0],
        "flattest": [(round(slope, 3), round(horizon, 1)) for slope, horizon in rows[:3]],
        "steepest": [(round(slope, 3), round(horizon, 1)) for slope, horizon in rows[-3:]],
        "bands": bands(ref),
        "slope_negative": estimate(ref, SHIPPED_SEED, SHIPPED_N)[0] < 0,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 11.30 against 14.0, and halving gives 9.98",
            all([result["shipped"] == 11.30, result["halved"] == 9.98,
                 not result["matches"], result["shipped_error"] == -0.193,
                 result["halved_error"] == -0.287, result["change"] == 1.31]),
            f"the clean fit reports {result['shipped']} hr against a ground truth of "
            f"{result['truth']} -- {result['shipped_error']:.1%} -- and halving the "
            f"sample gives {result['halved']} at {result['halved_error']:.1%}; the "
            f"change of {result['change']} hr is smaller than either error",
        ),
        practice.Check(
            "FINDING: the estimator is noisy and the shipped seed is a low draw",
            all([result["seeds"] == 20, result["median"] == 13.27,
                 result["span"] == [11.06, 35.49],
                 result["shipped_in_lower_quarter"]]),
            f"over {result['seeds']} seeds the median is {result['median']} hr with a "
            f"range of {result['span']}, and the demonstration's "
            f"{result['shipped']} sits in the lower quarter of it",
        ),
        practice.Check(
            "FINDING: the tail is a slope collapse",
            all([result["flattest"][0] == (0.754, 35.5),
                 max(horizon for _s, horizon in result["steepest"]) < 20.0]),
            f"the flattest slopes {result['flattest']} carry the longest horizons while "
            f"the steepest {result['steepest']} stay near the truth -- horizon_at "
            "divides by the slope, so uncertainty about the dependence becomes a long "
            "horizon rather than a wide interval",
        ),
        practice.Check(
            "FINDING: the 10% horizon is longer than the 50%",
            all([result["bands"] == [11.3, 55.09, 2.32], result["slope_negative"]]),
            f"the three bands are {result['bands']} hr at 50%, 10% and 90% -- the "
            "column labelled 10% is pessimistic in probability and optimistic in hours",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
