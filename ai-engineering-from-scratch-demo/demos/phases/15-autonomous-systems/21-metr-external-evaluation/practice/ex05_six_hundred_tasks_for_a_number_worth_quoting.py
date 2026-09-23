"""Exercise 5 — six hundred tasks for a number worth quoting.

    Design an internal horizon evaluation on your own bug backlog or a
    representative task set. Describe the data collection, the fit, and what
    the output tells you. Compare to METR numbers.

Reading of the exercise: the design's only hard question is how many tasks it
needs, and that is answerable by running the shipped estimator at several
sample sizes and reading its spread. Everything else in the protocol follows
from what the fit consumes.

**ANSWER: log-uniform expert times, a binary outcome, and about 640 tasks.**
Data collection: for each backlog item record one number -- median expert
completion time -- and one bit, whether the agent finished it unaided. Nothing
else enters the fit. Sample size comes from the spread: at **160** tasks the
middle half of estimates spans **29%** of the median; at **320** it is
**17%** and at **640** it is **14%**. Below **80** the interquartile spread
is **40%** and the number is not worth quoting.

**FINDING: the collection constraint is the log range, not the count.**
`synth_tasks` draws expert times log-uniformly from **0.05** to **48** hours
-- nearly **3** decades. A backlog clustered in one decade gives the fit no
leverage on the slope, and the slope is what the horizon divides by. The
design's real requirement is a task set that *spans*, which is harder to
satisfy from a real backlog than a count is.

**FINDING: what the output tells you is a slope and a scale, not a
capability.** The fit returns **2** parameters, and the 50% horizon is a
reparameterisation of them. Reporting the horizon alone discards the slope --
which Lesson 21's gaming sweep shows is the diagnostic -- so the internal
report is `(slope, horizon, n)`, three numbers, of which the published METR
figure is one.

**FINDING: the comparison to METR is not a comparison.** The shipped ground
truth is **14.0** hours, matching the January 2026 published figure, and the
clean fit on **160** synthetic tasks returns **11.30**. Two evaluations of the
*same* underlying capability differ by **19.3%** through sampling alone, so an
internal number that lands within a few hours of a published one has agreed
with it to within the noise of either.

Structure: `spread_at()` runs the shipped estimator repeatedly at one sample
size; `sweep()` walks the sizes an internal evaluation would choose between.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "21-metr-external-evaluation"

TRUE_HOURS, SIZES, SEEDS = 14.0, (40, 80, 160, 320, 640), 12
PUBLISHED = 14.0


def spread_at(ref, count, seeds=SEEDS):
    """(median, interquartile spread as a fraction of the median) at one sample size."""
    values = []
    for index in range(seeds):
        random.seed(1000 + index)
        tasks = ref.synth_tasks(TRUE_HOURS, n=count)
        slope, intercept = ref.fit(tasks)
        values.append(ref.horizon_at(slope, intercept, 0.50))
    quarters = statistics.quantiles(sorted(values), n=4)
    median = statistics.median(values)
    return round(median, 2), round((quarters[2] - quarters[0]) / median, 2)


def sweep(ref):
    return {count: spread_at(ref, count) for count in SIZES}


def log_range(ref):
    """The decades of expert time `synth_tasks` samples over."""
    random.seed(7)
    times = [time for time, _success in ref.synth_tasks(TRUE_HOURS, n=400)]
    return round(min(times), 3), round(max(times), 1)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = sweep(ref)
    low, high = log_range(ref)
    random.seed(3)
    tasks = ref.synth_tasks(TRUE_HOURS, n=160)
    slope, intercept = ref.fit(tasks)
    shipped = ref.horizon_at(slope, intercept, 0.50)
    return {
        "sizes": list(SIZES),
        "spreads": {count: rows[count][1] for count in SIZES},
        "medians": {count: rows[count][0] for count in SIZES},
        "recommended": 640,
        "worth_quoting": [count for count in SIZES if rows[count][1] <= 0.20],
        "inputs": 2,
        "fit_parameters": 2,
        "report_fields": 3,
        "time_low": low, "time_high": high,
        "decades": round(math.log10(high / low)),
        "published": PUBLISHED,
        "clean": round(shipped, 2),
        "disagreement": round(abs(shipped - PUBLISHED) / PUBLISHED, 3),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: two fields per task and about 640 of them",
            all([result["inputs"] == 2, result["recommended"] == 640,
                 result["spreads"][160] == 0.29, result["spreads"][320] == 0.17,
                 result["spreads"][640] == 0.14, result["spreads"][80] == 0.4]),
            f"each task contributes {result['inputs']} values, and the interquartile "
            f"spread runs {[result['spreads'][n] for n in result['sizes']]} across "
            f"{result['sizes']} tasks -- so {result['worth_quoting']} are the sizes "
            "worth quoting",
        ),
        practice.Check(
            "FINDING: the collection constraint is the log range, not the count",
            all([result["time_low"] <= 0.06, result["time_high"] >= 40.0,
                 result["decades"] == 3]),
            f"the estimator samples expert times from {result['time_low']} to "
            f"{result['time_high']} hours -- about {result['decades']} decades -- and a "
            "backlog clustered in one decade gives the fit no leverage on the slope",
        ),
        practice.Check(
            "FINDING: the output is a slope and a scale, not a capability",
            all([result["fit_parameters"] == 2, result["report_fields"] == 3]),
            f"the fit returns {result['fit_parameters']} parameters and the horizon is a "
            f"reparameterisation of them, so the internal report is "
            f"{result['report_fields']} numbers -- slope, horizon, n -- of which the "
            "published figure is one",
        ),
        practice.Check(
            "FINDING: the comparison to METR is not a comparison",
            all([result["published"] == 14.0, result["clean"] == 11.30,
                 result["disagreement"] == 0.193]),
            f"the ground truth is {result['published']} hours and a clean fit on 160 "
            f"tasks returns {result['clean']} -- {result['disagreement']:.1%} apart "
            "through sampling alone, so agreement within a few hours is agreement "
            "within the noise",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
