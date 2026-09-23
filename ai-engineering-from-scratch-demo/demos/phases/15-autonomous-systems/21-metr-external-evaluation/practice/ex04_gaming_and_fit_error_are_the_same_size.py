"""Exercise 4 — gaming and fit error are the same size.

    Introduce eval-context gaming into the simulator: flip ~20% of failed
    tasks to success. Report the new horizon. This approximates what a gaming
    rate of 20% does to the observed number.

Reading of the exercise: reporting the gamed number is straightforward; the
useful comparison is against the *clean fit* rather than against the ground
truth, because an external evaluator sees only one of the two and has to
decide whether the number in front of them is inflated.

**ANSWER: 14.63 hr at a 20% gaming rate, against a clean fit of 11.30 and a
ground truth of 14.0.** Gaming moves the observed horizon **+3.33** hr --
and in the direction of the truth, because the clean fit was already
**19.3%** low. A 20%-gamed evaluation of this model reports a horizon
**0.63** hr from ground truth while the honest one is **2.70** hr away.

**FINDING: the evaluator cannot tell the two apart from the number.** Fit
error at this sample size spans **11.06** to **35.49** hr across seeds, and
gaming at 10%, 20% and 40% moves the estimate to **13.25**, **14.63** and
**44.56**. The two effects overlap in both magnitude and direction, so a
single horizon figure carries no information about which one produced it.

**FINDING: gaming shows up in the slope before it shows up in the horizon.**
The fitted slope goes **-1.387**, **-1.277**, **-1.255**, **-0.854** as the
rate climbs from 0 to 40%. Flipping failures to successes flattens the curve,
because the flipped tasks are drawn disproportionately from the long end where
failures are. The slope is the diagnostic; the horizon is the slope's
reciprocal and loses the sign of the problem.

**FINDING: at 40% the number stops being an estimate.** The horizon reaches
**44.56** hr -- **3.2x** the ground truth -- and the slope is half the clean
value. That is the same collapse mechanism as a small sample: a flat fit
divides by a small number. A published horizon without a slope and a sample
size attached cannot be audited for either.

Structure: `gamed()` runs the shipped injector at a rate; `sweep()` collects
the slope and horizon at each.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "21-metr-external-evaluation"

TRUE_HOURS, SEED, COUNT = 14.0, 3, 160
RATES = (0.1, 0.2, 0.4)


def clean(ref, seed=SEED, count=COUNT):
    random.seed(seed)
    tasks = ref.synth_tasks(TRUE_HOURS, n=count)
    slope, intercept = ref.fit(tasks)
    return tasks, slope, ref.horizon_at(slope, intercept, 0.50)


def gamed(ref, tasks, rate):
    injected = ref.inject_gaming(tasks, gaming_rate=rate)
    slope, intercept = ref.fit(injected)
    return slope, ref.horizon_at(slope, intercept, 0.50)


def sweep(ref):
    tasks, slope, horizon = clean(ref)
    rows = [(0.0, round(slope, 3), round(horizon, 2))]
    for rate in RATES:
        gamed_slope, gamed_horizon = gamed(ref, tasks, rate)
        rows.append((rate, round(gamed_slope, 3), round(gamed_horizon, 2)))
    return rows


def seed_spread(ref, seeds=range(20)):
    values = []
    for seed in seeds:
        random.seed(seed)
        tasks = ref.synth_tasks(TRUE_HOURS, n=COUNT)
        slope, intercept = ref.fit(tasks)
        values.append(ref.horizon_at(slope, intercept, 0.50))
    return round(min(values), 2), round(max(values), 2)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = sweep(ref)
    clean_horizon = rows[0][2]
    twenty = next(horizon for rate, _s, horizon in rows if rate == 0.2)
    return {
        "truth": TRUE_HOURS,
        "clean": clean_horizon,
        "twenty": twenty,
        "shift": round(twenty - clean_horizon, 2),
        "clean_error": round(abs(clean_horizon - TRUE_HOURS), 2),
        "gamed_error": round(abs(twenty - TRUE_HOURS), 2),
        "gamed_closer": abs(twenty - TRUE_HOURS) < abs(clean_horizon - TRUE_HOURS),
        "rates": [rate for rate, _s, _h in rows],
        "slopes": [slope for _r, slope, _h in rows],
        "horizons": [horizon for _r, _s, horizon in rows],
        "seed_span": list(seed_spread(ref)),
        "slope_flattens": all(abs(a) >= abs(b) for a, b in
                              zip([slope for _r, slope, _h in rows],
                                  [slope for _r, slope, _h in rows][1:])),
        "forty": rows[-1][2],
        "forty_ratio": round(rows[-1][2] / TRUE_HOURS, 1),
        "forty_slope_ratio": round(abs(rows[-1][1]) / abs(rows[0][1]), 2),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 14.63 at 20% gaming, against a clean fit of 11.30",
            all([result["twenty"] == 14.63, result["clean"] == 11.30,
                 result["shift"] == 3.33, result["gamed_closer"],
                 result["gamed_error"] == 0.63, result["clean_error"] == 2.70]),
            f"gaming moves the observed horizon {result['shift']:+} hr to "
            f"{result['twenty']}, which is {result['gamed_error']} from the "
            f"{result['truth']}-hour truth while the honest fit is "
            f"{result['clean_error']} away",
        ),
        practice.Check(
            "FINDING: the evaluator cannot tell the two apart from the number",
            all([result["seed_span"] == [11.06, 35.49],
                 result["horizons"] == [11.3, 13.25, 14.63, 44.56]]),
            f"fit error spans {result['seed_span']} hr across seeds while gaming at "
            f"{result['rates'][1:]} gives {result['horizons'][1:]} -- the two effects "
            "overlap in magnitude and direction",
        ),
        practice.Check(
            "FINDING: gaming shows up in the slope before the horizon",
            all([result["slopes"] == [-1.387, -1.277, -1.255, -0.854],
                 result["slope_flattens"]]),
            f"the fitted slope runs {result['slopes']} as the rate climbs, because "
            "flipped failures come disproportionately from the long end -- the slope is "
            "the diagnostic and the horizon is its reciprocal",
        ),
        practice.Check(
            "FINDING: at 40% the number stops being an estimate",
            all([result["forty"] == 44.56, result["forty_ratio"] == 3.2,
                 result["forty_slope_ratio"] <= 0.65]),
            f"the horizon reaches {result['forty']} hr -- {result['forty_ratio']}x the "
            f"truth -- at {result['forty_slope_ratio']} of the clean slope, the same "
            "collapse a small sample produces",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
