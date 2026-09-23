"""Exercise 2 — the EWMA follows the drift it is meant to flag.

    Add a statistical detector: EWMA z-score on tool-call rate. Feed in a
    trajectory that drifts slowly and show the detector never fires. Now add
    a hard limit (no more than 50 tool calls in 10 minutes) and show the hard
    limit fires on the same trajectory.

Reading of the exercise: both halves are demonstrations, so the trajectory is
built deterministically -- a linear ramp with no noise -- and the detector's
parameters are stated rather than tuned until the result appears. A drift the
EWMA misses because it was tuned to miss it would prove nothing.

**ANSWER: the EWMA never exceeds 3 sigma; the hard limit fires at minute 87.**
Over **120** minutes the tool-call rate ramps from **1.0** to **6.9** a
minute. The EWMA z-score peaks at **0.99** with alpha **0.1**, never
crossing **3.0**. The rolling 10-minute count crosses **50** at minute
**87**, where the instantaneous rate is **5.3** a minute.

**FINDING: the EWMA is a difference detector and the drift has no
difference.** Its baseline is itself an average of the signal, so on a ramp
the mean tracks the value and the residual stays small: across all 120
minutes the largest single-minute deviation from the running mean is **0.50**
calls. The statistic is doing exactly what it was built to do, on a signal it
was never going to see.

**FINDING: a step of the same total size fires immediately.** Replacing the
ramp with a jump from 1.0 to 6.9 at minute 60 -- the same start, the same end,
the same total calls to **474** either way -- puts the peak z at **11.8** and
the EWMA fires at minute **60**. The detector's sensitivity is to the derivative,
so the identical endpoint is caught or missed depending only on how it was
reached.

**FINDING: the hard limit has no tuning and no baseline.** It is two numbers
from a policy document, fires at minute **87** on the ramp and minute **67**
on the step, and needs no history beyond a **10**-minute window. That is the
argument for layering: the statistical detector is strictly better on
abruptness and strictly blind to patience, and only one of the two failure
modes is chosen by an adversary.

Structure: `ramp()` and `step()` are the two trajectories; `ewma_peak()` and
`hard_limit()` are the two detectors over either.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "14-kill-switches-canaries"

MINUTES, START, END = 120, 1.0, 6.9
ALPHA, SIGMAS, MIN_SPREAD = 0.1, 3.0, 0.5    # a variance below half a call is not trusted
WINDOW, LIMIT = 10, 50


def ramp(minutes=MINUTES):
    step_size = (END - START) / (minutes - 1)
    return [START + index * step_size for index in range(minutes)]


def step(minutes=MINUTES):
    return [START if index < minutes // 2 else END for index in range(minutes)]


def ewma_peak(series, alpha=ALPHA):
    """(peak z-score, first minute above the threshold, largest residual)."""
    mean, variance, peak, fired, worst = series[0], 0.0, 0.0, None, 0.0
    for minute, value in enumerate(series[1:], start=1):
        deviation = value - mean
        spread = max(variance ** 0.5, MIN_SPREAD)
        score = abs(deviation) / spread
        if minute > WINDOW:
            peak = max(peak, score)
            worst = max(worst, abs(deviation))
            if score > SIGMAS and fired is None:
                fired = minute
        variance = (1 - alpha) * (variance + alpha * deviation ** 2)
        mean += alpha * deviation
    return round(peak, 2), fired, round(worst, 2)


def hard_limit(series, window=WINDOW, limit=LIMIT):
    for minute in range(window, len(series)):
        if sum(series[minute - window:minute]) >= limit:
            return minute
    return None


def solve():
    parity.load_reference(PHASE, LESSON, "main")      # D5: the lesson is the source
    drift, jump = ramp(), step()
    drift_peak, drift_fired, worst = ewma_peak(drift)
    jump_peak, jump_fired, _worst = ewma_peak(jump)
    return {
        "minutes": MINUTES,
        "rate_range": [round(drift[0], 1), round(drift[-1], 1)],
        "alpha": ALPHA, "sigmas": SIGMAS,
        "drift_peak": drift_peak, "drift_fired": drift_fired,
        "worst_residual": worst,
        "hard_on_drift": hard_limit(drift),
        "rate_at_fire": round(drift[hard_limit(drift)], 1),
        "window": WINDOW, "limit": LIMIT,
        "jump_peak": jump_peak, "jump_fired": jump_fired,
        "hard_on_jump": hard_limit(jump),
        "total_drift": round(sum(drift)),
        "total_jump": round(sum(jump)),
        "totals_within": abs(round(sum(drift)) - round(sum(jump))),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the EWMA never fires; the hard limit fires at minute 80",
            all([result["drift_fired"] is None, result["drift_peak"] == 0.99,
                 result["hard_on_drift"] == 87, result["rate_at_fire"] == 5.3,
                 result["rate_range"] == [1.0, 6.9]]),
            f"over {result['minutes']} minutes the rate ramps {result['rate_range']}; "
            f"the EWMA z peaks at {result['drift_peak']} against a "
            f"{result['sigmas']}-sigma threshold and never fires, while "
            f"{result['limit']} calls in {result['window']} minutes crosses at minute "
            f"{result['hard_on_drift']}",
        ),
        practice.Check(
            "FINDING: the EWMA is a difference detector and the drift has no difference",
            result["worst_residual"] == 0.5,
            f"the largest single-minute deviation from the running mean is "
            f"{result['worst_residual']} calls across the whole ramp -- the baseline is "
            "an average of the signal, so on a ramp it tracks the value",
        ),
        practice.Check(
            "FINDING: a step of the same total size fires immediately",
            all([result["jump_fired"] == 60, result["jump_peak"] == 11.8,
                 result["totals_within"] <= 1]),
            f"the same endpoints reached as a jump at minute 60 -- totals "
            f"{result['total_drift']} against {result['total_jump']} -- put the peak z "
            f"at {result['jump_peak']} and fire at minute {result['jump_fired']}",
        ),
        practice.Check(
            "FINDING: the hard limit has no tuning and no baseline",
            all([result["hard_on_drift"] == 87, result["hard_on_jump"] == 67]),
            f"two numbers from a policy document fire at minute "
            f"{result['hard_on_drift']} on the ramp and {result['hard_on_jump']} on the "
            "step, with no history beyond the window -- the statistical detector is "
            "better on abruptness and blind to patience",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
