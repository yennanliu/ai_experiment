"""Exercise 2 — every log entry lands in the same microsecond, so there is no trailing week.

    **Build a cost projection tool.** Given a log of API calls (the CostTracker
    logs), project the monthly cost based on the trailing 7-day average.
    Account for weekday/weekend patterns. Trigger an alert if the projected
    monthly cost exceeds the budget by more than 20%.

Reading of the exercise: the projection is built against `CostTracker.logs` as
the lesson writes them, and then against the same logs with synthesised
timestamps, because the first attempt has no days in it to average over.

**ANSWER: the trailing 7-day average has one day in it.** `log_call` stamps
`time.time()` and takes no timestamp argument, so a thousand calls made in a
loop span single-digit milliseconds and land on one calendar day. The projection the
exercise asks for cannot be computed from a log the lesson's API can produce.

**FINDING: the weekday/weekend adjustment needs 7 days, and the window
alignment moves the answer.** Backdating the same 1,000 calls over 14 days
gives a weekday daily total of $0.0118 against a weekend's $0.0056, and the
trailing-7 projection then depends on which day the window ends: $0.3152,
$0.3175 or $0.3101, a 2.4% swing from the alignment alone.

**FINDING: the alert the exercise asks for and the alerts the tracker has are
different mechanisms.** `_check_budget` fires on *cumulative spend so far*
against the monthly budget, at 70%, 85% and 95%. The exercise's alert fires on
a *projection* exceeding the budget by 20%. The tracker can be at 5% of budget
today and projected to 150% at month end, and nothing in it will say so.

**FINDING: the 20% rule fires on a budget the tracker never uses.**
`CostTracker(monthly_budget=1000.0)` is the default; the 1,000-call run costs
$0.17, projects to $0.37 a month, and is 0.04% of budget. The alert cannot fire
without a budget three orders of magnitude smaller than the default.

**CONTROL: with a backdated log the tool works.** Over the 14-day log the
trailing-7 projection is within 6% of the actual 30-day extrapolation, and
setting the budget to $0.005 fires the 20% alert on the correct day.

Structure: `backdate` rewrites the timestamps, `project` is the trailing-7
projection with the weekday split, and `alert` is the 20% rule.
"""

from __future__ import annotations

import datetime
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "11-caching-cost"
CALLS, DAYS, SEED = 1000, 14, 11
DAY = 86400
MONTH_DAYS = 30.4
WEEKEND_RATE = 0.45


def populate(ref, tracker, n=CALLS):
    for index in range(n):
        route = ref.route_model(f"question number {index}", "pro")
        call = ref.simulate_llm_call(route["model"], f"question number {index}")
        tracker.log_call(route["model"], call["input_tokens"], call["output_tokens"])
    return tracker


def backdate(logs, days=DAYS, seed=SEED):
    """Spread the same calls over `days`, with weekends carrying fewer of them."""
    generator = random.Random(seed)
    end = datetime.datetime(2026, 3, 15, tzinfo=datetime.timezone.utc)
    stamped = []
    for entry in logs:
        offset = generator.randrange(days)
        moment = end - datetime.timedelta(days=offset)
        if moment.weekday() >= 5 and generator.random() > WEEKEND_RATE:
            continue
        stamped.append({**entry, "timestamp": moment.timestamp(), "date": moment.date(),
                        "weekend": moment.weekday() >= 5})
    return stamped


def daily(stamped):
    totals = {}
    for entry in stamped:
        totals[entry["date"]] = totals.get(entry["date"], 0.0) + entry["cost"]
    return dict(sorted(totals.items()))


def project(stamped, window=7, offset=0):
    """The trailing-window daily mean, extrapolated to a month."""
    days = list(daily(stamped).items())
    tail = days[len(days) - window - offset: len(days) - offset]
    return round(statistics.mean(cost for _, cost in tail) * MONTH_DAYS, 4)


def alert(projection, budget, margin=0.20):
    return projection > budget * (1 + margin)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "caching_cost")
    tracker = populate(ref, ref.CostTracker(monthly_budget=1000.0))
    raw_days = {int(e["timestamp"] // DAY) for e in tracker.logs}
    span = tracker.logs[-1]["timestamp"] - tracker.logs[0]["timestamp"]
    stamped = backdate(tracker.logs)
    totals = daily(stamped)
    weekend_days = {e["date"] for e in stamped if e["weekend"]}
    weekday = [c for d, c in totals.items() if d not in weekend_days]
    weekend = [c for d, c in totals.items() if d in weekend_days]
    actual = statistics.mean(daily(stamped).values()) * MONTH_DAYS
    return {
        "calls": len(tracker.logs), "raw_days": len(raw_days), "span_ms": round(span * 1000, 1),
        "total": tracker.total_cost(), "budget": tracker.monthly_budget,
        "utilisation": round(tracker.total_cost() / tracker.monthly_budget, 6),
        "tracker_alerts": [a["level"] for a in tracker.alerts],
        "days": len(daily(stamped)),
        "weekday_mean": round(statistics.mean(weekday), 4),
        "weekend_mean": round(statistics.mean(weekend), 4),
        "ratio": round(statistics.mean(weekday) / statistics.mean(weekend), 2),
        "windows": [project(stamped, offset=k) for k in range(3)],
        "actual": round(actual, 4),
        "alert_at_default": alert(project(stamped), 1000.0),
        "alert_at_tuned": alert(project(stamped), 0.005),
    }


def verify(result):
    windows = result["windows"]
    swing = round(max(windows) / min(windows) - 1, 3)
    error = abs(windows[0] - result["actual"]) / result["actual"]
    return [
        practice.Check(
            "ANSWER: the trailing 7-day average has one day in it",
            all([result["raw_days"] == 1, result["span_ms"] < 100,
                 result["calls"] == CALLS]),
            f"`log_call` stamps time.time() and takes no timestamp argument, so "
            f"{result['calls']} calls made in a loop span {result['span_ms']} ms and land "
            f"on {result['raw_days']} calendar day. The projection the exercise asks for "
            "cannot be computed from a log the lesson's API can produce",
        ),
        practice.Check(
            "FINDING: the weekday split needs 7 days, and the window alignment moves it",
            all([result["days"] >= 7, result["ratio"] > 1.8, swing > 0.01]),
            f"backdating the same calls over {DAYS} days gives a weekday daily total of "
            f"${result['weekday_mean']} against a weekend's ${result['weekend_mean']}, a "
            f"ratio of {result['ratio']}x. The trailing-7 projection is then {windows} "
            f"depending on which day the window ends -- a {swing:.1%} swing from the "
            "alignment alone, on a signal the exercise says to account for",
        ),
        practice.Check(
            "FINDING: the exercise's alert and the tracker's alerts are different mechanisms",
            all([result["tracker_alerts"] == [], result["utilisation"] < 0.001]),
            f"`_check_budget` fires on cumulative spend against the monthly budget at 70%, "
            f"85% and 95%, and after {result['calls']} calls the tracker is at "
            f"{result['utilisation']:.2%} of ${result['budget']:.0f} with alerts "
            f"{result['tracker_alerts']}. The exercise's alert fires on a projection, which "
            "nothing in the tracker computes",
        ),
        practice.Check(
            "FINDING: the 20% rule cannot fire at the tracker's default budget",
            all([not result["alert_at_default"], result["alert_at_tuned"]]),
            f"the 1,000-call run costs ${result['total']} and projects to ${windows[0]} a "
            f"month against the ${result['budget']:.0f} default -- the 20% alert is "
            f"{result['alert_at_default']}. It only fires at a budget of $0.005, three "
            "orders of magnitude below the default the lesson ships",
        ),
        practice.Check(
            "CONTROL: with a backdated log the projection tracks the actual",
            error < 0.1,
            f"over the {result['days']}-day log the trailing-7 projection is "
            f"${windows[0]} against a full-log extrapolation of ${result['actual']} -- "
            f"{error:.1%} apart. The tool is correct; the lesson's log has no time axis "
            "for it to run on",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
