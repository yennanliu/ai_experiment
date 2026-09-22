"""Exercise 3 — the scheduled shape is a print statement.

    Write a cron-triggered eval agent that runs nightly against your top 20
    traces from the day.

Reading of the exercise: the module's scheduled runtime is a list of
`(time, event)` pairs that `main` prints -- nothing fires, and `EventBus`
has no clock. So the cron is built on top of the shipped bus with a virtual
clock, because a real one would make the test a fact about when it ran.

**ANSWER: a nightly job that picks the day's worst 20 traces and evals them,
firing 7 times over a simulated week.** Each tick selects the **20**
lowest-scoring traces of **150**, runs the eval, and publishes a summary:
**140** evals over the week, mean score **0.07** on the selected slice
against **0.50** across all **1050** traces.

**FINDING: "top 20" is a sampling decision that decides the answer.** Taking
the 20 worst traces reports a mean of **0.07**; taking 20 at random reports
**0.49**; taking the 20 most recent reports **0.49**. The same agent, the
same day, and a **7.0x** spread in the headline -- "top 20" is ambiguous
between *worst* and *most recent*, and the two readings differ by more than
any regression this eval is meant to catch.

**FINDING: a missed night is lost, not deferred.** The cron fires on a clock
match, so a tick that does not run has no record and the next tick still
takes *that* day's 150 traces. Skipping **2** of **7** nights leaves **40**
traces never evaluated and the weekly mean unchanged at **0.07** -- the
metric cannot tell a good week from a half-measured one. Pairing the schedule
with durable execution, as the lesson says, means carrying the backlog.

**FINDING: `EventBus.publish` runs handlers in order with no isolation.** A
handler that raises takes the rest of the subscriber list with it: with **3**
subscribers on the nightly event and the first one failing, **0** of the
remaining **2** run and `publish` propagates the exception. A scheduled
runtime that fans out to several jobs needs the failure contained to one
subscriber.

Structure: `Cron` is the clock the module lacks; `slice_for()` holds the
three sampling policies compared.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "29-production-runtimes"
DAYS, PER_DAY, TOP_N, SEED = 7, 150, 20, 3


class Cron:
    """The scheduler the module prints instead of implementing."""

    def __init__(self, ref, schedule):
        self.bus, self.schedule, self.fired = ref.EventBus(), schedule, []

    def tick(self, clock, payload):
        for when, event in self.schedule:
            if when == clock:
                self.fired.append((clock, event))
                return self.bus.publish(event, payload)
        return []


def traces(day, seed=SEED):
    rng = random.Random(seed * 100 + day)
    return [{"tid": f"d{day}t{i}", "score": round(rng.random(), 2), "seq": i}
            for i in range(PER_DAY)]


def slice_for(rows, policy, rng=None):
    if policy == "worst":
        return sorted(rows, key=lambda row: row["score"])[:TOP_N]
    if policy == "recent":
        return sorted(rows, key=lambda row: -row["seq"])[:TOP_N]
    return (rng or random.Random(SEED)).sample(rows, TOP_N)


def mean(rows):
    return round(sum(row["score"] for row in rows) / len(rows), 2) if rows else 0.0


def week(ref, policy="worst", skip=()):
    cron = Cron(ref, [("02:00", "memory.consolidate"), ("03:00", "eval.nightly")])
    evaluated, rng = [], random.Random(SEED)
    cron.bus.subscribe("eval.nightly", lambda payload: f"evaluated {payload}")
    for day in range(DAYS):
        if day in skip:
            continue
        chosen = slice_for(traces(day), policy, rng)
        evaluated += chosen
        cron.tick("03:00", f"day {day}")
    return {"evaluated": evaluated, "fired": len(cron.fired),
            "mean": mean(evaluated), "count": len(evaluated)}


def isolation(ref):
    bus = ref.EventBus()
    ran = []

    def boom(payload):
        raise RuntimeError("eval backend down")

    bus.subscribe("eval.nightly", boom)
    bus.subscribe("eval.nightly", lambda p: ran.append("second"))
    bus.subscribe("eval.nightly", lambda p: ran.append("third"))
    try:
        bus.publish("eval.nightly", "day 0")
        return {"raised": False, "ran": len(ran), "subscribers": 3}
    except RuntimeError:
        return {"raised": True, "ran": len(ran), "subscribers": 3}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    worst = week(ref)
    all_rows = [row for day in range(DAYS) for row in traces(day)]
    skipped = week(ref, skip=(2, 5))
    return {
        "days": DAYS, "per_day": PER_DAY, "top_n": TOP_N,
        "fired": worst["fired"], "count": worst["count"], "mean": worst["mean"],
        "all_mean": mean(all_rows), "all_traces": len(all_rows),
        "by_policy": {policy: mean(week(ref, policy)["evaluated"])
                      for policy in ("worst", "random", "recent")},
        "spread": round(mean(week(ref, "recent")["evaluated"])
                        / worst["mean"], 1),
        "skipped_fired": skipped["fired"], "skipped_mean": skipped["mean"],
        "missed": (DAYS - skipped["fired"]) * TOP_N,
        "isolation": isolation(ref),
    }


def verify(result):
    policies, isolated = result["by_policy"], result["isolation"]
    return [
        practice.Check(
            "ANSWER: a nightly eval over the day's worst 20 traces, firing 7 times",
            all([result["fired"] == 7, result["count"] == 140,
                 result["top_n"] == 20, result["per_day"] == 150,
                 result["mean"] == 0.07, result["all_mean"] == 0.5]),
            f"each tick selects the {result['top_n']} lowest-scoring traces of "
            f"{result['per_day']} and evals them: {result['fired']} firings, "
            f"{result['count']} evals, mean {result['mean']} on the selected slice "
            f"against {result['all_mean']} across all {result['all_traces']} traces",
        ),
        practice.Check(
            "FINDING: 'top 20' is a sampling decision that decides the answer",
            all([policies["worst"] == 0.07, policies["recent"] == 0.49,
                 policies["random"] == 0.49, result["spread"] == 7.0]),
            f"the worst-20 slice reports {policies['worst']}, a random 20 reports "
            f"{policies['random']} and the most recent 20 reports "
            f"{policies['recent']} -- a {result['spread']}x spread on the same agent and "
            "the same day. A nightly number means nothing without the slice rule",
        ),
        practice.Check(
            "FINDING: a missed night is lost, not deferred",
            all([result["skipped_fired"] == 5, result["missed"] == 40,
                 result["skipped_mean"] == result["mean"]]),
            f"the cron fires on a clock match, so skipping 2 of {result['days']} nights "
            f"leaves {result['missed']} traces never evaluated while the weekly mean "
            f"stays at {result['skipped_mean']}. The metric cannot tell a good week from "
            "a half-measured one unless the schedule carries a backlog",
        ),
        practice.Check(
            "FINDING: EventBus.publish runs handlers in order with no isolation",
            all([isolated["raised"] is True, isolated["ran"] == 0,
                 isolated["subscribers"] == 3]),
            f"with {isolated['subscribers']} subscribers on the nightly event and the "
            f"first raising, {isolated['ran']} of the remaining two run and publish "
            f"propagates ({isolated['raised']}). A scheduled runtime that fans out needs "
            "the failure contained to one subscriber",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
