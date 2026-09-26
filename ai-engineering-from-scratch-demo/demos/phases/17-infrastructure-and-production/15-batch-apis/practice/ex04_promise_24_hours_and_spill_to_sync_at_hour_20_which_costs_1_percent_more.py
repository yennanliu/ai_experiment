"""Exercise 4 — promise 24 hours and spill to sync at hour 20, which costs 1% more.

    Your batch API return SLA is 24h but P99 is 20 hours. How do you
    communicate this to the user — what is the downstream system behavior on
    the edge case?

Reading of the exercise: `code/main.py` models cost, not time, so completion
time is modelled here as a lognormal. It has the lesson's typical P50,
4 hours (the middle of 2-6), and the exercise's P99 of 20 hours. The pipeline
is the lesson's Problem: 50k documents with a 4K prompt, which the lesson
says takes 4 hours synchronously. The downstream behaviour is a spillover.
The system promises a delivery time; if the batch is not back 4 hours before
that time, it cancels and reruns on sync + cache. Both providers bill nothing
for cancelled or expired requests (checked in their docs). Each night is
priced from the reference's cost functions, and the batch is treated as
all-or-nothing: a night either spills or it doesn't.

**ANSWER: promise 24 hours and spill to sync at hour 20, which costs 1%
more.** Hour 20 is the P99, so 1% of nights spill. The expected night costs
$257.56 against $255.01 on pure batch, and every night is delivered by hour
24. Tell the user "within 24 hours", not "by morning". Without the spillover,
0.48% of batches pass 24 hours and expire, and the report is a day late.

**FINDING: promising "by morning" on the same batch spills half the nights.**
An 8-hour promise means spilling at hour 4, which is the P50. The expected
night is then $382.51, 1.5x batch, and even with no spillover 15.8% of
nights would miss the promise.

**FINDING: the skill's mis-triage alarm fires on this healthy provider.**
The skill says to alert when batch completion P95 exceeds 12 hours. With a
P99 of 20 hours the P95 is 10.19h, 12.48h and 14.06h at a P50 of 2, 4 and
6 hours. The alarm fires for any P50 above 3.5 hours, which is most of the
lesson's own typical range.

**FINDING: the reference cannot compute any of this.** `code/main.py` has no
notion of time: it has no hour, latency, SLA or percentile anywhere. The 24h,
P50 and P99 figures exist only in prose.

Structure: `latency()` fits the lognormal; `night_cost()` prices a promise
with the reference's `cost_batch_cache` and `cost_sync_cache`.
"""

from __future__ import annotations

import inspect
import math
import statistics

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "15-batch-apis"
P99_H, SLA_H, SYNC_RUN_H, ALERT_P95_H = 20.0, 24.0, 4.0, 12.0
PIPELINE = (50_000, 4000, 2000, 200)
Z = statistics.NormalDist()


def latency(p50_h):
    """Lognormal completion time with the given P50 and a P99 of 20 hours."""
    sigma = math.log(P99_H / p50_h) / Z.inv_cdf(0.99)
    return statistics.NormalDist(math.log(p50_h), sigma)


def late(dist, hours):
    return 1 - dist.cdf(math.log(hours))


def night_cost(ref, dist, promise_h):
    """(spill share, expected $) when a batch still out at promise - 4h reruns on sync."""
    spill = late(dist, promise_h - SYNC_RUN_H)
    batch, sync = ref.cost_batch_cache(*PIPELINE), ref.cost_sync_cache(*PIPELINE)
    return round(spill, 4), round((1 - spill) * batch + spill * sync, 2)


def alert_edge():
    """The P50 at which the P95 of the fitted distribution reaches 12 hours."""
    k = Z.inv_cdf(0.95) / Z.inv_cdf(0.99)
    return math.exp((math.log(ALERT_P95_H) - k * math.log(P99_H)) / (1 - k))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    dist = latency(4.0)
    source = inspect.getsource(ref).lower()
    return {
        "batch": round(ref.cost_batch_cache(*PIPELINE), 2),
        "promise_24": night_cost(ref, dist, SLA_H),
        "promise_8": night_cost(ref, dist, 8.0),
        "expire": round(late(dist, SLA_H), 4),
        "miss_8": round(late(dist, 8.0), 3),
        "p95": {
            p: round(math.exp(latency(p).inv_cdf(0.95)), 2) for p in (2.0, 4.0, 6.0)
        },
        "edge": round(alert_edge(), 2),
        "timeless": [
            w for w in ("hour", "latency", "sla", "p99", "percentile") if w in source
        ],
    }


def verify(result):
    spill24, cost24 = result["promise_24"]
    spill8, cost8 = result["promise_8"]
    p95 = result["p95"]
    return [
        practice.Check(
            "ANSWER: promise 24 hours and spill to sync at hour 20, which costs 1% more",
            all(
                [
                    spill24 == 0.01,
                    cost24 == 257.56,
                    result["batch"] == 255.01,
                    result["expire"] == 0.0048,
                ]
            ),
            f"{spill24:.0%} of nights spill; expected ${cost24} a night against "
            f"${result['batch']} pure batch; without spillover {result['expire']:.2%} "
            "of batches expire at 24h",
        ),
        practice.Check(
            "FINDING: promising 'by morning' on the same batch spills half the nights",
            spill8 == 0.5 and cost8 == 382.51 and result["miss_8"] == 0.158,
            f"an 8h promise spills at hour 4: {spill8:.0%} of nights, ${cost8} "
            f"({cost8 / result['batch']:.2f}x); with no spillover {result['miss_8']:.1%} miss",
        ),
        practice.Check(
            "FINDING: the skill's mis-triage alarm fires on this healthy provider",
            p95 == {2.0: 10.19, 4.0: 12.48, 6.0: 14.06} and result["edge"] == 3.5,
            f"P95 by P50 {p95} with P99 {P99_H}h; the P95 > {ALERT_P95_H}h alert fires "
            f"for every P50 above {result['edge']}h",
        ),
        practice.Check(
            "FINDING: the reference cannot compute any of this",
            result["timeless"] == [],
            f"words for time found in code/main.py: {result['timeless']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
