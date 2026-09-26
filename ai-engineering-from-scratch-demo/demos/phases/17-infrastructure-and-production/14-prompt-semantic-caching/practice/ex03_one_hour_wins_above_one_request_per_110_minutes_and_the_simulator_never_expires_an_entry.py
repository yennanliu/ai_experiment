"""Exercise 3 — one-hour wins above one request per 110 minutes, and the simulator never expires an entry.

    Calculate break-even for 1-hour TTL (2x write) vs 5-minute TTL (1.25x
    write) given your request arrival rate.

Reading of the exercise: "your request arrival rate" is the rate at which one
cacheable prefix is reused. The TTL is modelled as Anthropic documents it --
"The cache is refreshed for no additional cost each time the cached content is
used" -- so a request writes only when the gap since the prefix's last use
exceeds the TTL, and reads (0.1x) otherwise. The break-even is solved for
Poisson arrivals, checked on a seeded event simulation, and then applied to
the lesson's own workload.

**ANSWER: 1-hour wins whenever a prefix is reused more than once every 110
minutes.** With Poisson rate L per minute, a request misses with probability
exp(-L*TTL), so per request 5-min costs 1.25e^(-5L) + 0.1(1 - e^(-5L)) and
1-hour 2e^(-60L) + 0.1(1 - e^(-60L)) base-input units. They cross at
L = ln(1.9/1.15)/55 = 0.00913/min -- a mean gap of 109.5 minutes. The
simulation agrees: 5-min is cheaper at a 120-minute mean gap and 1-hour at 100.
At a 30-minute mean gap 1-hour costs 0.35x base per request against 1.07x.
Periodic traffic is sharper: at any fixed gap from 5 to 60 minutes every
5-min request is a 1.25x write while 1-hour reads at 0.1x after its first
write -- 12.3x cheaper over 1000 requests at a 30-minute gap. Beyond 60
minutes both always miss and 5-min wins, 1.25x against 2x.

**FINDING: the lesson's own traffic never reaches either TTL, so 1-hour only
costs more there.** The longest gap between two uses of one prefix in
`make_workload()` is 57 seconds; neither TTL ever expires. Switching to
1-hour buys no hits and pays 0.75x more on 12 writes: $5.85 -> $5.96.

**FINDING: the simulator's TTL is a price, not a lifetime.** `simulate` never
reads `arrived_at`; its cache is a set that is never evicted, and `ttl` only
chooses the write multiplier. In the reference, 1-hour is therefore dominated
at every arrival rate -- the break-even this exercise asks for cannot be
computed from it.

Structure: `bill()` is a refresh-on-use TTL cache over arrival times;
`poisson()` draws seeded arrivals; `crossover()` is the closed form.
"""

from __future__ import annotations

import inspect
import math
import random

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "14-prompt-semantic-caching"
READ = 0.1
TTLS = {"5min": (5.0, 1.25), "1hr": (60.0, 2.0)}
GAPS = (30, 100, 120)


def bill(times, ttl, write):
    """Mean cost per request, in base-input units, of one prefix used at `times` (minutes)."""
    last, total = None, 0.0
    for t in times:
        total += READ if last is not None and t - last <= ttl else write
        last = t
    return total / len(times)


def poisson(mean_gap, n=20000, seed=0):
    rng, t, out = random.Random(seed), 0.0, []
    for _ in range(n):
        t += rng.expovariate(1 / mean_gap)
        out.append(t)
    return out


def crossover():
    (t5, w5), (t60, w60) = TTLS["5min"], TTLS["1hr"]
    return math.log((w60 - READ) / (w5 - READ)) / (t60 - t5)


def workload_gap(reqs):
    last, gap = {}, 0.0
    for r in reqs:
        if r.prefix_hash in last:
            gap = max(gap, r.arrived_at - last[r.prefix_hash])
        last[r.prefix_hash] = r.arrived_at
    return gap


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    reqs = ref.make_workload()
    sim = {g: {k: bill(poisson(g), *v) for k, v in TTLS.items()} for g in GAPS}
    periodic = {g: {k: bill([g * i for i in range(1000)], *v) for k, v in TTLS.items()}
                for g in (3, 30, 90)}
    runs = {ttl: ref.simulate(reqs, ref.Config(False, True, False, 0.95, 0.0, ttl))
            for ttl in TTLS}
    return {
        "rate": crossover(), "sim": sim, "periodic": periodic,
        "max_gap_s": workload_gap(reqs), "costs": {k: round(v["cost"], 2) for k, v in runs.items()},
        "writes": runs["1hr"]["l2_writes"],
        "reads_arrival": "arrived_at" in inspect.getsource(ref.simulate),
    }


def verify(result):
    sim, per = result["sim"], result["periodic"]
    return [
        practice.Check(
            "ANSWER: 1-hour wins whenever a prefix is reused more than once every 110 minutes",
            all([round(1 / result["rate"], 1) == 109.5,
                 sim[100]["1hr"] < sim[100]["5min"], sim[120]["5min"] < sim[120]["1hr"],
                 round(per[30]["5min"] / per[30]["1hr"], 1) == 12.3,
                 per[90]["5min"] < per[90]["1hr"]]),
            f"crossover {result['rate']:.5f}/min = one reuse every {1 / result['rate']:.1f} "
            f"min; per-request cost by mean gap {sim}; periodic {per}",
        ),
        practice.Check(
            "FINDING: the lesson's own traffic never reaches either TTL",
            result["max_gap_s"] < 60 and result["costs"] == {"5min": 5.85, "1hr": 5.96}
            and result["writes"] == 12,
            f"longest reuse gap {result['max_gap_s']:.0f}s; bill {result['costs']} with "
            f"{result['writes']} writes in both",
        ),
        practice.Check(
            "FINDING: the simulator's TTL is a price, not a lifetime",
            not result["reads_arrival"],
            "simulate() never reads arrived_at and never evicts, so 1-hour is dominated at "
            "every arrival rate in the reference",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
