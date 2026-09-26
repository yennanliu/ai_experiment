"""Exercise 1 — a warm replica pays above 2.7 requests an hour, and the reference table never reads the price.

    Run `code/main.py`. Compute the break-even request rate above which a warm
    replica is cheaper than paying the cold-start tax via extra request drops at
    SLO.

Reading of the exercise: scale-to-zero is modelled as Poisson arrivals at rate
lambda with an idle timeout T; a request that finds no replica starts a cold
start of C = 328 s (the reference's RAW total), and it and every request
arriving during the boot miss the SLO. Each miss costs v, assumed $1. The warm
replica (min_workers=1) costs the GPU rate g = $4.50/hr (the reference's
argument) for the time the endpoint would have sat at zero, and saves the GPU
time spent booting. "Break-even" is where the two dollar amounts are equal.
The reference's drop budget (5/day) is a tolerance, not a price, so every miss
is charged.

**ANSWER: a warm replica is cheaper above 2.72 requests per hour.** Per hour,
scale-to-zero spends v * lambda * e^(-lambda T) * (1 + lambda C) on misses; the
warm replica costs g * e^(-lambda T) * (1 - lambda C) more. The e^(-lambda T)
cancels, so the idle timeout drops out: warm wins when lambda * (v (1 + lambda C)
+ g C) > g, i.e. lambda > 2.72/hr. A seeded 60-day simulation agrees at T = 60,
300 and 900 s: warm loses at 2.5 req/hr and wins at 3. With a 3 s snapshot
cold start the break-even moves to 4.47/hr; at v = $0.10 it is 7.75/hr.

**FINDING: the reference table reads neither the GPU price nor the cold-start
time.** `warm_pool_break_even` prints `warm better?` from drops > budget, with
drops = min(20, max(1, int(24 / rate))); `gpu_hourly` and `cold_seconds` are
only printed and the $3240 monthly cost is never compared with anything. The
column is identical for ($4.50, 328 s), ($0.85, 30 s) and ($50, 3 s): "yes" at
1 req/hr only. The first `cold_starts_per_day` assignment is dead code.

**FINDING: the reference answers the opposite direction.** Its only "yes" is
the lowest rate, so it says a warm replica pays *below* about 5 req/hr, where
the dollar model says it pays *above* 2.72/hr. At low traffic the idle GPU
costs more than the few misses it prevents.

Structure: `break_even()` is the closed form; `simulate()` checks it with the
reference's C and g; `reference_table()` captures the printed table.
"""

from __future__ import annotations

import contextlib
import io
import math
import random

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "10-cold-start-mitigation"
G, C, V = 4.50, 328.0, 1.0  # $/GPU-hour and cold seconds from main(); $ per SLO miss assumed
PRICES = ((4.50, 328, 5), (0.85, 30, 5), (50.0, 3, 5))


def break_even(g, cold_s, v):
    """Positive root of lambda * (v (1 + lambda C) + g C) = g, lambda in req/hr."""
    ch = cold_s / 3600
    a, b = v * ch, v + g * ch
    return (-b + math.sqrt(b * b + 4 * a * g)) / (2 * a)


def simulate(rate_hr, idle_s, days=60, seed=0):
    """$ saved by min_workers=1 over `days`: misses avoided minus net idle GPU cost."""
    rng, t, alive_end, ready_at = random.Random(seed), 0.0, -1.0, -1.0
    misses, zero_s, boot_s = 0, 0.0, 0.0
    while (t := t + rng.expovariate(rate_hr / 3600)) < days * 86400:
        if t > alive_end:  # scaled to zero: this request starts a cold start
            zero_s += t - max(alive_end, 0.0)
            ready_at, boot_s, misses = t + C, boot_s + C, misses + 1
        elif t < ready_at:
            misses += 1
        alive_end = max(ready_at, t) + idle_s
    return V * misses - G * (zero_s - boot_s) / 3600


def reference_table(ref, g, cold_s, budget):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ref.warm_pool_break_even(g, cold_s, budget)
    rows = [line.split() for line in buf.getvalue().splitlines()]
    return [(int(r[0]), r[3]) for r in rows if len(r) == 4 and r[0].isdigit()]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {
        "lambda": break_even(G, C, V),
        "snapshot": break_even(G, 3.0, V),
        "cheap_miss": break_even(G, C, 0.10),
        "sim": {t: (simulate(2.5, t), simulate(3.0, t)) for t in (60, 300, 900)},
        "tables": {p: reference_table(ref, *p) for p in PRICES},
    }


def verify(result):
    sim, tables = result["sim"], result["tables"]
    first = tables[PRICES[0]]
    return [
        practice.Check(
            "ANSWER: a warm replica is cheaper above 2.72 requests per hour",
            all([round(result["lambda"], 2) == 2.72, round(result["snapshot"], 2) == 4.47,
                 round(result["cheap_miss"], 2) == 7.75,
                 all(lo < 0 < hi for lo, hi in sim.values())]),
            f"closed form {result['lambda']:.2f}/hr at g=${G}, C={C:.0f}s, v=${V}; "
            f"simulated $ saved at 2.5 and 3 req/hr by idle timeout "
            f"{ {t: (round(a), round(b)) for t, (a, b) in sim.items()} }",
        ),
        practice.Check(
            "FINDING: the reference table reads neither the GPU price nor the cold-start time",
            all(t == first for t in tables.values()),
            f"'warm better?' by rate is {first} for every (price, cold s, budget) in "
            f"{list(PRICES)}",
        ),
        practice.Check(
            "FINDING: the reference answers the opposite direction",
            [r for r, w in first if w == "yes"] == [1] and result["lambda"] > 1,
            f"its only 'yes' is 1 req/hr, below the {result['lambda']:.2f}/hr above "
            "which the dollar model says the warm replica pays",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
