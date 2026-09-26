"""Exercise 4 — the cached rate prices a ticket at $0.375, but a 10% response share alone costs $3.00.

    Compute cost per resolved ticket for a support product: 3M tokens/ticket,
    ~800 tickets/day, GPT-5 cached rate.

Reading of the exercise: priced at GPT-5 list (OpenAI's model page, read
2026-09-26): $1.25 input, $0.125 cached input, $10 output per 1M tokens. Taken
literally -- every token at the cached rate -- that is the floor, so the
solution also prices what the exercise leaves out: the output share, the
cache-miss share, and the resolution rate "per resolved" divides by.

**ANSWER: $0.375 per ticket, $300 a day, $9,000 a month, at the cached
rate -- a floor.** 3M x $0.125/M. Uncached input is $3.75. With 90% of input
hitting the cache and 10% of tokens being response -- the bottom of the
lesson's 10-30% response range -- a ticket is $3.64: $3.00 of it is output,
8x the whole cached-rate answer. At 20% and 30% response it is $6.57 and $9.50.

**FINDING: "per resolved" needs a resolution rate the exercise does not
give.** At $3.64 a handled ticket, cost per *resolved* ticket is $3.64,
$4.55 and $6.07 at 100%, 80% and 60% resolution: unresolved tickets still
burn their 3M tokens, and the unit metric divides by the outcome, not the
volume.

**FINDING: the lesson's simulator prices every token as GPT-5 output.**
Driving the reference `simulate_day` with one 3M-token request gives $30.00
-- its flat $10/M is GPT-5's output price -- 80x the cached-rate answer.

**FINDING: the lesson's sample trace is priced at neither rate.** Its
`cost_usd: 0.0135` for 2950 tokens: GPT-5 list gives $0.0050 uncached and
$0.00185 with the trace's `cached_input: true`, and the simulator's flat
$10/M gives $0.0295.

**FINDING: 3M tokens is at least 8 calls.** GPT-5's context window is 400k,
so each ticket is a multi-turn agent run re-reading its context; that re-read
is exactly the traffic prefix caching discounts, and the 90% hit rate above
depends on the prefix staying byte-identical across turns.

Structure: `per_ticket()` prices a ticket for a cache-hit and response share;
`reference_flat()` runs the reference's own pricing path with its RNG
replaced by constants (restored after).
"""

from __future__ import annotations

import pathlib

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "27-finops-llms"
HERE = pathlib.Path(__file__).resolve().parent
EX02 = practice.load_module(next(HERE.glob("ex02_*.py")))
TOKENS, TICKETS, MONTH = 3_000_000, 800, 30
CONTEXT = 400_000                       # GPT-5 context window, same model page
RESOLUTION = (1.0, 0.8, 0.6)            # not given by the exercise: swept


def per_ticket(cached_share, response_share):
    """$ per ticket at GPT-5 list price for a given cache-hit and output share."""
    fresh, hit, out = EX02.PRICES["gpt-5"]
    inp = TOKENS * (1 - response_share)
    return (inp * (cached_share * hit + (1 - cached_share) * fresh)
            + TOKENS * response_share * out) / 1e6


def reference_flat(ref):
    """The simulator's own pricing: cost_per_req = tokens / 1e6 * 10.0."""
    policy = ref.TENANTS["tenant_A_normal"][0]
    saved, shim = ref.TENANTS, type("R", (), {"uniform": lambda *a: 1.0,
                                            "gauss": lambda *a: TOKENS})
    ref.TENANTS, real = {"one": (policy, ref.TenantState(), 0.01)}, ref.random
    ref.random = shim()
    try:
        ref.simulate_day(1, verbose=False)
        return ref.TENANTS["one"][1].spend_today_usd
    finally:
        ref.TENANTS, ref.random = saved, real


def trace_prices():
    """(lesson cost_usd, GPT-5 uncached, GPT-5 cached, flat $10/M) for the sample trace."""
    counts, cost = EX02.lesson_trace()
    priced = [sum(EX02.layer_dollars(counts, cached=c).values()) for c in (False, True)]
    return cost, *(round(v, 5) for v in priced), round(sum(counts.values()) / 1e6 * 10.0, 5)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    floor = per_ticket(1.0, 0.0)
    shares = [s / 100 for s in range(10, 31, 10)]
    realistic = {r: round(per_ticket(0.9, r), 4) for r in shares}
    return {
        "floor": floor, "day": floor * TICKETS, "month": floor * TICKETS * MONTH,
        "uncached": per_ticket(0.0, 0.0), "all_output": per_ticket(0.0, 1.0),
        "realistic": realistic,
        "resolved": {r: round(realistic[0.1] / r, 3) for r in RESOLUTION},
        "flat": reference_flat(ref), "calls": -(-TOKENS // CONTEXT), "trace": trace_prices(),
    }


def verify(result):
    real, res = result["realistic"], result["resolved"]
    return [
        practice.Check(
            "ANSWER: $0.375 per ticket, $300 a day, $9,000 a month, at the cached rate -- a floor",
            all([result["floor"] == 0.375, result["day"] == 300.0, result["month"] == 9000.0,
                 result["uncached"] == 3.75, real == {0.1: 3.6412, 0.2: 6.57, 0.3: 9.4987}]),
            f"cached ${result['floor']}/ticket, ${result['day']:.0f}/day, "
            f"${result['month']:,.0f}/month; uncached ${result['uncached']}; with 90% cache "
            f"hits by response share {real}",
        ),
        practice.Check(
            "FINDING: 'per resolved' needs a resolution rate the exercise does not give",
            res == {1.0: 3.641, 0.8: 4.551, 0.6: 6.069},
            f"cost per resolved ticket by resolution rate: {res}",
        ),
        practice.Check(
            "FINDING: the lesson's simulator prices every token as GPT-5 output",
            result["flat"] == result["all_output"] == 30.0,
            f"reference simulate_day on one 3M-token request: ${result['flat']:.2f}, "
            f"{result['flat'] / result['floor']:.0f}x the cached-rate answer",
        ),
        practice.Check(
            "FINDING: the lesson's sample trace is priced at neither rate",
            result["trace"] == (0.0135, 0.005, 0.00185, 0.0295),
            f"(lesson cost_usd, GPT-5 uncached, GPT-5 cached, flat $10/M): {result['trace']}",
        ),
        practice.Check(
            "FINDING: 3M tokens is at least 8 calls",
            result["calls"] == 8,
            f"3M tokens over a {CONTEXT:,}-token context window: {result['calls']} calls minimum",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
