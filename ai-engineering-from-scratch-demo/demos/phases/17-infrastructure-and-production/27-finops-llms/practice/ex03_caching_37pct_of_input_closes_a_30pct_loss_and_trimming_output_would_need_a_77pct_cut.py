"""Exercise 3 — caching 37% of input closes a 30% loss; trimming output would need a 77% cut.

    Your largest tenant is unit-economics-negative. Propose three interventions
    ordered by customer impact.

Reading of the exercise: "negative" is set at cost = 1.3x revenue (gross
margin -30%), per call shaped like the lesson's sample trace priced at GPT-5
list ($0.0035 input + $0.0015 output = $0.0050). Each intervention is scored by
the share of it that *alone* reaches break-even, so the ordering is by
customer impact and the sizing is by arithmetic, not by taste. Discounts:
GPT-5 cached input 90%, batch 50%, GPT-5 mini 80% cheaper per token, the
lesson's routing figure 60%.

**ANSWER: (1) invisible -- cache and route; (2) visible -- batch and cap
output; (3) contractual -- reprice.** Break-even shares: cache 36.6% of input
tokens, or route 28.8% of calls to GPT-5 mini; batch 46.2% of calls, or trim
76.9% of response tokens; or raise the price 30%. The zero-impact levers are
also the cheapest to reach. Output trimming is the most visible lever and the
weakest: response is 30% of this trace's dollars, so closing the gap with it
means cutting three quarters of every answer.

**FINDING: the lesson's "~5-10% of baseline" holds only if every call is
batched.** Stacking cache x batch x route (60%) on this trace gives 7.4% of
the naive bill; drop batch, as an interactive product must, and it is 14.8%.
With GPT-5 mini as the route it is 3.7%.

**FINDING: in the reference, the ladder pauses the most profitable tenant
and never alerts on the losing one.** Reading contracted_daily_usd as
revenue, seed 7 margins are A 99.3%, B 96.7%, C 23.1%; C spends over its
contract on days 3 and 9. The kill switch pauses A. Over 1000 disarmed seeds
C is over contract on 17.0% of days and over its 2x cap on none: a cap at 2x
contract lets a tenant lose up to 100% of revenue before anyone is told.

Structure: `break_evens()` and `stacked()` are closed-form over exercise 2's
trace pricing; `reference_margins()` reuses exercise 1's `run()`.
"""

from __future__ import annotations

import pathlib

from harness import parity, practice

HERE = pathlib.Path(__file__).resolve().parent
EX01 = practice.load_module(next(HERE.glob("ex01_*.py")))
EX02 = practice.load_module(next(HERE.glob("ex02_*.py")))
LOSS = 1.3             # the tenant's cost is 1.3x its revenue: gross margin -30%
CACHE_OFF, BATCH_OFF = 0.9, 0.5          # GPT-5 cached input; batch APIs
MINI_OFF = 1 - EX02.PRICES["gpt-5-mini"][0] / EX02.PRICES["gpt-5"][0]
LESSON_ROUTE = 0.6                       # the lesson's "60% cost reduction"


def trace_split():
    """(input $, output $) of the lesson's sample trace at GPT-5 list price."""
    dollars = EX02.layer_dollars(EX02.lesson_trace()[0])
    return sum(v for k, v in dollars.items() if k != "response"), dollars["response"]


def break_evens():
    """Share of each lever that alone brings cost down to revenue, in impact order."""
    inp, out = trace_split()
    cost, target = inp + out, (inp + out) / LOSS
    gap = cost - target
    return {
        "1a cache input prefixes": gap / (inp * CACHE_OFF),
        "1b route calls to gpt-5-mini": gap / (cost * MINI_OFF),
        "2a batch async calls": gap / (cost * BATCH_OFF),
        "2b trim response tokens": gap / out,
        "3 reprice at renewal": LOSS - 1,
    }


def stacked(batch=True, route=LESSON_ROUTE):
    """The lesson's four-lever stack as a fraction of the naive bill."""
    inp, out = trace_split()
    b = 1 - BATCH_OFF if batch else 1.0
    return (inp * (1 - CACHE_OFF) * b * (1 - route) + out * b * (1 - route)) / (inp + out)


def reference_margins(ref):
    """Shipped seed-7 run, reading contracted_daily_usd as the tenant's daily revenue."""
    history, log = EX01.run(ref, 7)
    out = {}
    for name, (policy, _, _) in ref.TENANTS.items():
        contract = policy.contracted_daily_usd
        spend = history[name]
        out[name] = {"contract": contract, "margin": round(1 - sum(spend) / (10 * contract), 3),
                     "negative_days": [d + 1 for d, usd in enumerate(spend) if usd > contract],
                     "cap": contract * policy.spend_cap_multiplier,
                     "paused": f"{name}: z=" in log}
    return out


def solve():
    ref = parity.load_reference(EX01.PHASE, EX01.LESSON, "main")
    steady = EX01.histories(ref)
    c_days = [usd for name, h in steady if name == "tenant_C_abusive" for usd in h]
    return {
        "split": tuple(round(v, 5) for v in trace_split()),
        "break_even": {k: round(v, 3) for k, v in break_evens().items()},
        "stack": round(stacked(), 3), "stack_live": round(stacked(batch=False), 3),
        "stack_mini": round(stacked(route=MINI_OFF), 3), "mini_off": MINI_OFF,
        "ref": reference_margins(ref),
        "c_negative": round(sum(usd > 20 for usd in c_days) / len(c_days), 3),
        "c_alerted": sum(usd > 40 for usd in c_days),
    }


def verify(result):
    be, ref = result["break_even"], result["ref"]
    return [
        practice.Check(
            "ANSWER: (1) invisible -- cache and route; (2) visible -- batch and cap output; "
            "(3) contractual -- reprice",
            be == {"1a cache input prefixes": 0.366, "1b route calls to gpt-5-mini": 0.288,
                   "2a batch async calls": 0.462, "2b trim response tokens": 0.769,
                   "3 reprice at renewal": 0.3},
            f"break-even share of each lever alone, per call ${result['split']} "
            f"(input, output) at a -30% margin: {be}",
        ),
        practice.Check(
            "FINDING: the lesson's '~5-10% of baseline' holds only if every call is batched",
            0.05 <= result["stack"] <= 0.10 < result["stack_live"],
            f"cache x batch x route(60%) = {result['stack']:.1%} of the naive bill; without "
            f"batch {result['stack_live']:.1%}; routing to gpt-5-mini "
            f"({result['mini_off']:.0%} off) {result['stack_mini']:.1%}",
        ),
        practice.Check(
            "FINDING: the ladder pauses the most profitable tenant and never alerts on the "
            "losing one",
            all([ref["tenant_A_normal"]["paused"], not ref["tenant_C_abusive"]["paused"],
                 ref["tenant_C_abusive"]["negative_days"] == [3, 9],
                 result["c_negative"] == 0.17, result["c_alerted"] == 0]),
            f"margins { {n: r['margin'] for n, r in ref.items()} }; C over contract on days "
            f"{ref['tenant_C_abusive']['negative_days']}; over 1000 seeds C is over contract "
            f"on {result['c_negative']:.1%} of days and over its cap on {result['c_alerted']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
