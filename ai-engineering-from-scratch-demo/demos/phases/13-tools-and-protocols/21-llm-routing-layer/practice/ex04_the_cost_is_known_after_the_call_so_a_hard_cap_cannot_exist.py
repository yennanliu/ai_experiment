"""Exercise 4 — the cost is known after the call, so a hard cap cannot exist.

    Design per-team budgets: each team has a monthly spend cap; gateway
    refuses requests once cap is hit. Pick an enforcement granularity
    (per-request or windowed).

Reading of the exercise: it asks for a granularity, which presumes the cap
can be enforced at all -- and it cannot be enforced exactly, because
`cost_usd` is computed from the response's `usage`. The gateway learns what a
request cost only after paying for it. So the design choice is not
per-request against windowed; it is which side of the cap to overshoot on,
and the two granularities differ in how far.

**ANSWER: per-request admission against a running total, which overshoots by
at most one call.** A team capped at a small budget is admitted while
`spent < cap` and refused after -- **2** admitted, **4** refused over 6
calls -- and the final spend lands at or above the cap by no more than the
last call's cost. The overshoot is bounded by the most expensive single
request the gateway will route.

**FINDING: the cap cannot be checked before the cost is known.** `route`
prices from `resp["usage"]`, which exists only after `provider_call`
returns, so a pre-check has nothing to compare. Reserving an estimate first
and reconciling after is the alternative: it never overshoots and it refuses
requests that would have fit, because the estimate has to be an upper bound.

**FINDING: the two granularities overshoot differently, and the window is
worse.** Checking every request caps the excess at one call; checking once
per window lets everything inside the window through, so the same workload
overruns by the window's whole spend. The exercise's "pick one" is a choice
between a bounded error and an unbounded one.

**FINDING: the refusal costs nothing and the gateway has nowhere to say so.**
A refused request never reaches `route`, so no `Invocation` exists -- there
is no `error`, no `attempts`, no zero-cost record. A team's refused calls are
absent from exactly the structure that would let anyone count them.

Structure: `Budget` is the ledger with a granularity switch, and `run` puts
one workload through it and reports both what was spent and what was
stopped.
"""

from __future__ import annotations

import contextlib
import inspect
import io

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "21-llm-routing-layer"
PROMPT = "summarise the quarterly incident report for the platform team"
CAP = 0.0005


class Budget:
    """A per-team ledger. `windowed` defers the check to a window boundary."""

    def __init__(self, ref, cap=CAP, *, windowed=False, window=3):
        self.ref, self.cap, self.windowed, self.window = ref, cap, windowed, window
        self.spent, self.admitted, self.refused, self.seen = 0.0, 0, 0, 0

    def allows(self):
        if self.windowed:
            return self.seen % self.window != 0 or self.spent < self.cap
        return self.spent < self.cap

    def ask(self, alias, prompt):
        self.seen += 1
        if not self.allows():
            self.refused += 1
            return None
        with contextlib.redirect_stdout(io.StringIO()):
            inv = self.ref.route(alias, [{"role": "user", "content": prompt}])
        self.spent += inv.cost_usd
        self.admitted += 1
        return inv


def run(ref, calls=6, **kwargs):
    budget = Budget(ref, **kwargs)
    for _ in range(calls):
        budget.ask("smart", PROMPT)
    return budget


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    strict = run(ref)
    windowed = run(ref, windowed=True)
    with contextlib.redirect_stdout(io.StringIO()):
        one = ref.route("smart", [{"role": "user", "content": PROMPT}])
    source = inspect.getsource(ref.route)
    return {
        "cap": CAP, "admitted": strict.admitted, "refused": strict.refused,
        "spent": round(strict.spent, 10), "over": round(strict.spent - CAP, 10),
        "one_call": round(one.cost_usd, 10),
        "overshoot_bounded": strict.spent - CAP <= one.cost_usd,
        "windowed_spent": round(windowed.spent, 10),
        "windowed_admitted": windowed.admitted, "windowed_refused": windowed.refused,
        "windowed_over": round(windowed.spent - CAP, 10),
        "prices_from_usage": 'resp["usage"]' in source or "u = resp" in source,
        "cost_after_call": source.index("provider_call") < source.index("cost_usd"),
        "refused_invocations": strict.refused,
        "records": strict.admitted,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: per-request admission against a running total, overshooting by one call",
            all([result["admitted"] == 2, result["refused"] == 4,
                 result["spent"] >= result["cap"], result["overshoot_bounded"]]),
            f"capped at {result['cap']}, the team is admitted while spent < cap: "
            f"{result['admitted']} admitted and {result['refused']} refused over 6 calls, "
            f"finishing at {result['spent']} -- {result['over']} over, which is within one "
            f"call's {result['one_call']}",
        ),
        practice.Check(
            "FINDING: the cap cannot be checked before the cost is known",
            all([result["prices_from_usage"], result["cost_after_call"]]),
            "route prices from resp['usage'], which exists only after provider_call returns, "
            "so a pre-check has nothing to compare. Reserving an upper-bound estimate and "
            "reconciling after is the alternative: it never overshoots and it refuses "
            "requests that would have fit",
        ),
        practice.Check(
            "FINDING: the two granularities overshoot differently, and the window is worse",
            all([result["windowed_spent"] > result["spent"],
                 result["windowed_over"] > result["over"],
                 result["windowed_admitted"] > result["admitted"]]),
            f"checking every request stops at {result['spent']} ({result['over']} over); "
            f"checking once per window admits {result['windowed_admitted']} and reaches "
            f"{result['windowed_spent']} ({result['windowed_over']} over). The choice is "
            "between a bounded error and one the window size sets",
        ),
        practice.Check(
            "FINDING: the refusal costs nothing and the gateway has nowhere to say so",
            all([result["refused_invocations"] == 4, result["records"] == 2]),
            f"a refused request never reaches route, so {result['refused_invocations']} "
            f"refusals produce no Invocation at all -- no error, no attempts, no zero-cost "
            f"row -- against {result['records']} records for the admitted ones. Refused "
            "calls are absent from exactly the structure that would let anyone count them",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
