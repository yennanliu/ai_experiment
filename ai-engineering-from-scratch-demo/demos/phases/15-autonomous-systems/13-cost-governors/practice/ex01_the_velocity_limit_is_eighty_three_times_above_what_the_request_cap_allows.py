"""Exercise 1 — the velocity limit sits above what the request cap allows.

    Run `code/main.py`. Confirm the velocity limit fires before the
    iteration cap on a polling-loop trajectory. Now disable the velocity
    limit and measure how much the agent "spends" before the iteration cap
    catches it.

Reading of the exercise: "confirm" is an instruction to check, so the first
thing to do is check it. It does not hold, and the reason is arithmetic
between two layers rather than anything about the trajectory -- so the second
half of the exercise is run anyway, and its answer is "the same amount".

**ANSWER: the velocity limit never fires, and disabling it changes nothing.**
The layered stack stops at `max_turns` on turn **200** having spent
**$4.32**; with the velocity limit disabled it stops at `max_turns` on turn
**200** having spent **$4.32**. Identical turns, tokens, dollars and stopping
reason.

**FINDING: the loop's rate is 104x below the limit meant to catch it.** At
**8000** tokens a turn and **30** seconds a turn, the polling loop burns
**$0.048** a minute against a `velocity_usd_per_min` of **$5.00**. The
measure and the threshold are three orders of magnitude apart.

**FINDING: no trajectory that respects the request cap can trip the velocity
limit.** `max_tokens_per_request` of **10000** is **$0.03** a turn, which at
two turns a minute is **$0.06** a minute -- **83.3x** below the velocity
threshold. The two layers are inconsistent by construction: the cheaper one
makes the more expensive one unreachable, whatever the agent does.

**FINDING: three of the five layers never fire on this profile.** The
request cap does not truncate, because **8000** is under **10000**; the
velocity limit is unreachable; and the monthly cap of **$500** is never
approached -- **10000** turns of pure loop reach **$239.52**. The stack the
lesson presents as five layers is, on its own demonstration, one layer
(`max_turns`) with a dollar cap behind it.

Structure: `run()` drives the shipped simulator with a chosen governor;
`rates()` converts the module's own constants into dollars a minute.
"""

from __future__ import annotations

import contextlib
import io

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "13-cost-governors"


def run(ref, **flags):
    governor = ref.Governor(**flags)
    with contextlib.redirect_stdout(io.StringIO()):
        finished = ref.simulate(governor, "probe")
    return governor, finished


def summary(finished):
    return [finished.turns, finished.tokens, round(finished.dollars, 2),
            finished.stopped_by]


def rates(ref, governor):
    """Dollars a minute for the loop, and for the most a request cap permits."""
    per_minute = 60.0 / governor.seconds_per_turn
    dollars = ref.DOLLARS_PER_KTOK / 1000.0
    return (round(ref.LOOP_TURN_TOKENS * dollars * per_minute, 4),
            round(governor.max_tokens_per_request * dollars * per_minute, 4))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    governor, layered = run(ref)
    _g, without = run(ref, enable_velocity=False)
    _g, monthly = run(ref, enable_request_cap=False, enable_iter_cap=False,
                      enable_velocity=False, enable_session_cap=False)
    uncapped = ref.Governor(enable_request_cap=False, enable_velocity=False,
                            enable_session_cap=False, enable_monthly_cap=False)
    uncapped.max_turns = 10_000
    with contextlib.redirect_stdout(io.StringIO()):
        unbounded = ref.simulate(uncapped, "unbounded")
    loop_rate, cap_rate = rates(ref, governor)
    return {
        "layered": summary(layered),
        "without_velocity": summary(without),
        "identical": summary(layered) == summary(without),
        "loop_rate": loop_rate,
        "cap_rate": cap_rate,
        "limit": governor.velocity_usd_per_min,
        "loop_headroom": round(governor.velocity_usd_per_min / loop_rate, 1),
        "cap_headroom": round(governor.velocity_usd_per_min / cap_rate, 1),
        "request_cap_truncates": ref.LOOP_TURN_TOKENS > governor.max_tokens_per_request,
        "loop_tokens": ref.LOOP_TURN_TOKENS,
        "request_cap": governor.max_tokens_per_request,
        "monthly": summary(monthly),
        "monthly_cap": governor.monthly_cap_usd,
        "unbounded_dollars": round(unbounded.dollars, 2),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the velocity limit never fires and disabling it changes nothing",
            all([result["layered"] == [200, 1440500, 4.32, "max_turns"],
                 result["identical"]]),
            f"the layered stack stops at {result['layered'][3]} on turn "
            f"{result['layered'][0]} having spent ${result['layered'][2]}, and with the "
            f"velocity limit disabled it stops at {result['without_velocity'][3]} on "
            f"turn {result['without_velocity'][0]} having spent "
            f"${result['without_velocity'][2]}",
        ),
        practice.Check(
            "FINDING: the loop's rate is 104x below the limit meant to catch it",
            all([result["loop_rate"] == 0.048, result["limit"] == 5.0,
                 result["loop_headroom"] == 104.2]),
            f"the polling loop burns ${result['loop_rate']} a minute against a "
            f"velocity_usd_per_min of ${result['limit']} -- a factor of "
            f"{result['loop_headroom']}",
        ),
        practice.Check(
            "FINDING: no trajectory respecting the request cap can trip the velocity limit",
            all([result["cap_rate"] == 0.06, result["cap_headroom"] == 83.3]),
            f"max_tokens_per_request of {result['request_cap']} permits at most "
            f"${result['cap_rate']} a minute, {result['cap_headroom']}x below the "
            "velocity threshold -- the cheaper layer makes the more expensive one "
            "unreachable whatever the agent does",
        ),
        practice.Check(
            "FINDING: three of the five layers never fire on this profile",
            all([not result["request_cap_truncates"],
                 result["monthly"][3] == "ran out of simulated turns",
                 result["unbounded_dollars"] == 239.52,
                 result["unbounded_dollars"] < result["monthly_cap"]]),
            f"the request cap does not truncate ({result['loop_tokens']} under "
            f"{result['request_cap']}), the velocity limit is unreachable, and 10000 "
            f"turns of pure loop reach ${result['unbounded_dollars']} against a "
            f"${result['monthly_cap']} monthly cap",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
