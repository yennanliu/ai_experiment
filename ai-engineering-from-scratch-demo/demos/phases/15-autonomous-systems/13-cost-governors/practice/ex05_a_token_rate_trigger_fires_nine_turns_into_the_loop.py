"""Exercise 5 — a token-rate trigger fires nine turns into the loop.

    Claude Code's `max_budget_usd` fires on session aggregate cost. Design a
    complementary velocity limit you would enforce externally. What triggers
    the cut-off, and what does re-enable look like?

Reading of the exercise: "complementary" is the constraint that matters -- a
second dollar threshold is not complementary to a dollar threshold, it is a
smaller one. So the design changes the denomination as well as the window,
and is then measured against the trajectory the shipped limit misses.

**ANSWER: tokens a minute, at 2x the observed baseline, over a short
window.** Baseline here is **5000** tokens a minute and the polling loop is
**16000**, so a threshold at **10000** sits between them. Over a 10-minute
rolling window it fires at turn **39** -- **9** turns after the loop begins
and **161** before `max_turns` -- with **$0.46** spent of the **$4.32** the
shipped stack allows. Re-enable is manual and carries the trajectory: the run
resumes only when a human has seen the window that tripped it.

**FINDING: tokens are the right denomination because dollars are derived.**
`DOLLARS_PER_KTOK` is a single module constant, so every dollar threshold in
the stack moves when a price does and no token threshold moves at all. That
is how the shipped `velocity_usd_per_min` ends up **83.3x** above what the
request cap permits -- a dollar threshold written against one price list and
never re-derived.

**FINDING: the window is the latency knob, and it is cheap.** At **10**
minutes the trigger fires on turn **39**; at **5**, turn **34**; at **2**,
turn **31** -- one turn after the loop's first full window. Shortening the
window costs nothing but sensitivity to a genuine burst, which is the trade a
2x-of-baseline threshold is already making.

**FINDING: re-enable has nowhere to live.** `Run` carries **5** fields, of
which `stopped_by` is a string, and `simulate` leaves the loop with no resume
path -- **0** fields record who cleared a stop or what they saw. The lesson's
own stack asks for exactly that as item **12**, and the simulator implements
the cut without the half that lets work continue.

Structure: `rate_at()` is the rolling token rate off the module's own cost
profile; `fires_at()` walks it to the first turn above the threshold.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "13-cost-governors"

FACTOR, WINDOWS = 2.0, (10.0, 5.0, 2.0)


def turns_per_minute(governor):
    return 60.0 / governor.seconds_per_turn


def baseline(ref, governor):
    return ref.NORMAL_TURN_TOKENS * turns_per_minute(governor)


def rate_at(ref, governor, turn, window_min):
    """Tokens a minute over the trailing window ending at `turn`."""
    span = int(window_min * turns_per_minute(governor))
    first = max(1, turn - span + 1)
    tokens = sum(ref.turn_cost(index) for index in range(first, turn + 1))
    return tokens / ((turn - first + 1) / turns_per_minute(governor)), span


def fires_at(ref, governor, window_min, factor=FACTOR):
    threshold = baseline(ref, governor) * factor
    for turn in range(1, 500):
        rate, span = rate_at(ref, governor, turn, window_min)
        if turn >= span and rate > threshold:
            return turn
    return None


def spent(ref, turn):
    tokens = sum(ref.turn_cost(index) for index in range(1, turn + 1))
    return round(tokens / 1000.0 * ref.DOLLARS_PER_KTOK, 2)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    governor = ref.Governor()
    triggers = [fires_at(ref, governor, window) for window in WINDOWS]
    dollars = ref.DOLLARS_PER_KTOK / 1000.0
    per_minute = turns_per_minute(governor)
    return {
        "baseline": baseline(ref, governor),
        "loop_rate": ref.LOOP_TURN_TOKENS * per_minute,
        "threshold": baseline(ref, governor) * FACTOR,
        "windows": list(WINDOWS),
        "triggers": triggers,
        "loop_starts": ref.LOOP_STARTS_AT,
        "turns_after_loop": triggers[0] - ref.LOOP_STARTS_AT,
        "max_turns": governor.max_turns,
        "turns_saved": governor.max_turns - triggers[0],
        "spent_at_trigger": spent(ref, triggers[0]),
        "spent_at_cap": spent(ref, governor.max_turns),
        "dollar_constants": 1,
        "shipped_headroom": round(governor.velocity_usd_per_min
                                  / (governor.max_tokens_per_request * dollars
                                     * per_minute), 1),
        "run_fields": len(ref.Run.__dataclass_fields__),
        "resume_fields": [name for name in ref.Run.__dataclass_fields__
                          if any(word in name for word in ("resume", "cleared", "by_whom"))],
        "simulate_resumes": "resume" in inspect.getsource(ref.simulate),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 2x the baseline in tokens a minute, firing on turn 39",
            all([result["baseline"] == 5000.0, result["loop_rate"] == 16000.0,
                 result["threshold"] == 10000.0, result["triggers"][0] == 39,
                 result["turns_after_loop"] == 9, result["turns_saved"] == 161,
                 result["spent_at_trigger"] == 0.46]),
            f"baseline {result['baseline']} tokens a minute against a loop at "
            f"{result['loop_rate']} puts the threshold at {result['threshold']}; it "
            f"fires on turn {result['triggers'][0]}, {result['turns_after_loop']} turns "
            f"into the loop and {result['turns_saved']} before max_turns, at "
            f"${result['spent_at_trigger']} of ${result['spent_at_cap']}",
        ),
        practice.Check(
            "FINDING: tokens are the right denomination because dollars are derived",
            all([result["dollar_constants"] == 1, result["shipped_headroom"] == 83.3]),
            f"{result['dollar_constants']} module constant converts tokens to dollars, "
            f"so every dollar threshold moves when a price does -- which is how the "
            f"shipped velocity limit ends up {result['shipped_headroom']}x above what "
            "the request cap permits",
        ),
        practice.Check(
            "FINDING: the window is the latency knob, and it is cheap",
            all([result["triggers"] == [39, 34, 31], result["windows"] == [10.0, 5.0, 2.0]]),
            f"windows of {result['windows']} minutes fire on turns "
            f"{result['triggers']} -- the shortest one turn after the loop's first full "
            "window, at no cost but sensitivity to a genuine burst",
        ),
        practice.Check(
            "FINDING: re-enable has nowhere to live",
            all([result["run_fields"] == 5, result["resume_fields"] == [],
                 not result["simulate_resumes"]]),
            f"Run carries {result['run_fields']} fields and "
            f"{len(result['resume_fields'])} record who cleared a stop or what they "
            "saw; simulate leaves the loop with no resume path, so the cut ships "
            "without the half that lets work continue",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
