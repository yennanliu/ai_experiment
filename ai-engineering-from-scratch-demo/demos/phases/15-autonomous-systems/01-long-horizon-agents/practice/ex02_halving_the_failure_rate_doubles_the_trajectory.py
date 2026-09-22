"""Exercise 2 — halving the failure rate doubles the trajectory.

    Set per-step reliability to 0.995. What trajectory length still clears
    50% end-to-end reliability? Compare to 0.99 and 0.999. Per-step
    reliability has exponential consequences at scale.

Reading of the exercise: the literal answer is already printed -- the lesson's
"max trajectory length for 50% end-to-end success" block runs
`max_steps_for_target` over all three rates. So the exercise is not asking for
a number the module withholds; it is asking what the three numbers mean next
to each other, and the answer is a law, not a table.

**ANSWER: 138 steps at 0.995, against 68 at 0.99 and 692 at 0.999.** Halving
the per-step failure rate from 1% to 0.5% multiplies the trajectory by
**2.03**; cutting it fivefold again multiplies by **5.01**. The allowed length
tracks `1 / (1 - p)`, not `p`.

**FINDING: the closed form is `ln 2 / (1 - p)`, and it is accurate to a step.**
It predicts **69.3**, **138.6** and **693.1** against the module's 68, 138 and
692 -- worst error **1.31** steps, all of it the floor. "Per-step reliability"
is therefore the wrong axis to plan on: on the failure rate the relationship is
a straight proportion, and 0.99 to 0.999 stops looking like a rounding change.

**FINDING: 50% is the lesson's own bottom flag.** `reliability_compounding`
labels anything under 0.5 "coin flip or worse" and reserves "ok" for 0.95 and
up. The lengths that clear *that* bar are **5**, **10** and **51** steps -- a
constant **13.5x** shorter, since the ratio is `ln 0.5 / ln 0.95` and the
per-step rate cancels. The exercise's bar is the line below which the lesson
refuses to call anything shippable.

**FINDING: the shipped table's only 0.995 row sits 62 steps past the answer.**
Of the **8** cases it prints, one uses 0.995, at **200** steps, scoring
**36.7%** and flagged "coin flip or worse". And the guard is one-sided:
`per_step >= 1.0` returns the sentinel **1000000000**, while `per_step = 0.0`
raises ValueError out of `math.log` rather than returning 0.

Structure: `lengths()` asks the module for both bars; `closed_form()` is the
hyperbola the three numbers are samples of.
"""

from __future__ import annotations

import inspect
import math
import re

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "01-long-horizon-agents"

RATES = (0.99, 0.995, 0.999)
COIN_FLIP, OK = 0.50, 0.95      # the lesson's own bottom and top flag thresholds


def lengths(ref, target):
    return [ref.max_steps_for_target(rate, target) for rate in RATES]


def closed_form(target=COIN_FLIP):
    """`p**n = target` solved for n, linearised: ln(target) / -(1 - p)."""
    return [-math.log(target) / (1 - rate) for rate in RATES]


def refused(ref, rate, steps):
    try:
        return ref.max_steps_for_target(rate, steps)
    except ValueError as exc:
        return type(exc).__name__


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    coin, ok = lengths(ref, COIN_FLIP), lengths(ref, OK)
    exact = closed_form()
    source = inspect.getsource(ref.reliability_compounding)
    cases = [(float(p), int(n)) for p, n in re.findall(r"\((0\.\d+), (\d+)\)", source)]
    only = [case for case in cases if case[0] == 0.995]
    return {
        "coin": coin,
        "ok": ok,
        "growth": [round(coin[i + 1] / coin[i], 2) for i in range(len(coin) - 1)],
        "closed_form": [round(value, 1) for value in exact],
        "worst_error": round(max(abs(exact[i] - coin[i]) for i in range(len(coin))), 2),
        "bar_ratio": [round(coin[i] / ok[i], 1) for i in range(len(coin))],
        "exact_ratio": round(math.log(COIN_FLIP) / math.log(OK), 1),
        "flags": re.findall(r'flag = "([^"]+)"', source),
        "cases": len(cases),
        "only_995": only,
        "only_995_score": round(ref.end_to_end_reliability(*only[0]), 3) if only else None,
        "sentinel": ref.max_steps_for_target(1.0, COIN_FLIP),
        "at_zero": refused(ref, 0.0, COIN_FLIP),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 138 steps at 0.995, against 68 at 0.99 and 692 at 0.999",
            all([result["coin"] == [68, 138, 692], result["growth"] == [2.03, 5.01]]),
            f"the three rates clear 50% end-to-end at {result['coin']} steps, growing "
            f"{result['growth']}x -- the allowed length tracks 1/(1-p), so halving the "
            "failure rate doubles the trajectory",
        ),
        practice.Check(
            "FINDING: ln 2 / (1 - p) reproduces all three within a step",
            all([result["closed_form"] == [69.3, 138.6, 693.1],
                 result["worst_error"] <= 1.5]),
            f"the closed form gives {result['closed_form']} against the module's "
            f"{result['coin']}, worst error {result['worst_error']} steps -- the "
            "failure rate is the axis the relationship is linear on",
        ),
        practice.Check(
            "FINDING: 50% is the lesson's own bottom flag",
            all([result["ok"] == [5, 10, 51], result["bar_ratio"] == [13.6, 13.8, 13.6],
                 result["exact_ratio"] == 13.5, result["flags"][0] == "coin flip or worse",
                 result["flags"][-1] == "ok"]),
            f"the same rates clear the lesson's 'ok' bar of 0.95 at only {result['ok']} "
            f"steps, {result['exact_ratio']}x shorter for every rate, so the exercise's "
            f"bar is where the lesson starts printing {result['flags'][0]!r}",
        ),
        practice.Check(
            "FINDING: the only 0.995 row printed sits 62 steps past the answer",
            all([result["cases"] == 8, result["only_995"] == [(0.995, 200)],
                 result["only_995_score"] == 0.367, result["sentinel"] == 10**9,
                 result["at_zero"] == "ValueError"]),
            f"{len(result['only_995'])} of the {result['cases']} shipped cases uses "
            f"0.995, at 200 steps, scoring {result['only_995_score']:.1%}; the guard "
            f"returns {result['sentinel']} above 1.0 and {result['at_zero']} at 0.0",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
