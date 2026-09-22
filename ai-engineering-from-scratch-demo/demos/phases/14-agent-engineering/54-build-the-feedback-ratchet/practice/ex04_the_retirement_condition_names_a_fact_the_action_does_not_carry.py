"""Exercise 4 — the retirement condition names a fact the action does not carry.

    Define a retirement condition for a policy rule.

Reading of the exercise: the lab already writes one -- "Remove or revise
after 180 days without recurrence" -- so the work is making it evaluable, and
the first thing that shows is which half of that sentence the action can
actually check.

**ANSWER: the condition needs a last-seen date and a recurrence count, and
`RatchetAction` carries neither.** Given a promotion date, a supplied today
and a recurrence log, the rule is three comparisons: the window has elapsed,
no recurrence inside it, and the control has not blocked legitimate work.
Evaluated against a fixture, the policy rule from the pilot audit is **due**
at 180 days with **0** recurrences, and the same rule with **1** recurrence
**stays**.

**FINDING: the sentence embeds the number and drops the counter.** `promote`
formats `expires_after_days` into the string, so the number survives as text
while `Signal.frequency` -- the only recurrence figure in the model -- is
multiplied into `priority` and never stored. Of the action's **7** fields,
**0** hold a date and **0** hold a count.

**FINDING: retirement needs a clock, and the module has none.** Nothing
imports `time` or `datetime`, so the condition can only be evaluated by a
caller who supplies today. That is the right shape -- the same one Lesson 46
needed for review dates -- and it means the artifact records a policy whose
evaluation lives entirely outside it.

**FINDING: the lesson's fourth retirement reason cannot be measured at all.**
The docs list four: architecture changed, a lower-level invariant replaced
the rule, no recurrence in the window, and the control blocks legitimate work
more often than it prevents harm. The first three are checkable from a log;
the fourth needs a count of refusals that were wrong, which is a number
nothing in this system produces -- **3** of **4** are evaluable.

Structure: `retire()` is the condition; `CASES` exercises it against a fixed
today.
"""

from __future__ import annotations

import datetime as dt
import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "54-build-the-feedback-ratchet"
PROMOTED_ON = dt.date(2026, 3, 1)
TODAY = dt.date(2026, 9, 22)
POLICY_SIGNAL = ("pilot audit", "production write was attempted", 5, 1, "security", 180)
REASONS = ["architecture changed", "a lower-level invariant replaced it",
           "no recurrence in the window", "it blocks legitimate work more than it helps"]
MEASURABLE = REASONS[:3]

# (label, recurrences inside the window, wrong refusals recorded)
CASES = [("quiet", 0, 0), ("recurred", 1, 0), ("noisy", 0, 4)]


def retire(promoted_on, today, window_days, recurrences, wrong_refusals):
    """Due only when the window has elapsed, nothing recurred, and it is not noisy."""
    elapsed = (today - promoted_on).days
    if elapsed < window_days:
        return "hold"
    if recurrences:
        return "keep"
    return "review-for-noise" if wrong_refusals else "retire"


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    signal = ref.Signal(*POLICY_SIGNAL)
    action = ref.promote(signal)
    elapsed = (TODAY - PROMOTED_ON).days
    verdicts = {label: retire(PROMOTED_ON, TODAY, signal.expires_after_days,
                              recurred, wrong)
                for label, recurred, wrong in CASES}
    early = retire(PROMOTED_ON, dt.date(2026, 5, 1), signal.expires_after_days, 0, 0)
    module = inspect.getsource(ref)
    fields = list(ref.RatchetAction.__dataclass_fields__)
    return {
        "check": action.retirement_check, "window": signal.expires_after_days,
        "elapsed": elapsed, "verdicts": verdicts, "early": early,
        "embeds_number": str(signal.expires_after_days) in action.retirement_check,
        "fields": fields,
        "date_fields": [name for name in fields if "date" in name or "day" in name],
        "count_fields": [name for name in fields if "frequency" in name or "count" in name],
        "frequency_used": "signal.frequency" in inspect.getsource(ref.promote),
        "imports_clock": any(word in module for word in ("import time", "import datetime")),
        "reasons": len(REASONS), "measurable": len(MEASURABLE),
        "priority": action.priority,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the condition is three comparisons and the fixture retires one rule",
            all([result["window"] == 180, result["elapsed"] == 205,
                 result["verdicts"]["quiet"] == "retire",
                 result["verdicts"]["recurred"] == "keep",
                 result["early"] == "hold"]),
            f"at {result['elapsed']} days against a {result['window']}-day window the "
            f"quiet rule is {result['verdicts']['quiet']!r}, the recurring one is "
            f"{result['verdicts']['recurred']!r}, and the same rule evaluated before the "
            f"window closes is {result['early']!r}",
        ),
        practice.Check(
            "FINDING: the sentence embeds the number and drops the counter",
            all([result["embeds_number"] is True, len(result["fields"]) == 7,
                 result["date_fields"] == [], result["count_fields"] == [],
                 result["frequency_used"] is True]),
            f"the retirement check reads {result['check']!r}; the number survives as text "
            f"while frequency is multiplied into priority ({result['priority']}) and never "
            f"stored -- {len(result['date_fields'])} of the {len(result['fields'])} fields "
            "hold a date and none holds a count",
        ),
        practice.Check(
            "FINDING: retirement needs a clock and the module has none",
            all([result["imports_clock"] is False]),
            "nothing imports time or datetime, so the condition can only be evaluated by a "
            "caller supplying today -- the right shape, and one that puts the evaluation "
            "entirely outside the artifact",
        ),
        practice.Check(
            "FINDING: the fourth retirement reason cannot be measured",
            all([result["reasons"] == 4, result["measurable"] == 3,
                 result["verdicts"]["noisy"] == "review-for-noise"]),
            f"{result['measurable']} of the {result['reasons']} documented reasons are "
            "checkable from a log; 'blocks legitimate work more than it helps' needs a "
            f"count of wrong refusals, which is why the noisy case returns "
            f"{result['verdicts']['noisy']!r} rather than a verdict",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
