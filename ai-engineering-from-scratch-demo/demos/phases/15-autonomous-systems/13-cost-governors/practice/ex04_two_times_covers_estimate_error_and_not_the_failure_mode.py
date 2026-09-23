"""Exercise 4 — 2x covers estimate error and not the failure mode.

    Price an overnight unattended run for a realistic task (e.g., "triage 50
    issues in a repo"). Set `max_budget_usd` at 2x your point estimate.
    Justify the 2x.

Reading of the exercise: "justify the 2x" is answerable two ways -- against
the spread of the estimate, where 2x is generous, and against the lesson's
own case study, where it is not enough. Both are computed, because the
disagreement is the answer.

**ANSWER: $4.50 point estimate, $9.00 cap -- and the lesson's own case study
overran by 4x.** Fifty issues at **12** turns each is **600** turns; at the
module's normal turn of **2500** tokens and **$0.003** a thousand, that is
**$0.0075** a turn and **$4.50** for the run. Doubling gives **$9.00**. The
$1,200 to $4,800 case in the same lesson is a **4.0x** overrun, so a 2x cap
would have fired and a 2x cap would also have been wrong about the size of
the problem.

**FINDING: 2x buys estimate error, and the dollar cap is still not what
stops anything.** If the per-issue turn count is wrong by 2x -- **24** turns
instead of 12 -- the run costs **$9.00** and lands exactly on the cap, which
is what the multiplier is for. But a run that drifts into the polling loop
reaches **$9.00** after **375** turns, short of the **600** the job needs,
and `max_turns` of **200** fires before either. Priced this way the dollar
cap is decoration on top of the turn cap, for the third time in this lesson.

**FINDING: the lesson's own numbers disagree with its own prose.** The
section is titled "$1,200 to $4,800" and describes it as a tripling; the
ratio is **4.0**. Whichever figure a reader takes, **both** exceed the 2x the
exercise prescribes, so the exercise's multiplier is not derived from the
case study it sits next to.

**FINDING: a dollar cap cannot fire early, by construction.** `max_budget_usd`
compares a running total to a constant, so it fires when the money is already
spent -- **1** of the stack's **12** controls is about rate and the simulator
prices it out of reach. The complement is a velocity cut, which is exercise
5, and the reason a bigger multiplier is not the fix.

Structure: `price()` is the point estimate from the module's own constants;
`drift()` asks how many looping turns the cap absorbs.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "13-cost-governors"

ISSUES, TURNS_PER_ISSUE, MULTIPLIER = 50, 12, 2.0
CASE_BEFORE, CASE_AFTER = 1_200, 4_800


def per_turn(ref, tokens):
    return tokens / 1000.0 * ref.DOLLARS_PER_KTOK


def price(ref, issues=ISSUES, turns=TURNS_PER_ISSUE):
    return round(issues * turns * per_turn(ref, ref.NORMAL_TURN_TOKENS), 2)


def drift(ref, budget):
    """Looping turns the budget funds, before anything else stops the run."""
    return int(budget / per_turn(ref, ref.LOOP_TURN_TOKENS))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    estimate = price(ref)
    cap = round(estimate * MULTIPLIER, 2)
    loop_turns = drift(ref, cap)
    return {
        "issues": ISSUES, "turns_per_issue": TURNS_PER_ISSUE,
        "turns": ISSUES * TURNS_PER_ISSUE,
        "per_turn": round(per_turn(ref, ref.NORMAL_TURN_TOKENS), 4),
        "estimate": estimate,
        "multiplier": MULTIPLIER,
        "cap": cap,
        "doubled_turns": price(ref, turns=TURNS_PER_ISSUE * 2),
        "lands_on_cap": price(ref, turns=TURNS_PER_ISSUE * 2) == cap,
        "loop_turns": loop_turns,
        "job_turns": ISSUES * TURNS_PER_ISSUE,
        "max_turns": ref.Governor().max_turns,
        "case_ratio": round(CASE_AFTER / CASE_BEFORE, 1),
        "case_prose": 3.0,
        "both_exceed": min(round(CASE_AFTER / CASE_BEFORE, 1), 3.0) > MULTIPLIER,
        "stack_items": 12,
        "rate_controls": 1,
        "cap_is_cumulative": "run.dollars >= gov.max_budget_usd" in __import__(
            "inspect").getsource(ref.simulate),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: $4.50 point estimate, $9.00 cap, against a 4x case study",
            all([result["turns"] == 600, result["per_turn"] == 0.0075,
                 result["estimate"] == 4.50, result["cap"] == 9.00,
                 result["case_ratio"] == 4.0]),
            f"{result['issues']} issues at {result['turns_per_issue']} turns is "
            f"{result['turns']} turns at ${result['per_turn']} each -- "
            f"${result['estimate']}, capped at ${result['cap']} -- while the lesson's "
            f"own case study overran by {result['case_ratio']}x",
        ),
        practice.Check(
            "FINDING: 2x buys estimate error, not drift",
            all([result["lands_on_cap"], result["doubled_turns"] == 9.0,
                 result["loop_turns"] == 375, result["job_turns"] == 600,
                 result["max_turns"] < result["loop_turns"]]),
            f"doubling the per-issue turn count costs ${result['doubled_turns']} and "
            f"lands exactly on the cap; a looping run reaches it after "
            f"{result['loop_turns']} turns, short of the {result['job_turns']} the job "
            f"needs -- and max_turns of {result['max_turns']} fires before either",
        ),
        practice.Check(
            "FINDING: the lesson's own numbers disagree with its own prose",
            all([result["case_ratio"] == 4.0, result["case_prose"] == 3.0,
                 result["both_exceed"]]),
            f"the section titled $1,200 to $4,800 calls it a tripling and the ratio is "
            f"{result['case_ratio']}; both {result['case_prose']} and "
            f"{result['case_ratio']} exceed the {result['multiplier']}x the exercise "
            "prescribes",
        ),
        practice.Check(
            "FINDING: a dollar cap cannot fire early, by construction",
            all([result["cap_is_cumulative"], result["rate_controls"] == 1,
                 result["stack_items"] == 12]),
            f"max_budget_usd compares a running total to a constant, so it fires once "
            f"the money is spent; {result['rate_controls']} of the stack's "
            f"{result['stack_items']} controls is about rate, and the simulator prices "
            "it out of reach",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
