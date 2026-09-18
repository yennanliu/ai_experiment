"""Exercise 4 — there is no failure to hook, because nothing fails.

    Build an error-recovery hook: on action failure (button not found), what does
    the agent do next?

Reading of the exercise: the hook needs somewhere to attach, so the first thing
checked is what the lesson's own loop does when an action does not work -- and
the answer is that it does not notice. The ladder is then designed with a budget,
because Exercise 3's arithmetic makes an unbounded retry the fastest way to
exhaust a context.

**ANSWER: a five-rung ladder with a per-rung cap and a global budget.** Re-ground
from a fresh screenshot; scroll and retry; `screenshot_region` on the expected
area; try an alternative descriptor; back out one step and re-plan. Then
`done(success=False)` with the trace, rather than a sixth attempt.

**FINDING: the lesson's loop cannot fail.** `apply_action` returns an unchanged
copy for any descriptor it does not recognise -- clicking "NoSuchButton" leaves
the page at `home` and raises nothing -- and `run_task` has **0** branches, so it
executes all 2 planned steps and reports the click as `click`. The only failure
signal is the *final page*, computed after the plan is exhausted.

**FINDING: which means failure is detected 2.0 steps late on average.** On a
five-step plan whose first action silently no-ops, the remaining four run against
the wrong state before the comparison happens. A hook needs a per-action
postcondition -- did the observation change in the way this action predicted --
and the simulator has no such predicate anywhere.

**FINDING: and an unbounded ladder is the fastest way to exhaust the context.**
Each rung costs a fresh screenshot, **10,549** tokens, so five rungs on one
failed action is **52,745** -- **5x** a successful step, and **40.2%** of a 128k
window. Exercise 3's whole budget is 44,696; one uncapped recovery exceeds it.

Structure: `bogus_run` feeds the lesson's own `run_task` an unrecognised
descriptor, `LADDER` is the design, and `ladder_cost` prices it against Exercise
3's screenshot figure.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "25-multimodal-agents-computer-use"
SCREEN, PATCH = (1920, 1080), 14
CONTEXT = 131072
LADDER = ("re-ground from a fresh screenshot", "scroll and retry",
          "screenshot_region on the expected area", "try an alternative descriptor",
          "back out one step and re-plan")
PLAN_STEPS = 5
COMPRESSED_BUDGET = 44_696          # Exercise 3


def screenshot_tokens(screen=SCREEN, patch=PATCH):
    return (screen[0] // patch) * (screen[1] // patch)


def ladder_cost(rungs=len(LADDER)):
    return rungs * screenshot_tokens()


def bogus_run(ref):
    task = ref.Task(
        goal="click a button that does not exist",
        plan=[{"action": "click", "x": 1, "y": 1, "element_desc": "NoSuchButton"},
              {"action": "done", "success": True, "explanation": "e"}],
        expected_page="confirmation")
    return ref.run_task(task)


def late_by(steps=PLAN_STEPS):
    """Mean steps between a silent failure and the end-of-plan check."""
    return round(sum(steps - position for position in range(1, steps + 1)) / steps, 1)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    result = bogus_run(ref)
    cost = ladder_cost()
    return {
        "ladder": LADDER, "rungs": len(LADDER),
        "final_page": result["final_page"], "success": result["success"],
        "steps_executed": len(result["trace"]),
        "raised": False,
        "branches": inspect.getsource(ref.run_task).count("if "),
        "postcondition": "expected" in inspect.getsource(ref.apply_action),
        "late_by": late_by(),
        "plan_steps": PLAN_STEPS,
        "frame": screenshot_tokens(),
        "ladder_tokens": cost,
        "vs_step": cost // screenshot_tokens(),
        "context_pct": round(cost / CONTEXT * 100, 1),
        "vs_budget": round(cost / COMPRESSED_BUDGET, 2),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a five-rung ladder with a per-rung cap and a global budget",
            all([result["rungs"] == 5, result["ladder"][0].startswith("re-ground"),
                 result["ladder"][-1].startswith("back out")]),
            f"{list(result['ladder'])}, then done(success=False) with the trace rather than a "
            f"sixth attempt. {result['rungs']} rungs, each capped, under one global step "
            "budget -- the cap is the design, not the ladder",
        ),
        practice.Check(
            "FINDING: the lesson's loop cannot fail",
            all([result["final_page"] == "home", not result["success"],
                 result["steps_executed"] == 2, not result["raised"],
                 result["branches"] == 0]),
            f"clicking 'NoSuchButton' leaves the page at {result['final_page']!r} and raises "
            f"nothing; run_task has {result['branches']} branches and executes all "
            f"{result['steps_executed']} planned steps, recording the click as a click. The "
            "only failure signal is the final page, compared after the plan is exhausted",
        ),
        practice.Check(
            "FINDING: failure is detected 2.0 steps late on average",
            all([result["late_by"] == 2.0, not result["postcondition"],
                 result["plan_steps"] == PLAN_STEPS]),
            f"on a {result['plan_steps']}-step plan the mean gap between a silent no-op and "
            f"the end-of-plan check is {result['late_by']} steps, all of them run against the "
            "wrong state. A hook needs a per-action postcondition -- did the observation "
            "change as this action predicted -- and no such predicate exists in apply_action",
        ),
        practice.Check(
            "FINDING: an unbounded ladder is the fastest way to exhaust the context",
            all([result["frame"] == 10_549, result["ladder_tokens"] == 52_745,
                 result["vs_step"] == 5, result["context_pct"] == 40.2,
                 result["vs_budget"] == 1.18]),
            f"each rung costs a fresh screenshot at {result['frame']:,} tokens, so "
            f"{result['rungs']} rungs on one failed action is {result['ladder_tokens']:,} -- "
            f"{result['vs_step']}x a successful step and {result['context_pct']}% of a 128k "
            f"window. Exercise 3's entire fifty-step budget is {COMPRESSED_BUDGET:,}; one "
            f"uncapped recovery is {result['vs_budget']}x that",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
