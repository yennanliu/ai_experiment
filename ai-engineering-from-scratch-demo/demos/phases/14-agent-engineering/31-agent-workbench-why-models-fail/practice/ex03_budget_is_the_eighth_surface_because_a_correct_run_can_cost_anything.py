"""Exercise 3 — budget is the eighth surface, because a correct run can cost anything.

    Add an eighth surface for your own product. Justify why it does not
    collapse into one of the existing seven.

Reading of the exercise: the justification has to be a demonstration, not an
argument -- a surface earns its place when there is a run that all seven
existing surfaces accept and it refuses. So the candidate is **budget**: a
declared ceiling on steps, tool calls and tokens, checked continuously and
failing closed. The test is whether a run can be in scope, logged, verified,
reviewed and handed off, and still be one nobody would ship.

**ANSWER: budget, and it refuses a run the other seven accept 7 of 7.** A
stub that stays inside `allowed_files`, runs its tests, passes acceptance
and writes a handoff note -- while taking **412** steps for a task whose
plan was **6** -- passes every existing surface and fails a budget of
**60**. **68.7x** over plan, **0** other refusals.

**FINDING: it does not collapse into scope, because scope is about *where*.**
The over-budget run touches **2** files, both allowed, and **0** forbidden
ones, so an authorization policy over paths has nothing to object to.
Widening scope to cover cost would mean an ACL keyed on a number that is not
known until the run ends, which is a different primitive: a quota, not a
permission.

**FINDING: it does not collapse into verification, because verification runs
once at the end.** The shipped gate is triggered on task close, and by then
**412** steps have been spent -- the check can report the overrun and cannot
prevent it. Budget has to be a *trigger on every step*, which is the same
distinction exercise 4 measured between scope at position 1 and review at
position 4.

**FINDING: it does not collapse into feedback either, and that is the
closest call.** Feedback already records every invocation, so the *data* for
a budget is there: the over-budget run logs all **412** steps. What feedback
lacks is a ceiling and a refusal -- it is a queue, and a budget is a function
over that queue with the authority to stop the worker. **1** of the two
halves ships.

Structure: `budget_gate()` is the eighth surface; `gauntlet()` runs one job
past all eight and reports which refuse.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "31-agent-workbench-why-models-fail"
PLANNED_STEPS, CEILING, ACTUAL = 6, 60, 412


def demo_task(ref):
    return ref.RepoTask(
        description="add input validation to /signup and a passing test",
        allowed_files=["app.py", "test_app.py"],
        forbidden_files=["README.md", "scripts/release.sh"],
        acceptance=["test_app.py::test_signup_rejects_short_password passes"])


def expensive_run(ref, task):
    """In scope, tested, verified, reviewed, handed off -- and 412 steps long."""
    result = ref.stub_agent(task, surfaces=list(ref.WORKBENCH_SURFACES))
    result.notes.append(f"took {ACTUAL} steps against a {PLANNED_STEPS}-step plan")
    return result


def scope_gate(task, result, steps):
    del steps
    return not [f for f in result.files_touched if f not in task.allowed_files]


def instructions_gate(task, result, steps):
    del task, steps
    return "instructions" in result.surfaces_present


def state_gate(task, result, steps):
    del task, steps
    return "state" in result.surfaces_present


def feedback_gate(task, result, steps):
    del task
    return result.tests_run and steps > 0


def verification_gate(task, result, steps):
    del task, steps
    return result.actually_passing


def review_gate(task, result, steps):
    del steps
    return not [f for f in result.files_touched if f in task.forbidden_files]


def handoff_gate(task, result, steps):
    del task, steps
    return "handoff" in result.surfaces_present


def budget_gate(task, result, steps, ceiling=CEILING):
    """The eighth surface: a ceiling checked per step, failing closed."""
    del task, result
    return steps <= ceiling


SURFACES = (("instructions", instructions_gate), ("state", state_gate),
            ("scope", scope_gate), ("feedback", feedback_gate),
            ("verification", verification_gate), ("review", review_gate),
            ("handoff", handoff_gate), ("budget", budget_gate))


def gauntlet(task, result, steps):
    return {name: gate(task, result, steps) for name, gate in SURFACES}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    task = demo_task(ref)
    run = expensive_run(ref, task)
    verdicts = gauntlet(task, run, ACTUAL)
    cheap = gauntlet(task, run, PLANNED_STEPS)
    return {
        "surfaces": len(SURFACES), "verdicts": verdicts,
        "refused_by": [name for name, ok in verdicts.items() if not ok],
        "passed_existing": sum(ok for name, ok in verdicts.items()
                               if name != "budget"),
        "existing": len(SURFACES) - 1,
        "overrun": round(ACTUAL / PLANNED_STEPS, 1),
        "ceiling": CEILING, "actual": ACTUAL,
        "cheap_refusals": [name for name, ok in cheap.items() if not ok],
        "touched": run.files_touched,
        "allowed": task.allowed_files,
        "forbidden_touched": [f for f in run.files_touched
                              if f in task.forbidden_files],
        "logged": ACTUAL, "tests_run": run.tests_run,
        "verified_at_close": run.actually_passing,
        "shipped_surfaces": len(ref.WORKBENCH_SURFACES),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: budget refuses a run the other seven accept 7 of 7",
            all([result["surfaces"] == 8, result["refused_by"] == ["budget"],
                 result["passed_existing"] == 7, result["existing"] == 7,
                 result["overrun"] == 68.7, result["cheap_refusals"] == []]),
            f"a run that stays in scope, runs its tests, passes acceptance and hands off "
            f"-- while taking {result['actual']} steps against a plan of "
            f"{PLANNED_STEPS} -- passes {result['passed_existing']}/"
            f"{result['existing']} existing surfaces and is refused by "
            f"{result['refused_by']}. {result['overrun']}x over plan",
        ),
        practice.Check(
            "FINDING: it does not collapse into scope, because scope is about where",
            all([result["touched"] == result["allowed"],
                 result["forbidden_touched"] == [],
                 result["verdicts"]["scope"] is True]),
            f"the over-budget run touches {result['touched']}, which is exactly "
            f"{result['allowed']}, and {len(result['forbidden_touched'])} forbidden "
            "files. An authorization policy over paths has nothing to object to; a quota "
            "is keyed on a number nobody knows until the run ends",
        ),
        practice.Check(
            "FINDING: it does not collapse into verification, which runs once at close",
            all([result["verified_at_close"] is True,
                 result["verdicts"]["verification"] is True,
                 result["actual"] == 412]),
            f"the shipped gate is triggered on task close and returns "
            f"{result['verified_at_close']}, by which point {result['actual']} steps are "
            "spent. It can report the overrun and cannot prevent it -- budget has to be a "
            "trigger on every step",
        ),
        practice.Check(
            "FINDING: it does not collapse into feedback, and that is the closest call",
            all([result["logged"] == 412, result["tests_run"] is True,
                 result["verdicts"]["feedback"] is True,
                 result["shipped_surfaces"] == 7]),
            f"feedback records all {result['logged']} invocations and returns "
            f"{result['verdicts']['feedback']}, so the data a budget needs is already "
            "there. What it lacks is a ceiling and the authority to stop the worker: a "
            "queue is half of a quota",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
