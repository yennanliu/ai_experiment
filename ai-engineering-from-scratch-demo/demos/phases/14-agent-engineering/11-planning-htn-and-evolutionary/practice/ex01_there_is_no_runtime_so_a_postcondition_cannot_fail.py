"""Exercise 1 — there is no runtime, so a postcondition cannot fail.

    Extend the HTN planner with backtracking: when an operator's postcondition
    fails at runtime, roll back and try the next method.

Reading of the exercise: `Operator.apply` adds `effects_add` unconditionally
and nothing ever compares the result against the world, so the failure the
exercise describes is not a state this code can reach. Backtracking therefore
needs two additions, not one -- an executor whose effects can fail, and the
alternative-method loop `plan` does not have. Both are built here and both
are measured against the shipped behaviour.

**ANSWER: an executor plus method backtracking.** Given a task with **2**
applicable methods where the first one's `run_tests` does not deliver
`tests_passing`, the shipped planner returns `None`. The backtracking planner
rolls back **2** applied operators and returns the second method's plan,
**4** operators long, after **1** rollback.

**FINDING: the shipped planner considers one method.** `plan` takes
`applicable[0]` and never revisits it, so with **2** applicable methods the
expansion that fails ends the search. Even at plan time -- before any
execution -- a second method that would have worked is never tried.

**FINDING: there is nothing to check a postcondition against.** `Operator`
has **4** fields and **0** of them names a postcondition or a verifier, and
`apply` unions `effects_add` into the state whatever happened. The planner's
model of the world is the planner's own arithmetic.

**FINDING: failure has one value and five causes.** `plan` returns `None` for
exceeding `max_depth`, for an unknown task, for an operator whose
preconditions do not hold, for an LLM that declines, and for an LLM
suggestion naming something unknown. All **5** arrive as `None`, so a caller
that wants to backtrack cannot tell a retryable failure from an impossible
one.

Structure: `execute()` is the missing runtime; `plan_with_backtracking()`
adds the loop over `applicable` and the rollback.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "11-planning-htn-and-evolutionary"
FLAKY = "run_tests"


def world(ref):
    names = (("open_editor", ("logged_in",), ("editor_open",)),
             ("write_tests", ("editor_open",), ("tests_written",)),
             ("run_tests", ("tests_written",), ("tests_passing",)),
             ("run_tests_verbose", ("tests_written",), ("tests_passing",)),
             ("open_pr", ("tests_passing",), ("pr_open",)))
    operators = {name: ref.Operator(name, pre, add) for name, pre, add in names}
    methods = {"ship": [
        ref.Method("m1", "ship", ("logged_in",),
                   ("open_editor", "write_tests", "run_tests", "open_pr")),
        ref.Method("m2", "ship", ("logged_in",),
                   ("open_editor", "write_tests", "run_tests_verbose", "open_pr"))]}
    return operators, methods


def execute(operators, plan, state, flaky=FLAKY):
    """The runtime the lesson does not have: an effect that may not land."""
    current, applied = set(state), []
    for step in plan:
        operator = operators[step]
        if not operator.applicable(current):
            return None, applied
        current = operator.apply(current)
        applied.append(step)
        if step == flaky:
            current -= set(operator.effects_add)
        if not set(operator.effects_add) <= current:
            return None, applied
    return current, applied


def plan_with_backtracking(ref, operators, methods, task, state):
    rollbacks, plans = 0, []
    for method in methods[task]:
        planner = ref.HTNPlanner(operators, {task: [method]}, ref.ScriptedLLM({}))
        candidate = planner.plan(task, state)
        if candidate is None:
            continue
        final, applied = execute(operators, candidate, state)
        plans.append(candidate)
        if final is not None:
            return {"plan": candidate, "rollbacks": rollbacks, "rolled_back": []}
        rollbacks += 1
        rolled = applied
    return {"plan": None, "rollbacks": rollbacks, "rolled_back": rolled}


def failure_causes(ref, operators, methods):
    empty = ref.ScriptedLLM({})
    bad = ref.ScriptedLLM({"unknown_task": ("no_such_operator",)})
    deep = ref.HTNPlanner(operators, methods, empty)
    return {
        "max_depth": deep.plan("ship", {"logged_in"}, depth=99),
        "unknown_task": ref.HTNPlanner(operators, methods, empty).plan("nope", set()),
        "precondition": ref.HTNPlanner(operators, methods, empty).plan(
            "open_pr", {"logged_in"}),
        "llm_declined": ref.HTNPlanner(operators, {}, empty).plan("ship", {"logged_in"}),
        "llm_unknown": ref.HTNPlanner(operators, {}, bad).plan(
            "unknown_task", {"logged_in"}),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    operators, methods = world(ref)
    state = {"logged_in"}
    shipped = ref.HTNPlanner(operators, methods, ref.ScriptedLLM({}))
    plan = shipped.plan("ship", state)
    ran, applied = execute(operators, plan, state)
    backtracked = plan_with_backtracking(ref, operators, methods, "ship", state)
    return {
        "shipped_plan": plan, "shipped_run": ran, "applied": applied,
        "applicable": len([m for m in methods["ship"] if m.applicable(state)]),
        "chosen": methods["ship"][0].name,
        "backtracked": backtracked["plan"], "rollbacks": backtracked["rollbacks"],
        "operator_fields": list(ref.Operator.__dataclass_fields__),
        "postcondition_fields": [f for f in ref.Operator.__dataclass_fields__
                                 if "post" in f or "verify" in f],
        "causes": failure_causes(ref, operators, methods),
    }


def verify(result):
    causes = result["causes"]
    return [
        practice.Check(
            "ANSWER: the shipped run dies at the flaky operator, backtracking recovers",
            all([result["shipped_run"] is None, result["applied"] == [
                "open_editor", "write_tests", "run_tests"],
                 result["backtracked"] == ["open_editor", "write_tests",
                                           "run_tests_verbose", "open_pr"],
                 result["rollbacks"] == 1]),
            f"the shipped plan {result['shipped_plan']} executes "
            f"{len(result['applied'])} operators and then fails, because run_tests does "
            f"not deliver its effect. Backtracking rolls back once and returns the "
            f"second method's {len(result['backtracked'])}-operator plan",
        ),
        practice.Check(
            "FINDING: the shipped planner considers one method",
            all([result["applicable"] == 2, result["chosen"] == "m1",
                 result["shipped_plan"][2] == FLAKY]),
            f"{result['applicable']} methods are applicable and plan takes "
            f"applicable[0] -- {result['chosen']} -- without ever revisiting it. Even at "
            "plan time, before any execution, a second method that would have worked is "
            "not tried",
        ),
        practice.Check(
            "FINDING: there is nothing to check a postcondition against",
            all([len(result["operator_fields"]) == 4,
                 result["postcondition_fields"] == [],
                 result["operator_fields"] == ["name", "preconditions", "effects_add",
                                               "effects_remove"]]),
            f"Operator carries {result['operator_fields']} -- "
            f"{len(result['postcondition_fields'])} of them a postcondition or a "
            "verifier -- and apply unions effects_add into the state whatever happened. "
            "The planner's model of the world is its own arithmetic",
        ),
        practice.Check(
            "FINDING: failure has one value and five causes",
            all([set(causes) == {"max_depth", "unknown_task", "precondition",
                                 "llm_declined", "llm_unknown"},
                 all(value is None for value in causes.values())]),
            f"plan returns None for all {len(causes)} of {sorted(causes)}. A caller that "
            "wants to backtrack cannot tell a retryable failure from an impossible one, "
            "which is why the backtracking loop here retries every method rather than "
            "the ones worth retrying",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
