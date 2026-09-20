"""Exercise 2 — the smallest change is one more input to the planner.

    Add a replanner node that fires if any worker returns an error. What is the
    smallest change to ReWOO that makes it Plan-and-Execute?

Reading of the exercise: ReWOO's three arrows are `question -> plan`,
`plan -> evidence`, `(question, plan, evidence) -> answer`. Plan-and-Execute
adds exactly one: `evidence -> plan`. So the smallest change is a parameter,
not a node -- `plan_for(question)` becomes `plan_for(question, evidence)` --
and the replanner is the loop that edge closes. Everything measured here runs
through the lesson's own `run_rewoo`, called more than once.

**ANSWER: one extra planner input, and the replanner fires once.** The
shipped `plan_for` takes **1** input besides `self`; the replanning one takes
**2**. A plan whose second step names an unregistered tool yields
`error: unknown tool 'web_search'` on pass **1**; the replanner fires, pass
**2** has **0** errors, and the answer is the demo's.

**FINDING: the trigger is a prefix, and it misses half the failures.**
`fake_search` answers an unknown query with `no result for ...`, which is not
prefixed `error:`. That plan finishes with **0** errors, fires the replanner
**0** times, and the solver stitches the miss into the final answer. **1** of
the **2** failure kinds reaches the trigger.

**FINDING: the replan needs a bound the lesson gives no reason to add.**
`ScriptedPlanner.plan_for` ignores `question` and returns the *same object*
every call -- **True** under `is` -- so an unbounded replanner on the shipped
planner is an infinite loop. Against a tool that always raises, the bounded
version burns its cap of **2** replans and still ends with **1** error,
because `dispatch` catches bare `Exception` and cannot say which errors a new
plan could fix.

**FINDING: every replan re-pays the planner prompt.** `planner_chars` is
**156** for one pass and **468** across three. The paper's 5x is per plan; a
task that replans twice has paid three planner prompts for one answer.

Structure: `plan_and_execute` is the loop; `ReplanningPlanner` is the one
extra edge.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "02-rewoo-plan-and-execute"
QUESTION = "What is the population of the capital of France, rounded?"
TEMPLATE = "The capital of France is {E1}; rounded population is {E3}."
BROKEN = (("E1", "search", {"query": "capital of France"}),
          ("E2", "web_search", {"query": "population of #E1"}),
          ("E3", "round_million", {"text": "#E2"}))
FIXED = (("E1", "search", {"query": "capital of France"}),
         ("E2", "search", {"query": "population of #E1"}),
         ("E3", "round_million", {"text": "#E2"}))
MISS = (("E1", "search", {"query": "capital of Atlantis"}),
        ("E2", "search", {"query": "population of #E1"}),
        ("E3", "round_million", {"text": "#E2"}))
ALWAYS_RAISES = (("E1", "search", {"query": "capital of France"}),
                 ("E2", "flaky", {"query": "population of #E1"}),
                 ("E3", "round_million", {"text": "#E2"}))


class ReplanningPlanner:
    """ReWOO's planner with the one extra input Plan-and-Execute needs."""

    def __init__(self, ref, plans):
        self.plans = [ref.Plan(steps=[ref.PlanStep(i, t, dict(a)) for i, t, a in rows])
                      for rows in plans]
        self.calls = 0

    def plan_for(self, question, evidence=None):
        plan = self.plans[min(self.calls, len(self.plans) - 1)]
        self.calls += 1
        return plan


def raises(query):
    raise ValueError(f"upstream is down for {query!r}")


def registry(ref):
    tools = ref.ToolRegistry()
    tools.register("search", ref.fake_search)
    tools.register("round_million", ref.rounded_million)
    tools.register("flaky", raises)
    return tools


def errors(evidence):
    return [v for v in evidence.values() if v.startswith("error:")]


def plan_and_execute(ref, plans, cap=2):
    """ReWOO run in a loop: replan while any worker's evidence reads as an error."""
    planner, solver = ReplanningPlanner(ref, plans), ref.ScriptedSolver(TEMPLATE)
    runs = []
    for _ in range(cap + 1):
        runs.append(ref.run_rewoo(QUESTION, planner, registry(ref), solver))
        if not errors(runs[-1].evidence):
            break
    return runs


def summarise(runs):
    last = runs[-1]
    return {"passes": len(runs), "replans": len(runs) - 1,
            "first_errors": errors(runs[0].evidence), "last_errors": errors(last.evidence),
            "answer": last.answer, "planner_chars": sum(r.planner_chars for r in runs)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped = ref.ScriptedPlanner(ref.Plan(steps=[]))
    return {
        "shipped_inputs": [p for p in inspect.signature(
            ref.ScriptedPlanner.plan_for).parameters if p != "self"],
        "replanning_inputs": [p for p in inspect.signature(
            ReplanningPlanner.plan_for).parameters if p != "self"],
        "constant": shipped.plan_for("a") is shipped.plan_for("b"),
        "recovered": summarise(plan_and_execute(ref, [BROKEN, FIXED])),
        "missed": summarise(plan_and_execute(ref, [MISS])),
        "unfixable": summarise(plan_and_execute(ref, [ALWAYS_RAISES])),
        "one_pass": summarise(plan_and_execute(ref, [ALWAYS_RAISES], 0))["planner_chars"],
    }


def verify(result):
    good, miss, bad = result["recovered"], result["missed"], result["unfixable"]
    return [
        practice.Check(
            "ANSWER: one extra planner input, and the replanner fires exactly once",
            all([result["shipped_inputs"] == ["question"],
                 result["replanning_inputs"] == ["question", "evidence"],
                 good["replans"] == 1, len(good["first_errors"]) == 1,
                 good["last_errors"] == [],
                 good["answer"].endswith("rounded population is 11 million.")]),
            f"plan_for goes from {result['shipped_inputs']} to "
            f"{result['replanning_inputs']} -- ReWOO's three arrows become four. Pass 1 "
            f"yields {good['first_errors']}, the replanner fires {good['replans']} time, "
            f"and pass {good['passes']} ends with {len(good['last_errors'])} errors",
        ),
        practice.Check(
            "FINDING: the trigger is a prefix, and a search miss does not carry it",
            all([miss["replans"] == 0, miss["first_errors"] == [],
                 "no result for" in miss["answer"]]),
            f"a plan that searches for something that does not exist finishes with "
            f"{len(miss['first_errors'])} errors and {miss['replans']} replans, and the "
            f"solver composes {miss['answer']!r}. 1 of the 2 failure kinds reaches the "
            "trigger; the other arrives as evidence",
        ),
        practice.Check(
            "FINDING: the replan needs a bound, and the bound cannot tell why it failed",
            all([result["constant"] is True, bad["replans"] == 2,
                 len(bad["last_errors"]) == 1,
                 bad["last_errors"][0].startswith("error: ValueError")]),
            f"ScriptedPlanner.plan_for returns the same object for two different "
            f"questions ({result['constant']}), so an unbounded replanner is an infinite "
            f"loop. Against a tool that always raises, the bounded one burns "
            f"{bad['replans']} replans and still ends with {bad['last_errors']}",
        ),
        practice.Check(
            "FINDING: every replan re-pays the planner prompt",
            all([result["one_pass"] == 156, bad["planner_chars"] == 468,
                 bad["planner_chars"] == 3 * result["one_pass"]]),
            f"planner_chars is {result['one_pass']} for one pass and "
            f"{bad['planner_chars']} across {bad['passes']}. The paper's 5x is per plan, "
            "so a task that replans twice has paid three planner prompts for one answer "
            "and the saving shrinks with every retry",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
