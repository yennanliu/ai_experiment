"""Exercise 5 — two of the three subtasks have nothing to search.

    Combine: use HTN to decompose a compound task into subtasks, then use
    evolutionary search on each subtask's primitive operator. Where does it
    shine, where does it over-engineer?

Reading of the exercise: the two halves join at the operator, so the question
is what an operator offers a search. `Operator` carries preconditions and
effects -- a *feasibility* predicate -- and nothing that scores one
implementation above another, so a fitness function has to be supplied per
subtask from outside. Where it shines and where it over-engineers is then
decided by whether such a function exists and whether its space is big enough
to be worth searching.

**ANSWER: the HTN plan is 3 operators, and evolving all three costs 120
evaluator calls to improve 1.** `tune_pipeline` decomposes into
`open_editor`, `choose_batch`, `open_pr`. Evolving the batch size takes its
cost from **30** to **0** in **40** calls; the other two are flat, so **80**
calls buy **0** improvement.

**FINDING: over-engineering is measurable as calls per unit of improvement.**
**2** of the **3** subtasks return the same cost for every candidate in their
space, so the search is doing exactly as much work as on the one that
matters. A combined system with no per-subtask gate spends its budget in
proportion to the plan's length rather than to where the gains are.

**FINDING: where it shines is a big space with a cheap exact evaluator.** The
batch parameter has **200** candidates and the search reaches the optimum
after **40** evaluations -- a **5.0x** saving over enumerating, on an
evaluator that is a pure function of one integer. That is the AlphaEvolve
condition, stated as a ratio.

**FINDING: the HTN layer supplies no fitness.** `Operator` has **4** fields
and **0** of them is a cost, a score or a quality signal, so evolution over
HTN operators has nothing to optimise until a human writes an evaluator per
subtask. The symbolic layer decides *whether* a step is legal and never
*how good* it is.

Structure: `decompose()` runs the lesson's own planner; `search()` is the
lesson's evolutionary loop over one integer parameter.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "11-planning-htn-and-evolutionary"
SPACE, BUDGET, BEST_BATCH = 200, 40, 137


def cost_for(subtask, value):
    """The fitness a human has to supply, because the operator carries none."""
    if subtask == "choose_batch":
        distance = abs(value - BEST_BATCH)
        return distance + distance % 7
    if subtask == "open_editor":
        return 0
    return 5


def world(ref):
    rows = (("open_editor", ("logged_in",), ("editor_open",)),
            ("choose_batch", ("editor_open",), ("batch_chosen",)),
            ("open_pr", ("batch_chosen",), ("pr_open",)))
    operators = {name: ref.Operator(name, pre, add) for name, pre, add in rows}
    methods = {"tune_pipeline": [ref.Method(
        "m1", "tune_pipeline", ("logged_in",),
        ("open_editor", "choose_batch", "open_pr"))]}
    return operators, methods


def decompose(ref, operators, methods):
    planner = ref.HTNPlanner(operators, methods, ref.ScriptedLLM({}))
    return planner.plan("tune_pipeline", {"logged_in"})


def search(subtask, seed=0, budget=BUDGET, size=4):
    """The lesson's evolutionary loop, over one integer parameter."""
    rng = random.Random(seed)
    calls = [0]

    def score(value):
        calls[0] += 1
        return cost_for(subtask, value)

    values = sorted({rng.randrange(1, SPACE + 1) for _ in range(size)})
    ranked = sorted((score(value), value) for value in values)
    start = ranked[0][0]
    while calls[0] < budget:
        parent = ranked[0][1]
        child = min(SPACE, max(1, parent + rng.choice((-32, -8, -1, 1, 8, 32))))
        ranked = sorted(ranked + [(score(child), child)])[:size]
    return {"start": start, "best": ranked[0][0], "value": ranked[0][1],
            "calls": calls[0], "improved": start - ranked[0][0]}


def budget_report(runs):
    flat = [name for name, row in runs.items() if row["improved"] == 0]
    return {
        "total_calls": sum(row["calls"] for row in runs.values()),
        "improved": [name for name, row in runs.items() if row["improved"] > 0],
        "flat": flat, "wasted": sum(runs[name]["calls"] for name in flat),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    operators, methods = world(ref)
    plan = decompose(ref, operators, methods)
    runs = {subtask: search(subtask) for subtask in plan}
    fields = list(ref.Operator.__dataclass_fields__)
    return {
        "plan": plan, "runs": runs, "space": SPACE,
        "saving": round(SPACE / runs["choose_batch"]["calls"], 1),
        "operator_fields": fields,
        "quality_fields": [f for f in fields if "cost" in f or "score" in f],
        **budget_report(runs),
    }


def verify(result):
    batch = result["runs"]["choose_batch"]
    return [
        practice.Check(
            "ANSWER: three subtasks, 120 evaluator calls, one improvement",
            all([result["plan"] == ["open_editor", "choose_batch", "open_pr"],
                 result["total_calls"] == 120, batch["start"] == 30,
                 batch["best"] == 0, batch["calls"] == 40,
                 result["improved"] == ["choose_batch"]]),
            f"tune_pipeline decomposes into {result['plan']}. Evolving the batch size "
            f"takes its cost from {batch['start']} to {batch['best']} in "
            f"{batch['calls']} calls; across all three subtasks the run spends "
            f"{result['total_calls']} calls and improves {len(result['improved'])}",
        ),
        practice.Check(
            "FINDING: over-engineering is calls per unit of improvement",
            all([len(result["flat"]) == 2, result["wasted"] == 80,
                 sorted(result["flat"]) == ["open_editor", "open_pr"]]),
            f"{len(result['flat'])} of 3 subtasks return the same cost for every "
            f"candidate -- {sorted(result['flat'])} -- so {result['wasted']} of "
            f"{result['total_calls']} calls buy nothing. With no per-subtask gate the "
            "budget is spent in proportion to the plan's length, not to the gains",
        ),
        practice.Check(
            "FINDING: where it shines is a big space with a cheap exact evaluator",
            all([result["space"] == 200, batch["calls"] == 40,
                 result["saving"] == 5.0, batch["value"] == BEST_BATCH]),
            f"the batch parameter has {result['space']} candidates and the search finds "
            f"{batch['value']} after {batch['calls']} evaluations -- a "
            f"{result['saving']}x saving over enumerating, on an evaluator that is a "
            "pure function of one integer. That ratio is the condition to check first",
        ),
        practice.Check(
            "FINDING: the HTN layer supplies no fitness",
            all([len(result["operator_fields"]) == 4, result["quality_fields"] == [],
                 result["operator_fields"][1] == "preconditions"]),
            f"Operator carries {result['operator_fields']} -- "
            f"{len(result['quality_fields'])} of them a cost or a score -- so evolution "
            "over HTN operators has nothing to optimise until a human writes an "
            "evaluator per subtask. The symbolic layer decides legality, never quality",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
