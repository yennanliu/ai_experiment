"""Exercise 1 — the waves are computed and then flattened away.

    Parallelize worker execution for independent plan nodes. What does it buy
    you on a 6-node DAG with 2 parallel groups?

Reading of the exercise: `topological()` is already a Kahn sweep -- its
`while pending` loop resolves one whole independent wave per iteration -- and
then appends every wave into one flat list. So "parallelize" needs no new
dependency analysis; it needs the loop to stop discarding what it computed.
The buy is therefore measured as makespan in waves, against the same DAG run
by the shipped `run_workers`.

**ANSWER: the 6-node DAG resolves into **3** waves of **3**, **2** and **1**.**
Sequential execution is **6** unit steps; wave execution is **3**, a **2.00x**
speedup, and the widest wave says **3** workers is the most concurrency this
plan can use. The evidence is identical either way -- **6** of **6** values
match the shipped run.

**FINDING: the shipped sweep already knows the waves.** Re-running the same
loop with the level boundaries kept yields `[3, 2, 1]` without one extra
dependency lookup. `topological` returns `list[PlanStep]`, which is a shape
that cannot hold the answer, so the information is lost at the return
statement rather than never computed.

**FINDING: the accounting's pairing is already wrong, and a sum hides it.**
`run_rewoo` computes `worker_chars` from `zip(plan.steps, evidence.values())`
-- declaration order against *completion* order. Declaring the same plan
backwards mis-associates **2** of **3** pairs while `worker_chars` stays at
**108**, because addition does not care which value it added to which.
Parallel execution makes that mis-association the normal case, and the next
non-additive statistic taken from the same zip will be wrong loudly.

**FINDING: the demo plan buys nothing.** `E1 -> E2 -> E3` is a chain: **3**
waves for **3** nodes, speedup **1.00x**. The win belongs to the plan's
shape, not to ReWOO.

Structure: `waves()` is the un-flattened sweep; `run_in_waves()` executes
them; both lean on the lesson's own `resolve_references` and `ToolRegistry`.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "02-rewoo-plan-and-execute"
WIDE = (("E1", "search", {"query": "capital of France"}),
        ("E2", "search", {"query": "capital of Germany"}),
        ("E3", "search", {"query": "population of Paris"}),
        ("E4", "join", {"left": "#E1", "right": "#E2"}),
        ("E5", "round_million", {"text": "#E3"}),
        ("E6", "join", {"left": "#E4", "right": "#E5"}))


def build(ref, rows):
    return ref.Plan(steps=[ref.PlanStep(sid, tool, dict(args)) for sid, tool, args in rows])


def registry(ref):
    tools = ref.ToolRegistry()
    tools.register("search", ref.fake_search)
    tools.register("round_million", ref.rounded_million)
    tools.register("join", lambda left, right: f"{left} + {right}")
    return tools


def waves(ref, plan):
    """`topological` without the flatten: one list per independently runnable group."""
    levels, known, pending = [], set(), list(plan.steps)
    while pending:
        ready = [s for s in pending
                 if all(f"E{r}" in known for r in ref.REFERENCE_RE.findall(str(s.args)))]
        if not ready:
            raise RuntimeError("cyclic plan or unresolved reference")
        levels.append(ready)
        known.update(step.id for step in ready)
        pending = [s for s in pending if s not in ready]
    return levels


def run_in_waves(ref, plan, tools):
    evidence = {}
    for level in waves(ref, plan):
        bound = {step.id: (step.tool, {k: ref.resolve_references(v, evidence)
                                       for k, v in step.args.items()})
                 for step in level}
        for step_id, (tool, args) in bound.items():
            evidence[step_id] = tools.dispatch(tool, args)
    return evidence


def accounting(ref, rows):
    """`run_rewoo`'s own bookkeeping: declaration order zipped against completions."""
    plan = build(ref, rows)
    solver = ref.ScriptedSolver("The capital of France is {E1}; rounded is {E3}.")
    run = ref.run_rewoo("population of the capital of France, rounded?",
                        ref.ScriptedPlanner(plan), registry(ref), solver)
    mispaired = sum(1 for step, key in zip(run.plan.steps, run.evidence) if step.id != key)
    return run.worker_chars, mispaired


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    wide = build(ref, WIDE)
    levels = waves(ref, wide)
    parallel = run_in_waves(ref, wide, registry(ref))
    sequential = ref.run_workers(wide, registry(ref))
    demo = (("E1", "search", {"query": "capital of France"}),
            ("E2", "search", {"query": "population of #E1"}),
            ("E3", "round_million", {"text": "#E2"}))
    return {
        "widths": [len(level) for level in levels],
        "nodes": len(WIDE), "makespan": len(levels),
        "speedup": round(len(WIDE) / len(levels), 2),
        "agree": sum(1 for k, v in parallel.items() if sequential[k] == v),
        "flat": ref.topological(wide).__class__.__name__,
        "chain_waves": len(waves(ref, build(ref, demo))),
        "chain_speedup": round(3 / len(waves(ref, build(ref, demo))), 2),
        "declared": accounting(ref, demo), "reordered": accounting(ref, demo[::-1]),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 6 nodes resolve into 3 waves of 3, 2 and 1 -- a 2.00x makespan win",
            all([result["widths"] == [3, 2, 1], result["nodes"] == 6,
                 result["makespan"] == 3, result["speedup"] == 2.0,
                 result["agree"] == 6]),
            f"the DAG resolves into waves of {result['widths']}: {result['nodes']} unit "
            f"steps become {result['makespan']}, a {result['speedup']}x speedup, and the "
            f"widest wave caps useful concurrency at {max(result['widths'])} workers. "
            f"{result['agree']} of {result['nodes']} evidence values match the shipped run",
        ),
        practice.Check(
            "FINDING: the shipped sweep already computes the waves, then flattens them",
            all([result["flat"] == "list", result["widths"] == [3, 2, 1]]),
            f"keeping the level boundaries of the same Kahn sweep yields "
            f"{result['widths']} with no extra dependency lookup. topological returns a "
            f"{result['flat']}, a shape that cannot hold the answer, so the grouping is "
            "lost at the return statement rather than never computed",
        ),
        practice.Check(
            "FINDING: the accounting's pairing is already wrong, and a sum hides it",
            all([result["declared"] == (108, 0), result["reordered"] == (108, 2),
                 result["declared"][0] == result["reordered"][0]]),
            f"run_rewoo zips plan.steps against evidence.values() -- declaration order "
            f"against completion order. Declaring the same plan backwards mis-associates "
            f"{result['reordered'][1]} of 3 pairs while worker_chars stays at "
            f"{result['reordered'][0]}, because a sum does not care which value it added "
            "to which. Parallel execution makes that mis-association the normal case",
        ),
        practice.Check(
            "FINDING: the demo plan buys nothing at all",
            all([result["chain_waves"] == 3, result["chain_speedup"] == 1.0]),
            f"E1 -> E2 -> E3 is a chain: {result['chain_waves']} waves for 3 nodes, "
            f"{result['chain_speedup']}x. The speedup is a property of the plan's shape, "
            "not of ReWOO, so the pattern promises nothing until a planner emits width",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
