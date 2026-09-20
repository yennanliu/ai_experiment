"""Exercise 5 — a sequence moves the ordering obligation to the planner.

    Port the toy to Plan-and-Act's trajectory shape: plan is a sequence, not a
    DAG. What tradeoffs change?

Reading of the exercise: the port is a deletion. `run_workers` is
`topological()` plus a loop; drop the sort and the loop already executes a
trajectory. So every tradeoff is a consequence of who now owns the ordering,
and each one is measured by running the same plans through both executors --
the lesson's `run_workers` and the sequence version built on the lesson's own
`resolve_references`.

**ANSWER: the port is `run_workers` without the sort.** On the demo chain the
two executors agree on **3** of **3** evidence values and produce the same
answer. On a 6-node DAG with two parallel groups the sequence takes **6**
steps where the DAG resolves in **3** waves.

**FINDING: the trajectory keeps one order and records nothing about the
rest.** That DAG admits **20** valid orders; a sequence stores **1** of them
and has no field that says the other **19** were equally correct. The
information is not slower, it is absent -- which is why Plan-and-Act's
contribution is *labelled trajectory data* rather than a runtime change.

**FINDING: the obligation moves to the planner.** The demo plan declared
backwards runs correctly under the DAG executor -- **3** of **3** evidence
values match -- while the sequence executor sends
`no result for 'population of #E1'` and then rounds the *reference name*,
answering `2 million` because `rounded_million` found a digit in `#E2`. The
DAG tolerates a planner that emits steps in any order; the sequence does not.

**FINDING: a dangling reference stops raising and becomes evidence.** In the
DAG a `#E9` that names nothing raises `RuntimeError` before **0** tool calls
have run. In the sequence `resolve_references` leaves the literal alone and
the tool answers `no result for 'population of #E9'` -- a well-formed answer
to a malformed plan. What the sequence gains in exchange is that a cycle
becomes unwritable and completion order equals declaration order, so the
positional bookkeeping that `run_rewoo` does mis-pairs **0** steps.

Structure: `run_sequence()` is the port; `linearizations()` counts what the
port throws away.
"""

from __future__ import annotations

import itertools

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "02-rewoo-plan-and-execute"
CHAIN = (("E1", "search", {"query": "capital of France"}),
         ("E2", "search", {"query": "population of #E1"}),
         ("E3", "round_million", {"text": "#E2"}))
DANGLING = (("E1", "search", {"query": "capital of France"}),
            ("E2", "search", {"query": "population of #E9"}),
            ("E3", "round_million", {"text": "#E2"}))
WIDE = (("E1", "search", {"query": "capital of France"}),
        ("E2", "search", {"query": "capital of Germany"}),
        ("E3", "search", {"query": "population of Paris"}),
        ("E4", "join", {"left": "#E1", "right": "#E2"}),
        ("E5", "round_million", {"text": "#E3"}),
        ("E6", "join", {"left": "#E4", "right": "#E5"}))


def build(ref, rows):
    return ref.Plan(steps=[ref.PlanStep(i, t, dict(a)) for i, t, a in rows])


def registry(ref):
    tools = ref.ToolRegistry()
    tools.register("search", ref.fake_search)
    tools.register("round_million", ref.rounded_million)
    tools.register("join", lambda left, right: f"{left} + {right}")
    return tools


def run_dag(ref, plan, tools):
    """The shipped executor, with the reference's argument order adapted."""
    return ref.run_workers(plan, tools)


def run_sequence(ref, plan, tools):
    """Plan-and-Act's shape: declaration order is execution order, full stop."""
    evidence = {}
    for step in plan.steps:
        bound = {k: ref.resolve_references(v, evidence) for k, v in step.args.items()}
        evidence[step.id] = tools.dispatch(step.tool, bound)
    return evidence


def waves(ref, plan):
    levels, known, pending = [], set(), list(plan.steps)
    while pending:
        ready = [s for s in pending
                 if all(f"E{r}" in known for r in ref.REFERENCE_RE.findall(str(s.args)))]
        levels.append(ready)
        known.update(step.id for step in ready)
        pending = [s for s in pending if s not in ready]
    return levels


def valid(ref, order):
    seen = set()
    for step in order:
        if any(f"E{r}" not in seen for r in ref.REFERENCE_RE.findall(str(step.args))):
            return False
        seen.add(step.id)
    return True


def linearizations(ref, plan):
    return sum(1 for order in itertools.permutations(plan.steps) if valid(ref, order))


def attempt(runner, ref, rows):
    try:
        return runner(ref, build(ref, rows), registry(ref)), ""
    except Exception as exc:
        return {}, str(exc)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    dag_chain, _ = attempt(run_dag, ref, CHAIN)
    seq_chain, _ = attempt(run_sequence, ref, CHAIN)
    dag_back, _ = attempt(run_dag, ref, CHAIN[::-1])
    seq_back, _ = attempt(run_sequence, ref, CHAIN[::-1])
    seq_dangling, _ = attempt(run_sequence, ref, DANGLING)
    _, dag_dangling = attempt(run_dag, ref, DANGLING)
    wide = build(ref, WIDE)
    return {
        "agree": sum(1 for k, v in seq_chain.items() if dag_chain[k] == v),
        "steps": len(WIDE), "waves": len(waves(ref, wide)),
        "orders": linearizations(ref, wide),
        "back_dag": sum(1 for k, v in dag_back.items() if dag_chain[k] == v),
        "back_seq": seq_back["E2"], "back_rounded": seq_back["E3"],
        "dag_dangling": dag_dangling, "seq_dangling": seq_dangling["E2"],
        "seq_order": list(seq_back) == [row[0] for row in CHAIN[::-1]],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the port is run_workers without the sort",
            all([result["agree"] == 3, result["steps"] == 6, result["waves"] == 3,
                 result["steps"] > result["waves"]]),
            f"on the demo chain the two executors agree on {result['agree']} of 3 evidence "
            f"values; on a 6-node DAG the sequence takes {result['steps']} steps where the "
            f"DAG resolves in {result['waves']} waves. The port deletes a function call "
            "and the trajectory falls out of the loop that was already there",
        ),
        practice.Check(
            "FINDING: the trajectory keeps one order and records nothing about the rest",
            all([result["orders"] == 20, result["waves"] == 3]),
            f"that DAG admits {result['orders']} valid orders; a sequence stores 1 and has "
            f"no field saying the other {result['orders'] - 1} were equally correct. The "
            "independence is absent rather than unused, which is why Plan-and-Act's "
            "contribution is labelled trajectory data and not a runtime change",
        ),
        practice.Check(
            "FINDING: the ordering obligation moves to the planner",
            all([result["back_dag"] == 3,
                 result["back_seq"] == "no result for 'population of #E1'",
                 result["back_rounded"] == "2 million", result["seq_order"] is True]),
            f"declared backwards, the DAG executor still matches {result['back_dag']} of 3 "
            f"evidence values, while the sequence executor sends {result['back_seq']!r} "
            f"and then rounds the reference name itself to {result['back_rounded']!r}. "
            "The DAG tolerates any declaration order; the sequence tolerates one",
        ),
        practice.Check(
            "FINDING: a dangling reference stops raising and becomes evidence",
            all([result["dag_dangling"] == "cyclic plan or unresolved reference",
                 result["seq_dangling"] == "no result for 'population of #E9'"]),
            f"the DAG raises {result['dag_dangling']!r} before any tool runs; the sequence "
            f"answers {result['seq_dangling']!r} -- a well-formed answer to a malformed "
            "plan. In exchange a cycle becomes unwritable and completion order equals "
            "declaration order, so run_rewoo's positional bookkeeping cannot mis-pair",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
