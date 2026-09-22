"""Exercise 3 — the split fails on the defect that costs nothing.

    Replace `Planner` with a small model (7B class) and keep `Solver` on a
    frontier model. Compare end-to-end quality — where does the split fail?

Reading of the exercise: a live 7B planner is not needed to answer "where
does the split fail", because the failure is structural. The solver's only
inputs are the question, the plan and the evidence, so the question is which
planner defects survive into evidence that looks fine. Five defects
characteristic of a small planner are run through the lesson's own
`run_rewoo` and classified by outcome and by how much tool work was paid for
before anyone noticed.

**ANSWER: five defects, five different outcomes.** A dangling `#E9`
reference raises `RuntimeError` after **0** tool calls; a step the solver
template needs but the plan omits raises `KeyError` after **2**; an
unregistered tool name lands in evidence as `error: unknown tool` after **2**;
a plausible-but-wrong query answers normally after **3**; and steps declared
out of dependency order produce byte-identical evidence to the correct plan.

**FINDING: the split fails on the one that costs nothing to produce.** The
wrong query yields `no result for 'capital of Atlantis'` -- **0** evidence
strings carry an error marker -- and the solver composes it into the final
answer. A frontier solver has no channel to the world, only to evidence, so
this is the one class it cannot repair. The other four are either loud or
free.

**FINDING: the loud ones are loud about the wrong thing.** `topological`
raises `cyclic plan or unresolved reference` for a plan with no cycle: one
message, two faults. And the missing step is visible in the plan before any
worker runs, yet it surfaces as a `KeyError` from the solver after **2** of
**3** tool calls are already paid for.

**FINDING: declaration order is free.** A shuffled plan gives evidence and an
answer identical to the correct one, so that flavour of small-model
sloppiness costs **0**. Plan quality has to be scored on the DAG, not on the
token sequence.

Structure: `attempt()` runs one defective plan through the lesson's pipeline
with a call-counting registry and reports what came back.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "02-rewoo-plan-and-execute"
QUESTION = "What is the population of the capital of France, rounded?"
TEMPLATE = "The capital of France is {E1}; rounded population is {E3}."
CORRECT = (("E1", "search", {"query": "capital of France"}),
           ("E2", "search", {"query": "population of #E1"}),
           ("E3", "round_million", {"text": "#E2"}))
DEFECTS = {
    "unknown tool": (("E1", "search", {"query": "capital of France"}),
                     ("E2", "web_search", {"query": "population of #E1"}),
                     ("E3", "round_million", {"text": "#E2"})),
    "dangling reference": (("E1", "search", {"query": "capital of France"}),
                           ("E2", "search", {"query": "population of #E9"}),
                           ("E3", "round_million", {"text": "#E2"})),
    "missing step": (("E1", "search", {"query": "capital of France"}),
                     ("E2", "search", {"query": "population of #E1"})),
    "wrong query": (("E1", "search", {"query": "capital of Atlantis"}),
                    ("E2", "search", {"query": "population of #E1"}),
                    ("E3", "round_million", {"text": "#E2"})),
    "shuffled order": CORRECT[::-1],
}


class Counting:
    """Wraps the lesson's tools so the work paid for before a fault is visible."""

    def __init__(self):
        self.calls = 0

    def wrap(self, fn):
        def inner(**kwargs):
            self.calls += 1
            return fn(**kwargs)
        return inner


def attempt(ref, rows):
    counter = Counting()
    tools = ref.ToolRegistry()
    tools.register("search", counter.wrap(ref.fake_search))
    tools.register("round_million", counter.wrap(ref.rounded_million))
    plan = ref.Plan(steps=[ref.PlanStep(i, t, dict(a)) for i, t, a in rows])
    try:
        run = ref.run_rewoo(QUESTION, ref.ScriptedPlanner(plan), tools,
                            ref.ScriptedSolver(TEMPLATE))
    except Exception as exc:
        return {"outcome": type(exc).__name__, "detail": str(exc),
                "answer": None, "marked": 0, "paid": counter.calls}
    marked = sum(1 for v in run.evidence.values() if v.startswith("error:"))
    return {"outcome": "answered", "detail": "", "answer": run.answer,
            "marked": marked, "paid": counter.calls, "evidence": dict(run.evidence)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    good = attempt(ref, CORRECT)
    seen = {label: attempt(ref, rows) for label, rows in DEFECTS.items()}
    return {
        "good": good, "seen": seen,
        "outcomes": sorted({row["outcome"] for row in seen.values()}),
        "identical": seen["shuffled order"]["answer"] == good["answer"],
        "silent": [label for label, row in seen.items()
                   if row["outcome"] == "answered" and row["marked"] == 0
                   and row["answer"] != good["answer"]],
    }


def verify(result):
    seen, good = result["seen"], result["good"]
    return [
        practice.Check(
            "ANSWER: five weak-planner defects, five outcomes",
            all([seen["dangling reference"]["outcome"] == "RuntimeError",
                 seen["dangling reference"]["paid"] == 0,
                 seen["missing step"]["outcome"] == "KeyError",
                 seen["missing step"]["paid"] == 2,
                 seen["unknown tool"]["marked"] == 1,
                 seen["wrong query"]["paid"] == 3, good["paid"] == 3]),
            f"a dangling #E9 raises RuntimeError after "
            f"{seen['dangling reference']['paid']} tool calls; an omitted step raises "
            f"KeyError after {seen['missing step']['paid']} of 3; an unregistered tool "
            f"leaves {seen['unknown tool']['marked']} marked evidence string; a wrong "
            f"query answers normally after {seen['wrong query']['paid']}",
        ),
        practice.Check(
            "FINDING: the split fails on the defect that costs nothing to produce",
            all([result["silent"] == ["wrong query"],
                 seen["wrong query"]["marked"] == 0,
                 "no result for" in seen["wrong query"]["answer"]]),
            f"the wrong query yields {seen['wrong query']['marked']} marked evidence "
            f"strings and the answer {seen['wrong query']['answer']!r}. The solver's "
            "inputs are the question, the plan and the evidence -- it has no channel to "
            f"the world, so {result['silent']} is the class it cannot repair",
        ),
        practice.Check(
            "FINDING: the loud failures are loud about the wrong thing",
            all([seen["dangling reference"]["detail"] ==
                 "cyclic plan or unresolved reference",
                 seen["missing step"]["detail"] == "'E3'",
                 seen["missing step"]["paid"] > seen["dangling reference"]["paid"]]),
            f"topological raises {seen['dangling reference']['detail']!r} for a plan with "
            f"no cycle -- one message, two faults -- and the omitted step surfaces as "
            f"KeyError {seen['missing step']['detail']} from the solver after "
            f"{seen['missing step']['paid']} tool calls, though it was visible in the "
            "plan before any worker ran",
        ),
        practice.Check(
            "FINDING: declaration order is free, so plan quality is a DAG property",
            all([result["identical"] is True,
                 seen["shuffled order"]["paid"] == good["paid"],
                 seen["shuffled order"]["marked"] == 0]),
            f"the shuffled plan pays {seen['shuffled order']['paid']} tool calls and "
            f"returns the same answer as the correct one ({result['identical']}). "
            "Scoring a distilled planner on its token sequence would charge it for a "
            "defect the executor cannot even observe",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
