"""Exercise 2 — a third level needs a type the lesson does not have.

    Add a third level (top → sub → sub-sub → worker). Measure how often the
    perturbed path corrects itself vs fully diverges as depth grows.

Reading of the exercise: try to add the level first, because it does not go
in. `SubManager` aggregates `l.answer` over its children and a sub-sub manager
returns a `SubSummary`, which has no `answer`. Once an adapter is written, the
correction rate turns out to be a constant.

**ANSWER: it never corrects, at any depth, and the rate is fixed by the
shape.** With an adapter in place, measuring the user's surviving content
words in the top synthesis gives **0 of 5** at depth 2, depth 3 and depth 4
alike, over both the happy and the perturbed labels -- **6** runs, **0**
corrections. Every level composes its output as a join over its children's
strings, so it can drop a word and has no way to reintroduce one. Correction
would need a level that compares against the original question, and the only
level holding it is the top.

**FINDING: the levels are not composable.** `LeafOutput` has fields
worker/question/answer and `SubSummary` has sub_manager/leaves/summary --
**0** names in common. `SubManager.run` builds its summary from
`l.answer for l in leaves`, so handing it another `SubManager` raises
`AttributeError`. The three-level hierarchy the lesson ships is the *only*
depth its types allow; "add a third level" is a type change, not a
configuration.

**FINDING: the task reaches 0 leaves at any depth.** `SubManager.run` reads
`self.split.get(w.name, task)` and every child has a `split` entry, so the
fallback never fires. Leaves shown the user's question verbatim number **0**
at depths 2, 3 and 4 alike -- deeper levels only add more constant briefs.

**FINDING: the trace records one level however deep the tree goes.** Worker
invocations double with depth -- **8, 16, 32** -- while
`TopSynthesis.branches[].leaves` holds **4** at every depth, because anything
below the top sub-manager is flattened into a summary string before its parent
sees it. The structure meant to record the hierarchy records exactly one level
of it, so the deeper tree is invisible to the thing rendering the trace.

Structure: `Adapter` gives a `SubManager` the `LeafOutput` shape its parent
expects; `Counting` tallies real invocations; `chain()` stacks them.
"""

from __future__ import annotations

import dataclasses
import inspect
import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "06-hierarchical-architecture"
TASK = "Ship the premium tier feature to production."
STOP = {"the", "to", "of", "a", "and", "for", "in", "on", "is"}
DEPTHS = (2, 3, 4)
BRANCHING = 2


def words(text):
    """Content words, lowercased, with the obvious stopwords dropped."""
    return {word for word in re.findall(r"[a-z]+", text.lower())} - STOP


class Adapter:
    """A SubManager wearing a Worker's interface, because the types do not compose."""

    def __init__(self, ref, name, inner):
        self.ref, self.name, self.inner = ref, name, inner

    def run(self, question):
        summary = self.inner.run(question)
        return self.ref.LeafOutput(worker=self.name, question=question,
                                   answer=summary.summary)


class Counting:
    """A Worker that records every invocation, so deep runs can be counted."""

    def __init__(self, ref, name, tally):
        self.inner = ref.Worker(name, {"topic": f"canned finding {name}"})
        self.name, self.tally = name, tally

    def run(self, question):
        self.tally.append(self.name)
        return self.inner.run(question)


def chain(ref, depth, tally, branching=BRANCHING):
    """A hierarchy `depth` levels below the top, in the shipped shape."""
    def wrap(children, name):
        return ref.SubManager(name, children,
                              {child.name: f"fixed brief for {child.name}"
                               for child in children})

    nodes = [Counting(ref, f"w{n}", tally) for n in range(branching)]
    for level in range(depth - 1):
        inner = wrap(nodes, f"sub{level}")
        nodes = [Adapter(ref, f"a{level}{n}", inner) for n in range(branching)]
    return ref.TopManager("top", {f"b{n}": wrap(nodes, "sub-top")
                                  for n in range(branching)})


def measure(ref, depth, labels):
    """Overlap, leaves recorded in the trace, workers invoked, leaves shown the task."""
    tally = []
    run = chain(ref, depth, tally).run(TASK, branch_labels=labels)
    questions = [leaf.question for branch in run.branches for leaf in branch.leaves]
    return (len(words(TASK) & words(run.synthesis)), len(questions), len(tally),
            sum(TASK in question for question in questions))


def composable(ref):
    """Whether a SubManager can stand where a Worker does, unadapted."""
    leaf = {field.name for field in dataclasses.fields(ref.LeafOutput)}
    summary = {field.name for field in dataclasses.fields(ref.SubSummary)}
    inner = ref.SubManager("inner", [ref.Worker("w", {"t": "a"})], {"w": "brief"})
    outer = ref.SubManager("outer", [inner], {"inner": "brief"})
    try:
        outer.run(TASK)
        raised = None
    except AttributeError as exc:
        raised = type(exc).__name__
    return leaf & summary, raised


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    labels = [f"b{n}" for n in range(BRANCHING)]
    results = {depth: measure(ref, depth, labels) for depth in DEPTHS}
    perturbed = {depth: measure(ref, depth, ["b0", "b0"])[0] for depth in DEPTHS}
    shared, raised = composable(ref)
    return {
        "asked": len(words(TASK)), "depths": list(DEPTHS),
        "overlap": [row[0] for row in results.values()],
        "leaves": [row[1] for row in results.values()],
        "invoked": [row[2] for row in results.values()],
        "task_at_leaf": [row[3] for row in results.values()],
        "perturbed": list(perturbed.values()),
        "runs": len(DEPTHS) * 2,
        "shared_fields": sorted(shared), "raised": raised,
        "aggregates": "l.answer for l in leaves" in inspect.getsource(ref.SubManager.run),
    }


def verify(result):
    corrections = sum(n > 0 for n in result["overlap"] + result["perturbed"])
    return [
        practice.Check(
            "ANSWER: it never corrects, at any depth",
            all([corrections == 0, result["overlap"] == [0, 0, 0],
                 result["perturbed"] == [0, 0, 0], result["runs"] == 6]),
            f"across depths {result['depths']} on both label sets -- {result['runs']} "
            f"runs -- surviving overlap is {result['overlap']} of {result['asked']} and "
            "corrections are 0; a join over children's strings cannot reintroduce a word",
        ),
        practice.Check(
            "FINDING: the levels are not composable",
            all([result["shared_fields"] == [], result["raised"] == "AttributeError",
                 result["aggregates"]]),
            f"LeafOutput and SubSummary share {len(result['shared_fields'])} field names "
            f"and SubManager.run builds its summary from l.answer, so nesting one inside "
            f"another raises {result['raised']} -- the shipped depth is the only one its "
            "types allow",
        ),
        practice.Check(
            "FINDING: the task reaches 0 leaves at any depth",
            all([result["task_at_leaf"] == [0, 0, 0], result["overlap"] == [0, 0, 0]]),
            f"leaves shown the user's question verbatim number "
            f"{result['task_at_leaf']} across depths {result['depths']}, because "
            "SubManager reads self.split.get(w.name, task) and every child has an entry "
            "-- deeper levels only add more constant briefs",
        ),
        practice.Check(
            "FINDING: the trace records one level however deep the tree goes",
            all([result["invoked"] == [8, 16, 32], result["leaves"] == [4, 4, 4],
                 result["overlap"] == [0, 0, 0]]),
            f"worker invocations go {result['invoked']} across depths "
            f"{result['depths']} while branches[].leaves stays {result['leaves']} -- "
            "everything below the top sub-manager is flattened into a summary string "
            "before its parent sees it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
