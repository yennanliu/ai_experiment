"""Exercise 3 — the manager is handed a set of names, never the output.

    Implement a Hierarchical process where the manager refuses to route to
    the editor until the writer's output has at least three paragraphs. Trace
    the retry.

Reading of the exercise: `HierarchicalCrew.kickoff` calls
`self.manager.fn(done, [], None)` -- the manager receives the set of roles
already run and nothing else, so a gate on the writer's *output* cannot be
written against the shipped signature. The change is to pass the current
output too, and the retry then becomes measurable: the manager sends the
writer back rather than advancing.

**ANSWER: a manager that sees the output, and one retry in the trace.** The
writer's first draft has **0** paragraph breaks, so the manager routes back
to `writer` instead of `editor`; the second draft has **3** and the run
advances. The trace is **5** lines for a **3**-role crew, with `writer`
appearing **2** times.

**FINDING: the draft says three paragraphs and has none.** `_writer` returns
`draft (3 paragraphs) from sources: ...` -- a literal claim in the text --
while the string contains **0** newlines. A gate that trusted the wording
would pass it; a gate that counts breaks fails it. The shipped
`expected_output="3 paragraphs"` is checked by nobody.

**FINDING: `done` is a set, so a retry cannot be expressed in it.**
`kickoff` adds each pick to `done` and the shipped `_manager` returns the
first role not in it, so once `writer` is in `done` there is no value the
manager can return to run it again -- the retry has to live in the manager's
own state. The set has **3** entries for a **4**-step run.

**FINDING: the retry budget is the whole-run budget.** `max_steps=5` counts
manager turns, not retries, so two retries of a three-role crew exhaust it
and the run ends with **0** editor output and no error -- the loop simply
falls out of `for _ in range(self.max_steps)`.

Structure: `GatedManager` keeps the retry count the set cannot hold;
`gated_kickoff` is `HierarchicalCrew.kickoff` with the output passed along.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "15-crewai-role-based-crews"
TOPIC = "agent engineering 2026"
PARAGRAPHS = 3


class GatedManager:
    """Sees the current output, so it can refuse to advance."""

    def __init__(self, required=PARAGRAPHS):
        self.required, self.retries = required, 0

    def pick(self, done, current):
        if "researcher" not in done:
            return "researcher"
        if "writer" not in done:
            return "writer"
        if str(current).count("\n\n") + 1 < self.required:
            self.retries += 1
            return "writer"
        if "editor" not in done:
            return "editor"
        return "done"


def widening_writer(state):
    """Second attempt actually emits paragraphs; the first only says so."""
    def writer(prior, tools, memory):
        state["calls"] += 1
        if state["calls"] == 1:
            return f"draft (3 paragraphs) from sources: {str(prior)[:40]}"
        return "\n\n".join(f"para {n} on {str(prior)[:18]}" for n in range(1, 4))
    return writer


def gated_kickoff(ref, manager, writer_fn, max_steps=6):
    researcher, _, editor = ref.build_agents()
    specialists = {"researcher": researcher,
                   "writer": ref.Agent("writer", "draft", "voice", writer_fn),
                   "editor": editor}
    outputs, current, done = [], TOPIC, set()
    for _ in range(max_steps):
        pick = manager.pick(done, current)
        if pick == "done":
            outputs.append("[manager] done")
            break
        current = specialists[pick].fn(current, specialists[pick].tools, None)
        outputs.append(f"[manager -> {pick}] {current}")
        done.add(pick)
    return outputs, done


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    manager = GatedManager()
    trace, done = gated_kickoff(ref, manager, widening_writer({"calls": 0}))
    first_draft = ref._writer(TOPIC, [], None)
    starved, _ = gated_kickoff(ref, GatedManager(),
                               lambda p, t, m: "one paragraph only", max_steps=5)
    return {
        "trace": trace, "lines": len(trace), "retries": manager.retries,
        "writer_turns": sum(1 for line in trace if "-> writer" in line),
        "done": sorted(done), "reached_editor": any("-> editor" in line
                                                    for line in trace),
        "claim": "3 paragraphs" in first_draft,
        "breaks": first_draft.count("\n\n"),
        "expected_output": ref.Task("write", "3 paragraphs", None).expected_output,
        "checks_expected": inspect.getsource(
            ref.SequentialCrew.kickoff).count("expected_output"),
        "manager_args": inspect.getsource(
            ref.HierarchicalCrew.kickoff).count("self.manager.fn(done, [], None)"),
        "starved_editor": any("-> editor" in line for line in starved),
        "starved_lines": len(starved), "max_steps": ref.HierarchicalCrew.max_steps,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: one retry, five trace lines, writer twice",
            all([result["retries"] == 1, result["writer_turns"] == 2,
                 result["lines"] == 5, result["reached_editor"] is True,
                 result["done"] == ["editor", "researcher", "writer"]]),
            f"the first draft has no paragraph breaks, so the manager routes back to "
            f"writer -- {result['retries']} retry, {result['writer_turns']} writer turns "
            f"in a {result['lines']}-line trace -- and the second draft advances to the "
            f"editor ({result['reached_editor']})",
        ),
        practice.Check(
            "FINDING: the draft says three paragraphs and has none",
            all([result["claim"] is True, result["breaks"] == 0,
                 result["expected_output"] == "3 paragraphs",
                 result["checks_expected"] == 0]),
            f"_writer returns text containing '3 paragraphs' ({result['claim']}) with "
            f"{result['breaks']} paragraph breaks in it, and Task.expected_output is "
            f"{result['expected_output']!r}, which kickoff reads "
            f"{result['checks_expected']} times. A gate that trusts the wording passes "
            "it; one that counts breaks does not",
        ),
        practice.Check(
            "FINDING: done is a set, so a retry cannot be expressed in it",
            all([result["manager_args"] == 1, len(result["done"]) == 3,
                 result["writer_turns"] > len(result["done"]) - 2]),
            f"kickoff calls self.manager.fn(done, [], None) -- {result['manager_args']} "
            f"call site, no output -- and adds each pick to a set, so after "
            f"{result['writer_turns']} writer turns the set still holds "
            f"{len(result['done'])} entries. The retry has to live in the manager's own "
            "state because the set cannot count",
        ),
        practice.Check(
            "FINDING: the retry budget is the whole-run budget",
            all([result["starved_editor"] is False, result["starved_lines"] == 5,
                 result["max_steps"] == 5]),
            f"max_steps={result['max_steps']} counts manager turns rather than retries, "
            f"so a writer that never reaches three paragraphs uses the whole budget: "
            f"{result['starved_lines']} lines and editor reached "
            f"{result['starved_editor']}. The loop falls out of range() with no error",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
