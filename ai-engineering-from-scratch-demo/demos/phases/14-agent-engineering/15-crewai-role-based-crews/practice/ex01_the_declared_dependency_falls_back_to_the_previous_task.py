"""Exercise 1 — the declared dependency falls back to the previous task.

    Convert the Sequential crew to a Flow. Count the touchpoints where
    variability drops. Note where readability dropped.

Reading of the exercise: "variability" here means decisions resolved at run
time rather than written down, and `SequentialCrew.kickoff` has exactly three
kinds -- which agent runs, what it is fed, and whether memory is written. A
`Flow` step makes all three literal. Counting them is therefore a matter of
counting the crew's run-time branches, and the readability cost is counted in
the same units on the other side.

**ANSWER: the same three outputs, with 3 touchpoints made literal and 4
named steps to pay for them.** The Flow reproduces the crew's output
verbatim -- **3** of **3** strings match -- and the per-task input decision,
which the crew resolves from `task.context` at run time, becomes a parameter
in each listener. The cost is shape: the crew declares the pipeline in **3**
`Task` rows and the Flow needs **4** decorated functions plus a terminator.

**FINDING: a declared dependency silently becomes an implicit one.**
`kickoff` computes `agent_input = joined or prior`, so a task whose declared
`context` produced nothing falls back to the *previous* task's output. A task
declaring a dependency on a task the crew was never given still runs, on the
wrong input, with **0** warnings.

**FINDING: the Flow has no step cap.** `HierarchicalCrew` carries
`max_steps=5` and `Flow.kickoff` loops `while topic in self.listeners` with
**0** counters, so a listener that re-emits its own topic never returns. The
shape that "code owns" is the one with no bound on it.

**FINDING: the trace shapes are not comparable.** A crew returns
`list[str]` -- role and text, **2** facts per entry -- and a flow returns
`(step_name, topic, output)`, **3**. The flow's extra field is the topic,
which is the thing that made the routing literal in the first place.

Structure: `as_flow()` builds the Flow from the lesson's own agent functions;
`crew_outputs()` runs the shipped `SequentialCrew` beside it.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "15-crewai-role-based-crews"
TOPIC = "agent engineering 2026"


def crew_outputs(ref, tasks=None):
    researcher, writer, editor = ref.build_agents()
    tasks = tasks or [ref.Task("research", "3 sources", researcher),
                      ref.Task("write", "3 paragraphs", writer),
                      ref.Task("edit", "800 words", editor)]
    crew = ref.SequentialCrew(agents=[researcher, writer, editor], tasks=tasks,
                              memory=ref.Memory())
    return crew.kickoff({"topic": TOPIC})


def as_flow(ref):
    """Every run-time decision of the crew written down as a step."""
    researcher, writer, editor = ref.build_agents()
    flow = ref.Flow()

    @flow.start
    def research(topic):
        return "researched", researcher.fn(topic, researcher.tools, None)

    @flow.listen("researched")
    def draft(prior):
        return "drafted", writer.fn(prior, writer.tools, None)

    @flow.listen("drafted")
    def edit(prior):
        return "edited", editor.fn(prior, editor.tools, None)

    @flow.listen("edited")
    def stop(prior):
        return None

    return flow, (research, draft, edit, stop)


def orphan_context(ref):
    """A task declaring a dependency the crew was never given."""
    researcher, writer, editor = ref.build_agents()
    ghost = ref.Task("unrun", "nothing", researcher)
    tasks = [ref.Task("research", "3 sources", researcher),
             ref.Task("write", "3 paragraphs", writer, context=[ghost]),
             ref.Task("edit", "800 words", editor)]
    return crew_outputs(ref, tasks)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    crew = crew_outputs(ref)
    flow, steps = as_flow(ref)
    trace = flow.kickoff(TOPIC)
    flow_outputs = [f"[{role}] {output}" for role, (_, _, output)
                    in zip(("researcher", "writer", "editor"), trace)]
    orphaned = orphan_context(ref)
    source = inspect.getsource(ref.Flow.kickoff)
    return {
        "crew": crew, "flow": flow_outputs,
        "matched": sum(a == b for a, b in zip(crew, flow_outputs)),
        "steps": len(steps), "tasks": 3,
        "orphan_matches_prior": orphaned[1] == crew[1],
        "orphan_warnings": sum("warn" in line.lower() for line in orphaned),
        "flow_caps": source.count("max"), "crew_caps": ref.HierarchicalCrew.max_steps,
        "crew_entry_fields": 2, "flow_entry_fields": len(trace[0]),
        "topics": [topic for _, topic, _ in trace],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: identical outputs, three decisions made literal, four steps",
            all([result["matched"] == 3, result["crew"] == result["flow"],
                 result["steps"] == 4, result["tasks"] == 3,
                 result["topics"] == ["researched", "drafted", "edited"]]),
            f"the Flow reproduces the crew's output verbatim -- {result['matched']} of 3 "
            f"strings match -- with the per-task input decision written into each "
            f"listener and the routing written as topics {result['topics']}. The cost is "
            f"{result['steps']} decorated functions against {result['tasks']} Task rows",
        ),
        practice.Check(
            "FINDING: a declared dependency silently becomes an implicit one",
            all([result["orphan_matches_prior"] is True,
                 result["orphan_warnings"] == 0]),
            f"kickoff computes agent_input = joined or prior, so a task whose declared "
            f"context produced nothing falls back to the previous task's output: the "
            f"orphaned run is identical to the plain one "
            f"({result['orphan_matches_prior']}) with {result['orphan_warnings']} "
            "warnings",
        ),
        practice.Check(
            "FINDING: the Flow has no step cap",
            all([result["flow_caps"] == 0, result["crew_caps"] == 5]),
            f"HierarchicalCrew carries max_steps={result['crew_caps']} and Flow.kickoff "
            f"mentions a cap {result['flow_caps']} times -- it loops while the topic has "
            "a listener. The shape that code owns is the one with no bound on it",
        ),
        practice.Check(
            "FINDING: the trace shapes are not comparable",
            all([result["crew_entry_fields"] == 2, result["flow_entry_fields"] == 3,
                 len(result["topics"]) == 3]),
            f"a crew returns list[str] -- role and text, "
            f"{result['crew_entry_fields']} facts per entry -- and a flow returns "
            f"(step_name, topic, output), {result['flow_entry_fields']}. The extra field "
            "is the topic, which is what made the routing literal",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
