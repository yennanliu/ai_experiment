"""Exercise 2 — the resume keeps the plan and the node throws it away.

    **Medium.** Add a `planner` node that runs before `agent` and writes a
    structured `plan: list[str]` into state. Have `agent` mark plan steps as
    done. Fail the test if `plan` is lost across a checkpoint resume (wrong
    reducer).

Reading of the exercise: the graph is `planner -> agent -> quiet`, interrupted
before `agent`, so there is a checkpoint between writing the plan and editing
it. The same graph is then compiled three times with three reducers for `plan`
-- none, `operator.add`, and a merge keyed by step index -- and the plan is read
at the interrupt and again after the resume. That is the only way to tell which
of the two events the exercise names is actually destroying it.

**ANSWER: with a merge keyed by index the plan survives.** The planner writes
three steps, `agent` returns only `["research the topic (done)"]`, and the state
holds `['research the topic (done)', 'draft the answer', 'review the draft']`
-- three steps, one of them marked.

**FINDING: with no reducer at all the plan goes from 3 steps to 1.** The
default channel is last-write-wins, so a node returning just the step it
changed replaces the whole list and the other two steps are gone. This is the
failure the exercise is pointing at, and it has nothing to do with resuming.

**FINDING: with `operator.add` the plan goes from 3 to 4, with a duplicate.**
`['research the topic', 'draft the answer', 'review the draft', 'research the
topic (done)']`. The two reducers a reader reaches for first fail in opposite
directions: one drops the steps it did not mention, the other keeps both the
old and the new copy of the step it did.

**FINDING: nothing is lost across the resume.** At the interrupt all three
compilations hold the identical three steps, and reading the interrupt's own
checkpoint config back *after* the resume returns those same three steps
byte-for-byte. The checkpointer is not the mechanism; `Command(resume=True)`
replays from a snapshot that still has the plan in it, and then `agent` writes
over it. A test that fails "if `plan` is lost across a checkpoint resume" is
watching the wrong edge.

**FINDING: `add_messages` on a `list[str]` raises.** The obvious copy-paste --
reuse the reducer already in `State` -- fails with
`NotImplementedError: Unsupported message type: <class '...ChatPromptTemplate'>`,
because `add_messages` coerces every element through the message converter and
a plain string is read as a prompt template. The error names a class the state
never contained.

Structure: `PLAN` is the three steps, `merge_by_index` the reducer that works,
`state_type` builds the `TypedDict` functionally so the annotation carries the
real reducer object, `build` compiles the graph with it, and `trial` runs
one compilation to the interrupt, resumes it, and re-reads the interrupt's own
checkpoint afterwards.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "16-langgraph-state-machines"
PLAN = ["research the topic", "draft the answer", "review the draft"]


def merge_by_index(old, new):
    """Replace step i with the incoming step i, keeping the rest."""
    merged = list(old or [])
    for index, step in enumerate(new or []):
        if index < len(merged):
            merged[index] = step
        else:
            merged.append(step)
    return merged


def state_type(reducer):
    """Built functionally: under PEP 563 a class-body annotation cannot close over
    `reducer`, and the graph needs the real object rather than a forward reference."""
    plan = list if reducer is None else Annotated[list, reducer]
    return TypedDict("PlanState", {"messages": Annotated[list, add_messages], "plan": plan})


def build(reducer):
    def planner(state):
        return {"plan": list(PLAN), "messages": [AIMessage(content="planned")]}

    def agent(state):
        return {"plan": [state["plan"][0] + " (done)"],
                "messages": [AIMessage(content="working")]}

    def quiet(state):
        return {"messages": [AIMessage(content="no plan key in this update")]}

    graph = StateGraph(state_type(reducer))
    for name, node in (("planner", planner), ("agent", agent), ("quiet", quiet)):
        graph.add_node(name, node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "agent")
    graph.add_edge("agent", "quiet")
    graph.add_edge("quiet", END)
    return graph.compile(checkpointer=MemorySaver(), interrupt_before=["agent"])


def trial(reducer, label):
    """Run to the interrupt, resume, then re-read the interrupt's own checkpoint."""
    app = build(reducer)
    config = {"configurable": {"thread_id": label}}
    list(app.stream({"messages": [HumanMessage("go")], "plan": []}, config,
                    stream_mode="updates"))
    paused = app.get_state(config)
    list(app.stream(Command(resume=True), config, stream_mode="updates"))
    return {"at_interrupt": list(paused.values["plan"]), "next": paused.next,
            "after_resume": list(app.get_state(config).values["plan"]),
            "replayed": list(app.get_state(paused.config).values["plan"])}


def solve():
    parity.load_reference(PHASE, LESSON, "main")
    arms = {"none": trial(None, "none"), "add": trial(operator.add, "add"),
            "merge": trial(merge_by_index, "merge")}
    try:
        trial(add_messages, "messages")
        wrong = None
    except Exception as problem:  # noqa: BLE001 - the class is the finding
        wrong = f"{type(problem).__name__}: {str(problem).splitlines()[0]}"
    return {"steps": len(PLAN), "arms": arms, "add_messages_error": wrong,
            "same_at_interrupt": len({tuple(arm["at_interrupt"]) for arm in arms.values()}) == 1}


def verify(result):
    arms = result["arms"]
    return [
        practice.Check(
            "ANSWER: with a merge keyed by index the plan survives at three steps",
            all([arms["merge"]["after_resume"] ==
                 ["research the topic (done)", "draft the answer", "review the draft"],
                 len(arms["merge"]["after_resume"]) == result["steps"]]),
            f"the planner writes {result['steps']} steps, agent returns only the one it "
            f"marked, and the state holds {arms['merge']['after_resume']}",
        ),
        practice.Check(
            "FINDING: with no reducer the plan goes from 3 steps to 1",
            all([len(arms["none"]["at_interrupt"]) == result["steps"],
                 arms["none"]["after_resume"] == ["research the topic (done)"]]),
            f"the default channel is last-write-wins, so a node returning just the step it "
            f"changed replaces the whole list: {arms['none']['at_interrupt']} becomes "
            f"{arms['none']['after_resume']}. This is the failure the exercise points at",
        ),
        practice.Check(
            "FINDING: with operator.add the plan goes from 3 to 4, with a duplicate",
            all([len(arms["add"]["after_resume"]) == result["steps"] + 1,
                 arms["add"]["after_resume"][0] == PLAN[0],
                 arms["add"]["after_resume"][-1] == PLAN[0] + " (done)"]),
            f"{arms['add']['after_resume']} -- the two reducers a reader reaches for first "
            "fail in opposite directions: one drops the steps it did not mention, the other "
            "keeps both copies of the step it did",
        ),
        practice.Check(
            "FINDING: nothing is lost across the resume",
            all([result["same_at_interrupt"], arms["none"]["next"] == ("agent",),
                 all(arm["replayed"] == arm["at_interrupt"] for arm in arms.values())]),
            f"at the interrupt all three compilations hold {arms['merge']['at_interrupt']}, "
            "and re-reading the interrupt's own checkpoint config after the resume returns "
            "the same three steps. The checkpointer is not the mechanism -- resume replays "
            "from a snapshot that still has the plan, and then agent writes over it",
        ),
        practice.Check(
            "FINDING: add_messages on a list[str] raises",
            all([result["add_messages_error"] is not None,
                 "NotImplementedError" in result["add_messages_error"],
                 "ChatPromptTemplate" in result["add_messages_error"]]),
            f"reusing the reducer already in State fails with "
            f"{result['add_messages_error']!r}: add_messages coerces every element through "
            "the message converter, and a plain string is read as a prompt template. The "
            "error names a class the state never contained",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
