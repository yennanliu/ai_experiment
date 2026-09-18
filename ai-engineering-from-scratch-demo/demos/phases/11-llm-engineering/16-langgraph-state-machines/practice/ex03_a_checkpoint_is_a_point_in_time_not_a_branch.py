"""Exercise 3 — a checkpoint is a point in time, not a branch.

    **Hard.** Build a supervisor graph that routes between three subgraphs
    (`researcher`, `writer`, `reviewer`) using `Send`. Each subgraph has its
    own state and checkpointer. Add an `interrupt_before=["writer"]` on the
    outer graph so a human can approve the research brief. Confirm that
    time-travel from a prior checkpoint re-runs only the forked branch.

Reading of the exercise: built as written -- three compiled subgraphs, each
with its own `StateGraph` and its own `MemorySaver`, added as nodes on an outer
graph whose conditional edge returns `Send`, compiled with
`interrupt_before=["writer"]`. The nodes produce deterministic strings instead
of calling a model, because every claim in the exercise is about the graph. The
last sentence is then checked rather than assumed.

**ANSWER: the graph runs, and the interrupt fires through `Send`.** Eight node
executions, nine checkpoints, and the pause arrives with `next == ('writer',)`
after the researcher has written the brief -- so a human can approve it, which
is what the exercise wanted. `interrupt_before` is checked when the node is
scheduled, and a `Send` schedules a node.

**FINDING: `Send(node, state)` double-counts every accumulating channel.**
Passing the whole state as the payload takes the log from 7 entries to **22**
for the same eight executions: the payload carries `log`, the subgraph's
`operator.add` re-adds it on the way in, and the subgraph returns the whole
accumulated list, which `operator.add` merges again on the way out. Sending
only the fields the subgraph reads fixes it. `Send` looks like a message and
behaves like a state update.

**FINDING: each subgraph's own checkpointer is inert.** `researcher` is
compiled with a `MemorySaver` and holds **0** entries after the run; all nine
checkpoints belong to the parent. A subgraph used inside another graph gets the
parent's checkpointer, so "each subgraph has its own state and checkpointer" is
half true -- the state is its own, the checkpointer is decoration unless
`checkpointer=True` is passed at compile time.

**FINDING: time travel re-runs the whole suffix, not the forked branch.**
Forking at the `writer` checkpoint re-runs `writer -> supervisor -> reviewer ->
supervisor`: four of the original eight, every node downstream of the fork.
Forking at the `researcher` checkpoint re-runs `researcher -> supervisor` and
stops at the same interrupt again. There is no branch to isolate -- a
checkpoint is a position in a sequence, and forking replays everything after
it.

**FINDING: the fan-out `Send` exists for cannot be used here.** One payload has
to serve three nodes. Send each only what it reads and `writer` raises
`KeyError: 'brief'`; send all three every field and the step raises
`InvalidUpdateError: At key 'topic': Can receive only one value per step`,
because each subgraph's schema contains every key and returns every key while
the parent declares a reducer only for `log`. Sequencing them through the
supervisor is what makes the graph work -- which is why the supervisor runs 4
of the 8 executions, and which turns `Send` into a goto.

Structure: `KEYS` names the channels and `WORK` the three subgraphs. `leaf`
compiles one subgraph with its own checkpointer, `wire` assembles the
supervisor and the three nodes, `build` adds the sequential route and the
interrupt, `run` drives it, `travel` forks from two earlier checkpoints, and
`fan_out` is the parallel route, run once with a narrow payload and once wide.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, Send

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "16-langgraph-state-machines"
KEYS = {"topic": str, "brief": str, "draft": str, "review": str, "log": Annotated[list, operator.add]}
WORK = (("researcher", "brief", lambda s: f"brief on {s['topic']}"),
        ("writer", "draft", lambda s: f"draft from {s['brief']}"),
        ("reviewer", "review", lambda s: f"review of {s['draft']}"))
READS = ("topic", "brief", "draft")


def leaf(name, field, produce):
    graph = StateGraph(TypedDict(f"{name}State", KEYS))
    graph.add_node(name, lambda state: {field: produce(state), "log": [name]})
    graph.set_entry_point(name)
    graph.add_edge(name, END)
    return graph.compile(checkpointer=MemorySaver())  # its own, and ignored by the parent


def wire(route, loop):
    subgraphs = {n: leaf(n, f, p) for n, f, p in WORK}
    graph = StateGraph(TypedDict("GraphState", KEYS))
    graph.add_node("supervisor", lambda state: {"log": ["supervisor"]})
    for name, sub in subgraphs.items():
        graph.add_node(name, sub)
        graph.add_edge(name, "supervisor" if loop else END)
    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", route, [*subgraphs, END])
    return graph, subgraphs


def build(send_whole_state):
    def route(state):
        done, reads = set(state.get("log", [])), {k: state.get(k) for k in READS}
        for name, _, _ in WORK:
            if name not in done:
                return [Send(name, state if send_whole_state else reads)]
        return END

    graph, subgraphs = wire(route, loop=True)
    return graph.compile(checkpointer=MemorySaver(), interrupt_before=["writer"]), subgraphs


def run(app, thread):
    config = {"configurable": {"thread_id": thread}}
    start = {"topic": "prompt caching", "log": []}
    nodes = [next(iter(e)) for e in app.stream(start, config, stream_mode="updates")]
    paused = app.get_state(config)
    nodes += [next(iter(e)) for e in app.stream(Command(resume=True), config,
                                               stream_mode="updates")]
    return config, {"nodes": nodes, "paused_next": paused.next,
                    "brief_at_pause": paused.values.get("brief"),
                    "log": app.get_state(config).values["log"],
                    "checkpoints": len(list(app.get_state_history(config)))}


def travel(app, config, pending):
    snap = next(s for s in app.get_state_history(config) if s.next == (pending,))
    return [next(iter(e)) for e in app.stream(Command(resume=True), snap.config,
                                              stream_mode="updates")]


def fan_out(reads):
    graph, _ = wire(lambda state: [Send(name, {key: state.get(key) for key in reads})
                                   for name, _, _ in WORK], loop=False)
    app = graph.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": ",".join(reads)}}
    try:
        list(app.stream({"topic": "t", "log": []}, config, stream_mode="updates"))
        return None
    except Exception as problem:  # noqa: BLE001 - the exception is the finding
        return f"{type(problem).__name__}: {str(problem).splitlines()[0]}"


def solve():
    parity.load_reference(PHASE, LESSON, "main")
    lean, subgraphs = build(send_whole_state=False)
    config, measured = run(lean, "lean")
    _, inflated = run(build(send_whole_state=True)[0], "fat")
    return {**measured, "inflated_log": len(inflated["log"]),
            "inflated_nodes": len(inflated["nodes"]), "fan_out_wide": fan_out(READS),
            "sub_checkpoints": len(list(subgraphs["researcher"].checkpointer.list(None))),
            "fork_writer": travel(lean, config, "writer"),
            "fork_researcher": travel(lean, config, "researcher"),
            "supervisor_runs": measured["log"].count("supervisor"),
            "fan_out_narrow": fan_out(("topic",))}


def verify(result):
    nodes = result["nodes"]
    return [
        practice.Check(
            "ANSWER: eight executions, nine checkpoints, and the interrupt fires via Send",
            all([len(nodes) == 8, result["checkpoints"] == 9,
                 result["paused_next"] == ("writer",),
                 result["brief_at_pause"] == "brief on prompt caching"]),
            f"the run is {nodes} over {result['checkpoints']} checkpoints, pausing with "
            f"next == {result['paused_next']} and {result['brief_at_pause']!r} in hand. "
            "interrupt_before is checked when a node is scheduled, and a Send schedules it",
        ),
        practice.Check(
            "FINDING: Send(node, state) double-counts every accumulating channel",
            all([len(result["log"]) == 7, result["inflated_log"] == 22,
                 result["inflated_nodes"] == len(nodes)]),
            f"the whole state as payload takes the log from {len(result['log'])} entries to "
            f"{result['inflated_log']} for the same {result['inflated_nodes']} executions: "
            "the payload carries log, the subgraph re-adds it, and its output is merged in",
        ),
        practice.Check(
            "FINDING: each subgraph's own checkpointer is inert",
            all([result["sub_checkpoints"] == 0, result["checkpoints"] == 9]),
            f"researcher is compiled with a MemorySaver and holds "
            f"{result['sub_checkpoints']} entries; all {result['checkpoints']} checkpoints "
            "belong to the parent, so a subgraph's own is decoration unless checkpointer=True",
        ),
        practice.Check(
            "FINDING: time travel re-runs the whole suffix, not the forked branch",
            all([result["fork_writer"] == ["writer", "supervisor", "reviewer", "supervisor"],
                 result["fork_researcher"] == ["researcher", "supervisor", "__interrupt__"]]),
            f"forking at writer re-runs {result['fork_writer']} -- four of the original "
            f"{len(nodes)}, every node downstream -- and forking at researcher re-runs "
            f"{result['fork_researcher']}. A checkpoint is a position, not a branch",
        ),
        practice.Check(
            "FINDING: the fan-out Send exists for cannot be used here",
            all(["InvalidUpdateError" in (result["fan_out_wide"] or ""),
                 "KeyError" in (result["fan_out_narrow"] or ""),
                 result["supervisor_runs"] == 4]),
            f"one payload has to serve three nodes, so a narrow Send raises "
            f"{result['fan_out_narrow']!r} and a wide one {result['fan_out_wide']!r} -- each "
            "subgraph's schema holds every key and returns every key, and the parent has a "
            f"reducer only for log. Sequencing is why the supervisor runs "
            f"{result['supervisor_runs']} of {len(nodes)}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
