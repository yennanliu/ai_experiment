"""Exercise 5 — a node returns one dict, so a partial has nowhere to go.

    Add streaming: each node yields partial state while it runs. Print the
    deltas as they arrive.

Reading of the exercise: `NodeFn` is `Callable[[State], Update]` and `Runner`
calls `fn(state)` once, so a node has exactly one opportunity to say
anything. Streaming therefore changes the node type to a generator and the
runner to a loop over it, and the interesting part is what that does to the
two things the runner does between nodes: checkpoint, and check for a pause.

**ANSWER: generator nodes and a streaming runner.** The same four-node path
emits **7** deltas instead of **4** updates, and the final state is identical
to the non-streaming run -- **6** of **6** keys match. The deltas arrive in
order, `classify` first with its route and `send` last with its step count.

**FINDING: the checkpoint cadence did not change.** The runner still saves
once per node, so a node that emitted **3** deltas has **1** checkpoint and a
crash after delta 2 loses all three. Streaming makes the run observable
without making it any more resumable -- those are different features that
look like one.

**FINDING: a pause can now be seen mid-node.** The shipped runner checks
`_pause_reason` after `fn(state)` returns, so a node cannot stop halfway. The
streaming version sees the flag on the delta that sets it and stops with
**1** of `human_gate`'s **2** deltas applied -- which is a new state the
checkpointer has no row for.

**FINDING: the node type is the whole constraint.** `NodeFn` is
`Callable[[State], Update]`; every one of the lesson's **6** nodes is written
as a single `return`, and the runner's `update = fn(state)` reads it once.
Streaming is not an addition to this design, it is a different signature --
which is why LangGraph exposes streaming as a property of the *runtime*
rather than of the node.

Structure: `stream_node()` wraps a lesson node as a generator; `stream_run()`
is `Runner.run` with the single call replaced by a loop.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "13-langgraph-stateful-graphs"
INPUT = "the CLI crashes on ctrl-c"


def split_update(update):
    """One delta per key, so a node's progress is visible as it goes."""
    return [{key: value} for key, value in update.items()]


def stream_node(fn):
    def generator(state):
        yield from split_update(fn(state) or {})
    return generator


def stream_run(ref, graph, state, checkpointer, session="s1"):
    """Runner.run with fn(state) replaced by a loop over its deltas."""
    current, deltas, paused = graph.entry, [], None
    while current is not None and current != ref.END:
        node = stream_node(graph.nodes[current])
        for delta in node(state):
            deltas.append((current, delta))
            state = {**state, **delta}
            if state.get("_pause_reason"):
                paused = current
                break
        checkpointer.save(session, current, state)
        if paused:
            break
        current = graph._next(current, state)
    return state, deltas, paused


def shipped_run(ref, graph, state, checkpointer, session="s2"):
    try:
        return ref.Runner(graph, checkpointer).run(session, state), None
    except ref.PausedAtNode as paused:
        return paused.state, paused.node


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    approved = {"input": INPUT, "step": 0, "human_approval": True}
    streamed, deltas, _ = stream_run(ref, ref.build_graph(), dict(approved),
                                     ref.InMemoryCheckpointer())
    plain, _ = shipped_run(ref, ref.build_graph(), dict(approved),
                           ref.InMemoryCheckpointer())
    gated = ref.InMemoryCheckpointer()
    _, gate_deltas, paused_at = stream_run(
        ref, ref.build_graph(), {"input": INPUT, "step": 0, "human_approval": False},
        gated, session="s3")
    per_node = {name: sum(1 for node, _ in deltas if node == name)
                for name, _ in deltas}
    return {
        "deltas": len(deltas), "nodes": len(per_node), "per_node": per_node,
        "first": deltas[0], "last": deltas[-1],
        "matched": sum(streamed.get(key) == value for key, value in plain.items()),
        "keys": len(plain),
        "checkpoints": len(ref.InMemoryCheckpointer().history("none")) + len(per_node),
        "paused_at": paused_at,
        "gate_deltas": sum(1 for node, _ in gate_deltas if node == "human_gate"),
        "gate_total": len(ref._human_gate({"step": 0})),
        "gate_rows": len(gated.history("s3")),
        "node_type": str(inspect.signature(ref.Runner.run).return_annotation),
        "single_call": inspect.getsource(ref.Runner.run).count("fn(state)"),
        "lesson_nodes": len(ref.build_graph().nodes),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 7 deltas over 4 nodes, same final state as the shipped run",
            all([result["deltas"] == 7, result["nodes"] == 4,
                 result["matched"] == 6, result["keys"] == 6,
                 result["first"][0] == "classify", result["last"][0] == "send"]),
            f"the same four-node path emits {result['deltas']} deltas across "
            f"{result['nodes']} nodes -- {result['per_node']} -- and the final state "
            f"matches the shipped run on {result['matched']} of {result['keys']} keys. "
            f"The first delta is {result['first'][1]} and the last "
            f"{result['last'][1]}",
        ),
        practice.Check(
            "FINDING: the checkpoint cadence did not change",
            all([result["checkpoints"] == 4, result["deltas"] > result["checkpoints"],
                 result["per_node"]["classify"] == 2]),
            f"the runner still saves once per node, so {result['checkpoints']} "
            f"checkpoints cover {result['deltas']} deltas and a node that emitted "
            f"{result['per_node']['classify']} of them has one row. Streaming makes the "
            "run observable without making it more resumable",
        ),
        practice.Check(
            "FINDING: a pause can now be seen mid-node",
            all([result["paused_at"] == "human_gate", result["gate_deltas"] == 1,
                 result["gate_total"] == 2, result["gate_rows"] == 3]),
            f"the shipped runner checks _pause_reason after fn(state) returns, so a node "
            f"cannot stop halfway. The streaming version stops with "
            f"{result['gate_deltas']} of human_gate's {result['gate_total']} deltas "
            f"applied, at a state the {result['gate_rows']} checkpoint rows describe "
            "only after the fact",
        ),
        practice.Check(
            "FINDING: the node type is the whole constraint",
            all([result["single_call"] == 1, result["lesson_nodes"] == 6]),
            f"Runner.run calls fn(state) {result['single_call']} time and every one of "
            f"the lesson's {result['lesson_nodes']} nodes is written as a single return. "
            "Streaming is not an addition to this design, it is a different signature -- "
            "which is why LangGraph makes streaming a property of the runtime",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
