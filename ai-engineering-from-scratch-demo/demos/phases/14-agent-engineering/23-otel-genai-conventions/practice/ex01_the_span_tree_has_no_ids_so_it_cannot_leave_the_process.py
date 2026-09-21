"""Exercise 1 — the span tree has no ids, so it cannot leave the process.

    Instrument your Lesson 01 ReAct loop with `invoke_agent` (INTERNAL) +
    per-tool spans. Send to a Jaeger instance.

Reading of the exercise: the instrumenting is straightforward and the sending
is where the design shows. `Span` records a name, a kind, attributes,
children and two counters -- and parent-child lives in a Python list, which
an exporter cannot serialise. So the loop is instrumented as asked, and then
the export is attempted, because that is the step that says what the
convention is actually for.

**ANSWER: 1 invoke_agent INTERNAL span over 6 chat spans and 5 tool spans.**
Wrapping Lesson 01's `ToyLLM.respond` and `ToolRegistry.dispatch` gives **12**
spans for the shipped script, named `invoke_agent research_bot`,
`chat` and `tool_call {name}`, with the three tools appearing **5** times
between them. The agent span is INTERNAL because the loop is in-process.

**FINDING: `Span` has 6 fields and none of them is an identity.** Flattening
the tree for export yields **12** rows and **0** recoverable parent links:
the only thing connecting a tool span to its agent is a Python object
reference. An OTLP exporter needs `trace_id`, `span_id` and
`parent_span_id` -- **3** fields that have to be added before any of this
reaches Jaeger, which is exactly the lesson's "spans without parent links"
pitfall in the emitter rather than in the caller.

**FINDING: `end_span` takes no argument, so it closes whatever is on top.**
Ending one span too many pops the root; the next `start_span` then raises
`IndexError` from an empty stack rather than reporting an unbalanced trace.
The failure surfaces **1** call later, in unrelated code.

**FINDING: the timestamps are durations, not positions.** `start_ns` comes
from `perf_counter_ns`, whose zero is arbitrary, and the module never calls
`time_ns` -- so every `start_ns` is smaller than the wall clock and no span
can be placed on a timeline beside a log line or another service. Jaeger
wants `start_time_unix_nano`; what is recorded cannot be converted into one.

Structure: `instrument()` wraps the Lesson 01 objects; `flatten()` is the
exporter that cannot find a parent.
"""

from __future__ import annotations

import inspect
import time

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "23-otel-genai-conventions"
AGENT = "research_bot"
MODEL = "claude-opus-4-6"


def instrument(tracer, loop, agent_name=AGENT):
    """invoke_agent around the run, chat per model call, tool_call per dispatch."""
    respond, dispatch = loop.llm.respond, loop.tools.dispatch

    def traced_respond(history):
        tracer.start_span("chat", attributes={
            "gen_ai.operation.name": "chat", "gen_ai.provider.name": "anthropic",
            "gen_ai.request.model": MODEL, "gen_ai.response.model": MODEL})
        reply = respond(history)
        tracer.end_span()
        return reply

    def traced_dispatch(call):
        tracer.start_span(f"tool_call {call.name}", attributes={
            "gen_ai.operation.name": "tool_call", "gen_ai.tool.name": call.name})
        result = dispatch(call)
        tracer.end_span()
        return result

    loop.llm.respond, loop.tools.dispatch = traced_respond, traced_dispatch
    return agent_name


def run_traced(ref, loop):
    tracer = ref.Tracer(capture_inline=False)
    name = instrument(tracer, loop)
    tracer.start_span(f"invoke_agent {name}", kind="INTERNAL", attributes={
        "gen_ai.agent.name": name, "gen_ai.operation.name": "invoke_agent",
        "gen_ai.provider.name": "anthropic", "gen_ai.request.model": MODEL})
    answer = loop.run("what is the total including tax?")
    tracer.end_span()
    return tracer, answer


def flatten(span, rows=None):
    """What an exporter can see: a flat list, with no way back to the parent."""
    rows = [] if rows is None else rows
    if span.name != "__root__":
        rows.append({"name": span.name, "kind": span.kind,
                     "attrs": sorted(span.attributes)})
    for child in span.children:
        flatten(child, rows)
    return rows


def unbalanced(ref):
    tracer = ref.Tracer()
    tracer.start_span("invoke_agent x")
    tracer.end_span()
    tracer.end_span()
    try:
        tracer.start_span("chat")
    except IndexError as exc:
        return type(exc).__name__
    return "no error"


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    loop_ref = parity.load_reference(PHASE, "01-the-agent-loop", "main")
    tracer, answer = run_traced(ref, loop_ref.build_demo_agent())
    rows = flatten(tracer.root)
    root = tracer.root.children[0]
    return {
        "answer": answer, "spans": len(rows),
        "agent_span": (root.name, root.kind),
        "chats": sum(row["name"] == "chat" for row in rows),
        "tools": sum(row["name"].startswith("tool_call") for row in rows),
        "tool_names": sorted({row["name"] for row in rows
                              if row["name"].startswith("tool_call")}),
        "span_fields": list(ref.Span.__dataclass_fields__),
        "id_fields": [f for f in ref.Span.__dataclass_fields__ if f.endswith("id")],
        "parent_links": sum("parent" in key for row in rows for key in row["attrs"]),
        "needed": ["trace_id", "span_id", "parent_span_id"],
        "unbalanced": unbalanced(ref),
        "monotonic": root.start_ns < time.time_ns(),
        "uses_epoch": "time.time_ns" in inspect.getsource(ref),
        "uses_perf_counter": "perf_counter_ns" in inspect.getsource(ref),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: one invoke_agent INTERNAL span over 6 chat and 5 tool spans",
            all([result["spans"] == 12, result["chats"] == 6, result["tools"] == 5,
                 result["agent_span"] == (f"invoke_agent {AGENT}", "INTERNAL"),
                 len(result["tool_names"]) == 3,
                 result["answer"] == "the total including 15% tax is 138.0"]),
            f"wrapping Lesson 01's respond and dispatch gives {result['spans']} spans: "
            f"{result['agent_span'][0]} as {result['agent_span'][1]}, "
            f"{result['chats']} chat spans and {result['tools']} tool_call spans across "
            f"{len(result['tool_names'])} tools, with the run still answering "
            f"{result['answer']!r}",
        ),
        practice.Check(
            "FINDING: Span has 6 fields and none of them is an identity",
            all([len(result["span_fields"]) == 6, result["id_fields"] == [],
                 result["parent_links"] == 0, result["spans"] == 12]),
            f"Span carries {result['span_fields']} -- {len(result['id_fields'])} of them "
            f"an id -- so flattening the tree gives {result['spans']} rows with "
            f"{result['parent_links']} recoverable parent links. An exporter needs "
            f"{result['needed']} before any of this reaches Jaeger",
        ),
        practice.Check(
            "FINDING: end_span takes no argument, so it closes whatever is on top",
            result["unbalanced"] == "IndexError",
            f"ending one span too many pops the root, and the next start_span raises "
            f"{result['unbalanced']} from an empty stack rather than reporting an "
            "unbalanced trace. The failure surfaces one call later, in unrelated code",
        ),
        practice.Check(
            "FINDING: the timestamps are durations, not positions",
            all([result["monotonic"] is True, result["uses_epoch"] is False,
                 result["uses_perf_counter"] is True,
                 "start_ns" in result["span_fields"],
                 "start_time_unix_nano" not in result["span_fields"]]),
            f"start_ns comes from perf_counter_ns (present: "
            f"{result['uses_perf_counter']}), whose zero is arbitrary, and the module "
            f"never calls time_ns ({result['uses_epoch']}) -- so every start_ns is "
            "smaller than the wall clock and no span can be placed on a timeline beside a "
            "log line. Jaeger wants start_time_unix_nano; this cannot become one",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
