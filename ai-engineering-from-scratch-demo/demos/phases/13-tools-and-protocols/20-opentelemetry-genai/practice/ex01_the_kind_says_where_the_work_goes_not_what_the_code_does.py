"""Exercise 1 — the kind says where the work goes, not what the code does.

    Run `code/main.py`. Count the spans and identify which is CLIENT vs
    INTERNAL.

Reading of the exercise: counting is one number, so the split is read as a
rule and then tested against the one span that looks like a counterexample.
`tool.execute` is INTERNAL and contains a CLIENT child, and `llm.chat` is
CLIENT while its implementation is a `time.sleep` -- so the kind is a claim
about the boundary the work crosses, not about what the function body does.

**ANSWER: 9 spans in 1 trace -- 5 CLIENT and 4 INTERNAL.** CLIENT is
`llm.chat` twice and `mcp.call` three times; INTERNAL is
`agent.invoke_agent` once and `tool.execute` three times. Every span shares
one `trace_id` and every non-root span names a parent.

**FINDING: the rule is "does it leave the process", and both hard cases obey
it.** `llm.chat` is CLIENT although `fake_llm_call` only sleeps -- the kind
describes the provider call it stands for. `tool.execute` is INTERNAL
although it is the slowest kind of work here, because the crossing happens
in its `mcp.call` child. A parent can be INTERNAL and contain CLIENT, and
that nesting is the point.

**FINDING: the two spans per tool call are one operation named twice.** All
**3** `tool.execute` spans and all **3** `mcp.call` spans carry
`gen_ai.operation.name == "execute_tool"`, so a consumer grouping by
operation counts **6** tool executions where **3** happened. The
`gen_ai.tool.name` attribute is on both too.

**FINDING: `SPANS` is a module-level list with no reset.** Running
`agent_loop` twice leaves **18** spans in **2** traces, because the exporter
is a global that nothing clears. A second run in the same process double-
counts everything above.

Structure: `collect` runs one agent loop with the module's `SPANS` cleared
first, so every count is of one trace.
"""

from __future__ import annotations

import contextlib
import io
from collections import Counter

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "20-opentelemetry-genai"


def collect(ref, runs=1):
    """Run the loop `runs` times over a cleared exporter."""
    ref.SPANS.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(runs):
            ref.agent_loop()
    return list(ref.SPANS)


def by(spans, attribute):
    return dict(Counter(getattr(span, attribute) for span in spans))


def names_of(spans, kind):
    return sorted({span.name for span in spans if span.kind == kind})


def parents_of(spans, name):
    by_id = {span.span_id: span.name for span in spans}
    return sorted({by_id[s.parent_span_id] for s in spans if s.name == name})


def traces(spans):
    return len({span.trace_id for span in spans})


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    spans = collect(ref)
    twice = collect(ref, runs=2)
    root = next(span for span in spans if span.parent_span_id is None)
    operations = Counter(span.attrs.get("gen_ai.operation.name") for span in spans)
    llm_providers = {s.attrs.get("gen_ai.provider.name") for s in spans
                     if s.name == "llm.chat"}
    return {
        "count": len(spans), "kinds": by(spans, "kind"), "names": by(spans, "name"),
        "traces": traces(spans), "client": names_of(spans, "CLIENT"),
        "internal": names_of(spans, "INTERNAL"), "root": root.name,
        "parented": sum(s.parent_span_id is not None for s in spans),
        "llm_body_sleeps": llm_providers == {"openai"},
        "mcp_parents": parents_of(spans, "mcp.call"),
        "execute_tool": operations["execute_tool"],
        "tool_calls": by(spans, "name")["tool.execute"],
        "twice": len(twice), "twice_traces": traces(twice),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 9 spans in 1 trace -- 5 CLIENT and 4 INTERNAL",
            all([result["count"] == 9, result["traces"] == 1,
                 result["kinds"] == {"INTERNAL": 4, "CLIENT": 5},
                 result["client"] == ["llm.chat", "mcp.call"],
                 result["internal"] == ["agent.invoke_agent", "tool.execute"],
                 result["parented"] == 8, result["root"] == "agent.invoke_agent"]),
            f"{result['count']} spans in {result['traces']} trace, {result['kinds']}: CLIENT "
            f"is {result['client']} and INTERNAL is {result['internal']}, rooted at "
            f"{result['root']!r} with {result['parented']} of them naming a parent. The "
            f"names break down {result['names']}",
        ),
        practice.Check(
            "FINDING: the rule is whether the work leaves the process",
            all([result["llm_body_sleeps"], result["mcp_parents"] == ["tool.execute"]]),
            f"llm.chat is CLIENT although fake_llm_call only sleeps -- the kind stands for "
            f"the provider call -- and tool.execute is INTERNAL although every mcp.call's "
            f"parent is {result['mcp_parents']}, because the crossing happens in the child. "
            "A parent can be INTERNAL and contain CLIENT, and that nesting is the point",
        ),
        practice.Check(
            "FINDING: the two spans per tool call are one operation named twice",
            all([result["execute_tool"] == 6, result["tool_calls"] == 3]),
            f"{result['execute_tool']} spans carry gen_ai.operation.name 'execute_tool' -- "
            f"the {result['tool_calls']} tool.execute spans and their {result['tool_calls']} "
            "mcp.call children -- so a consumer grouping by operation counts six tool "
            "executions where three happened",
        ),
        practice.Check(
            "FINDING: SPANS is a module-level list with no reset",
            all([result["twice"] == 18, result["twice_traces"] == 2]),
            f"running agent_loop twice leaves {result['twice']} spans in "
            f"{result['twice_traces']} traces, because the exporter is a global nothing "
            "clears. A second run in the same process doubles every count above",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
