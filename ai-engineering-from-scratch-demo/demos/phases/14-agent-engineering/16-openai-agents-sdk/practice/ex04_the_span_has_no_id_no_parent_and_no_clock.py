"""Exercise 4 — the span has no id, no parent and no clock.

    Wire `add_trace_processor` to a JSON logger. What shape does it emit per
    span?

Reading of the exercise: there is no `add_trace_processor`; `Runner` holds a
`trace` and appends to it, so the processor has to be a walk over the tree
after the run. That makes the shape question answerable exactly: a `Span`
has `name`, `attributes`, `children` -- **3** fields -- and the JSON a real
processor emits needs several the toy cannot supply.

**ANSWER: a processor that walks the tree and emits one JSON object per
span.** The demo's refund case yields **7** spans, each with `name`,
`depth`, `parent` and `attributes` -- parent and depth supplied by the walk,
because the span itself does not know either. Every line round-trips through
`json.loads`.

**FINDING: three of the four fields are reconstructed, not recorded.**
`Span` has **3** fields and **0** of them is an id, a parent, a start time or
a duration. A processor attached at the end can rebuild the tree shape from
nesting, and cannot rebuild anything about time -- so `duration_ms`, the
field every tracing backend sorts by, is unavailable in principle.

**FINDING: attribute keys are per-span-kind, so the log is not a table.**
Across the demo's **7** spans the union of attribute keys is **8** --
`from`, `hop`, `instructions`, `output`, `passed`, `reason`, `to`,
`user_input` -- and no key appears on every span. A JSON-lines logger emits them fine; a columnar sink gets a
sparse table with one column per span kind.

**FINDING: the trace is a mutable field, so two runs share it.** `Runner`
builds one `Span(name="run")` at construction and `run()` only appends, so
calling it twice without resetting yields **4** children where one run
leaves **2** --
the lesson's own `main()` reassigns `runner.trace` before each case, which is
the caller remembering to do the runtime's job.

Structure: `to_json()` is the processor; the runs are the lesson's own
`Runner` over its own triage agents.
"""

from __future__ import annotations

import json

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "16-openai-agents-sdk"
CASE = "I need a refund for invoice 4711"


def to_json(span, depth=0, parent=None):
    """The trace processor: one JSON object per span, tree shape from the walk."""
    rows = [json.dumps({"name": span.name, "depth": depth, "parent": parent,
                        "attributes": span.attributes}, sort_keys=True, default=str)]
    for child in span.children:
        rows += to_json(child, depth + 1, span.name)
    return rows


def triage(ref):
    billing = ref.Agent("billing", "handle refunds and invoices", ref._billing_policy)
    support = ref.Agent("support", "handle bugs and errors", ref._support_policy)
    return ref.Agent("triage", "route queries to the right specialist",
                     ref._triage_policy,
                     handoffs=[ref.Handoff(target=billing), ref.Handoff(target=support)])


def emitted(ref):
    runner = ref.Runner(input_guardrails=[ref.InputGuardrail("pii_block",
                                                             ref._pii_check)],
                        output_guardrails=[ref.OutputGuardrail("length_cap",
                                                               ref._length_check)],
                        trace=ref.Span(name="run", attributes={"user_input": CASE}))
    runner.run(triage(ref), CASE)
    parsed = [json.loads(row) for row in to_json(runner.trace)]
    keys = sorted({key for row in parsed for key in row["attributes"]})
    return {
        "spans": len(parsed), "parsed": len(parsed), "fields": sorted(parsed[0]),
        "keys": keys, "depths": sorted({row["depth"] for row in parsed}),
        "shared": [k for k in keys if all(k in r["attributes"] for r in parsed)],
        "roots": sum(1 for row in parsed if row["parent"] is None),
    }


def reuse(ref, runs):
    runner = ref.Runner(trace=ref.Span(name="run"))
    for _ in range(runs):
        runner.run(triage(ref), CASE)
    return len(runner.trace.children)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    fields = list(ref.Span.__dataclass_fields__)
    return {
        **emitted(ref), "span_fields": fields,
        "time_fields": [f for f in fields
                        if "time" in f or "duration" in f or "id" in f],
        "reused_children": reuse(ref, 2), "single_children": reuse(ref, 1),
        "reused_runs": 2,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: seven spans, four fields each, all valid JSON",
            all([result["spans"] == 7, result["parsed"] == 7,
                 result["fields"] == ["attributes", "depth", "name", "parent"],
                 result["roots"] == 1, result["depths"] == [0, 1, 2]]),
            f"the refund case yields {result['spans']} spans, each emitted as "
            f"{result['fields']} and parsing cleanly, with {result['roots']} root and "
            f"depths {result['depths']}. parent and depth come from the walk, because "
            "the span does not know either",
        ),
        practice.Check(
            "FINDING: three of the four fields are reconstructed, not recorded",
            all([result["span_fields"] == ["name", "attributes", "children"],
                 result["time_fields"] == []]),
            f"Span carries {result['span_fields']} and "
            f"{len(result['time_fields'])} of them is an id, a parent, a start time or "
            "a duration. The tree shape is rebuildable from nesting; duration_ms, the "
            "field every tracing backend sorts by, is unavailable in principle",
        ),
        practice.Check(
            "FINDING: attribute keys are per-span-kind, so the log is not a table",
            all([len(result["keys"]) == 8, result["shared"] == [],
                 set(result["keys"]) >= {"passed", "reason", "hop"}]),
            f"across the {result['spans']} spans the union of attribute keys is "
            f"{result['keys']} -- {len(result['keys'])} of them -- and "
            f"{len(result['shared'])} appear on every span. JSON lines handle that; a "
            "columnar sink gets one column per span kind",
        ),
        practice.Check(
            "FINDING: the trace is a mutable field, so two runs share it",
            all([result["reused_children"] == 4, result["single_children"] == 2,
                 result["reused_children"] == 2 * result["single_children"]]),
            f"Runner builds one Span at construction and run() only appends, so one run "
            f"leaves {result['single_children']} children and "
            f"{result['reused_runs']} runs leave {result['reused_children']} on the same "
            "trace. The lesson's own main() reassigns runner.trace before each case -- "
            "the caller doing the runtime's job",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
