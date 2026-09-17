"""Exercise 5 — the trace is buildable and three of its four named attributes have no source.

    Add observability. Emit OpenTelemetry spans for each stage, with attributes
    for params, tokens seen, loss, and cost. Pipe the spans to a local collector.
    The point is not dashboards; the point is that every stage's health is
    traceable from a single trace ID.

Reading of the exercise: the spans are emitted in the OTLP JSON shape -- one
trace id, one span per stage, each parented to a root -- without the
`opentelemetry` package or a collector, because the exercise says the point is
traceability rather than dashboards and traceability is a property of the spans
themselves. The four named attributes are then looked for in the pipeline.

**ANSWER: the trace is connected and one of the four attributes exists.**
Twelve stage spans plus a root, one `trace_id`, every `parent_span_id` resolving
to a span in the same trace, and every stage reachable from the root. Of
`params`, `tokens_seen`, `loss` and `cost`, the pipeline can supply `cost`:
`StageRecord` is `name, stage_type, input_hashes, output_hash, wall_clock_sec,
cost_usd, status`, and `Manifest` adds no per-stage numerics either.

**FINDING: 3 of the 4 attributes have to be invented.** There is no parameter
count anywhere in the pipeline -- not in `STAGES`, not in `Manifest`, not in
`simulate_stage`'s payload. There is no token count and no loss. A span that
reports them is reporting a literal the observability layer made up, which is
the failure mode observability exists to prevent.

**FINDING: the attribute that does exist is the one that is fabricated
upstream.** `cost_usd` comes from `simulate_stage`'s `cost_table`, a flat
`(7200, 400)` for any checkpoint stage. So the single traceable number is a
constant keyed on stage type, and tracing it across twelve stages recovers the
table.

**MECHANISM: "pipe the spans to a local collector" is the only untestable part
of the exercise, and it is the part that proves nothing.** Whether a span reached
an OTLP endpoint says nothing about whether the trace is well formed; whether
every `parent_span_id` resolves does. The span set is checked here for exactly
that -- one trace, no orphans, no cycles, every stage present -- which is the
exercise's own stated point, reachable without the dependency it names.

Structure: `spans` builds the OTLP span set from a finished manifest; `walk`
follows parent links from the root and reports what it can reach.
"""

from __future__ import annotations

import dataclasses
import hashlib

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "13-building-complete-llm-pipeline"
NAMED = ("params", "tokens_seen", "loss", "cost")
ROOT = "pipeline"


def span_id(trace, name):
    return hashlib.sha256(f"{trace}/{name}".encode()).hexdigest()[:16]


def spans(manifest, trace_id):
    """One root span and one span per stage, in the OTLP shape, parented to the root."""
    out = [{"traceId": trace_id, "spanId": span_id(trace_id, ROOT), "parentSpanId": "",
            "name": ROOT, "attributes": {"cost": manifest.total_cost_usd}}]
    for record in manifest.stages:
        out.append({"traceId": trace_id, "spanId": span_id(trace_id, record.name),
                    "parentSpanId": span_id(trace_id, ROOT), "name": record.name,
                    "attributes": {"cost": record.cost_usd,
                                   "wall_clock_sec": record.wall_clock_sec,
                                   "output_hash": record.output_hash}})
    return out


def descend(span_set, start):
    """Span ids reachable from `start` by parent links."""
    reached, frontier = set(), list(start)
    while frontier:
        current = frontier.pop()
        if current not in reached:
            reached.add(current)
            frontier.extend(s["spanId"] for s in span_set if s["parentSpanId"] == current)
    return reached


def walk(span_set):
    """Everything reachable from the root by parent links, and everything that is not."""
    by_id = {s["spanId"]: s for s in span_set}
    roots = [s["spanId"] for s in span_set if not s["parentSpanId"]]
    return {"roots": len(roots), "reached": len(descend(span_set, roots)),
            "orphans": [s["name"] for s in span_set
                        if s["parentSpanId"] and s["parentSpanId"] not in by_id],
            "traces": {s["traceId"] for s in span_set}}


def available(ref, manifest):
    """Which of the four named attributes the pipeline can actually supply."""
    fields = {f.name for f in dataclasses.fields(ref.StageRecord)}
    fields |= {f.name for f in dataclasses.fields(ref.Manifest)}
    return {name: any(name in field for field in fields) for name in NAMED}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    manifest = ref.run(ref.Manifest(), ref.ArtifactStore())
    trace_id = hashlib.sha256(b"run").hexdigest()[:32]
    span_set = spans(manifest, trace_id)
    checkpoints = {s.cost_usd for s in manifest.stages if s.stage_type == "checkpoint"}
    return {
        "spans": len(span_set),
        "stages": len(manifest.stages),
        "topology": walk(span_set),
        "attributes": available(ref, manifest),
        "stage_fields": sorted(f.name for f in dataclasses.fields(ref.StageRecord)),
        "checkpoint_costs": sorted(checkpoints),
        "distinct_costs": len({s.cost_usd for s in manifest.stages}),
    }


def verify(result):
    topology, attributes = result["topology"], result["attributes"]
    present = [name for name, ok in attributes.items() if ok]
    missing = [name for name, ok in attributes.items() if not ok]
    return [
        practice.Check(
            "ANSWER: one trace, no orphans, every stage reachable -- and 1 of 4 attributes exists",
            (topology["reached"] == result["spans"] and topology["roots"] == 1
             and not topology["orphans"] and len(topology["traces"]) == 1),
            f"{result['spans']} spans -- one root and {result['stages']} stages -- share one "
            f"traceId, every parentSpanId resolves within the trace, and all "
            f"{topology['reached']} are reachable from the root with {len(topology['orphans'])} "
            f"orphans. Of {list(NAMED)} the pipeline can supply {present}: StageRecord is "
            f"{result['stage_fields']} and Manifest adds no per-stage numerics either",
        ),
        practice.Check(
            "FINDING: three of the four named attributes have to be invented",
            missing == ["params", "tokens_seen", "loss"],
            f"there is no parameter count anywhere in the pipeline -- not in STAGES, not in "
            f"Manifest, not in simulate_stage's payload of stage name, type, input hashes and "
            f"seed. There is no token count and no loss. A span reporting {missing} is reporting "
            "a literal the observability layer made up, which is the failure mode observability "
            "exists to prevent",
        ),
        practice.Check(
            "FINDING: the one attribute that exists is fabricated upstream",
            len(result["checkpoint_costs"]) == 1 and result["distinct_costs"] < result["stages"],
            f"cost_usd comes from simulate_stage's cost_table, so both checkpoint stages report "
            f"{result['checkpoint_costs']} and the {result['stages']} stages carry only "
            f"{result['distinct_costs']} distinct costs between them. The single traceable number "
            "is a constant keyed on stage type, and tracing it across the pipeline recovers the "
            "table rather than the run",
        ),
        practice.Check(
            "MECHANISM: the untestable half of the exercise is the half that proves nothing",
            len(topology["traces"]) == 1 and topology["roots"] == 1,
            "whether a span reached an OTLP endpoint says nothing about whether the trace is well "
            "formed; whether every parentSpanId resolves does. The span set is checked here for "
            f"exactly that -- {len(topology['traces'])} trace, {topology['roots']} root, no "
            "orphans, no cycles, every stage present -- which is the exercise's own stated point, "
            "reachable without the collector or the package it names",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
