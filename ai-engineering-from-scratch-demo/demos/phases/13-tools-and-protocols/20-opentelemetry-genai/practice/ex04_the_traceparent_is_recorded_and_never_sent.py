"""Exercise 4 — the traceparent is recorded and never sent.

    Propagate a traceparent from a parent agent span into an MCP request's
    `_meta.traceparent` field. Verify the MCP server would see the same trace
    id.

Reading of the exercise: the lesson already builds a `traceparent` string, so
the first thing to check is where it goes -- and it goes into the client
span's own attributes, which the server never reads. `fake_mcp_call`
constructs no request at all, so there is no `_meta` for it to travel in;
propagation means building the message, not moving a field.

**ANSWER: the traceparent rides `params._meta` and the server recovers the
trace id.** The request carries
`00-<32 hex>-<16 hex>-01`, the server parses it into a remote parent, and its
SERVER span shares the client's `trace_id` while naming the client's
`span_id` as parent -- **1** trace across **2** processes.

**FINDING: the shipped traceparent is written to the span, not to a
request.** `fake_mcp_call` sets `mcp_span.attrs["traceparent"]` and returns
`{"tool": ..., "result": "ok"}` -- a dict with **2** keys and no `params`,
no `_meta`, no message of any kind. The string is correct and goes nowhere:
an attribute on the client span tells the *exporter* what the server should
have been told.

**FINDING: without it the server starts a new trace, and nothing looks
wrong.** A server that receives no traceparent mints its own `trace_id`, so
the two halves of one call become **2** traces that no query joins. Both
traces are internally well-formed, which is why a missing header is not an
error anywhere -- it is a join that silently returns nothing.

**FINDING: the format carries the sampling decision, and dropping it loses
the only field that is not an identifier.** The four parts are version,
trace id, parent id and flags; `01` means sampled. A propagation that copies
only the trace id lets a child be sampled differently from its parent, which
produces traces with holes rather than traces that are absent.

Structure: `mcp_request` is the message the lesson does not build, and
`server_span` is the far side parsing the header back into a parent.
"""

from __future__ import annotations

import contextlib
import inspect
import io

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "20-opentelemetry-genai"


def traceparent_of(span):
    return f"00-{span.trace_id}-{span.span_id}-01"


def mcp_request(span, tool):
    """The request `fake_mcp_call` never builds, with the header in _meta."""
    return {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": tool, "arguments": {},
                       "_meta": {"traceparent": traceparent_of(span)}}}


def parse(header):
    version, trace_id, parent_id, flags = header.split("-")
    return {"version": version, "trace_id": trace_id, "parent_id": parent_id,
            "sampled": flags == "01"}


def server_span(ref, request, name="mcp.server.handle"):
    """The far side: a SERVER span continuing the caller's trace, or starting one."""
    header = request["params"]["_meta"].get("traceparent")
    if header is None:
        return ref.start_span(name, "SERVER")
    remote = parse(header)
    span = ref.Span(name=name, kind="SERVER", trace_id=remote["trace_id"],
                    span_id=ref._hex(8), parent_span_id=remote["parent_id"])
    ref.SPANS.append(span)
    return span


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.SPANS.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        ref.agent_loop()
    client = next(span for span in ref.SPANS if span.name == "mcp.call")
    shipped_result = ref.fake_mcp_call(client, "get_weather")

    request = mcp_request(client, "get_weather")
    joined = server_span(ref, request)
    orphan = server_span(ref, {"params": {"_meta": {}}})
    parts = parse(request["params"]["_meta"]["traceparent"])
    source = inspect.getsource(ref.fake_mcp_call)
    return {
        "on_span": "traceparent" in client.attrs,
        "shipped_keys": sorted(shipped_result),
        "builds_request": "_meta" in source or "params" in source,
        "sent": request["params"]["_meta"]["traceparent"],
        "parts": parts, "trace_len": len(parts["trace_id"]),
        "parent_len": len(parts["parent_id"]),
        "same_trace": joined.trace_id == client.trace_id,
        "parented": joined.parent_span_id == client.span_id,
        "joined_kind": joined.kind,
        "orphan_trace": orphan.trace_id != client.trace_id,
        "traces_without": len({client.trace_id, orphan.trace_id}),
        "sampled": parts["sampled"], "fields": len(request["params"]["_meta"]
                                                   ["traceparent"].split("-")),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the traceparent rides params._meta and the server recovers the trace",
            all([result["same_trace"], result["parented"],
                 result["joined_kind"] == "SERVER",
                 result["trace_len"] == 32, result["parent_len"] == 16,
                 result["parts"]["version"] == "00"]),
            f"the request carries {result['sent']}, the server parses it into a remote "
            f"parent, and its {result['joined_kind']} span shares the client's trace id "
            f"({result['same_trace']}) while naming the client's span id as parent "
            f"({result['parented']}) -- one trace across two processes",
        ),
        practice.Check(
            "FINDING: the shipped traceparent is written to the span, not to a request",
            all([result["on_span"], result["shipped_keys"] == ["result", "tool"],
                 not result["builds_request"]]),
            f"fake_mcp_call sets the attribute on the client span and returns "
            f"{result['shipped_keys']} -- two keys, no params, no _meta, no message of any "
            "kind. The string is correct and goes nowhere: an attribute on the client span "
            "tells the exporter what the server should have been told",
        ),
        practice.Check(
            "FINDING: without it the server starts a new trace, and nothing looks wrong",
            all([result["orphan_trace"], result["traces_without"] == 2]),
            f"a server receiving no traceparent mints its own trace id, so the two halves of "
            f"one call become {result['traces_without']} traces that no query joins. Both "
            "are internally well-formed, which is why a missing header is not an error -- it "
            "is a join that silently returns nothing",
        ),
        practice.Check(
            "FINDING: the format carries the sampling decision",
            all([result["fields"] == 4, result["sampled"]]),
            f"the header has {result['fields']} parts -- version, trace id, parent id and "
            f"flags -- and 01 means sampled ({result['sampled']}). A propagation that copies "
            "only the trace id lets a child be sampled differently from its parent, which "
            "produces traces with holes rather than traces that are absent",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
