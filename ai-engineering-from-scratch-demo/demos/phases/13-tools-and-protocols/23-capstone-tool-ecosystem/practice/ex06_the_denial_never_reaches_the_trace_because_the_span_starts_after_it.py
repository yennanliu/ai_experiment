"""Exercise 6 — the denial never reaches the trace, because the span starts after it.

    Export the simulated spans through an OTel SDK to a local collector.
    Assert receipt, trace identifiers, parentage, and error status.

Reading of the exercise: the SDK is out of reach at this tier and the
collector is not, so the OTLP/HTTP payload is built by hand and posted to a
real receiver, and every assertion reads what came out of the socket rather
than what went in. Asserting an error status then turns out to require
writing one first, because the run produces none.

**ANSWER: a real collector receives the run and every assertion is made
receiver-side.** **12** spans arrive over loopback in **1** POST; every
`traceId` is 32 hex and every `spanId` 16; the parents reconstruct **2** trees
with one root each and a depth of **4**; and **1** span carries `status.code`
2. The untested layer is the SDK in between -- batching, retry, resource
detection and context propagation.

**FINDING: the denial never reaches the trace, because the span starts after
the policy check.** Bob's `generate_report` adds a `403` row to `AUDIT` and
**0** spans to `SPANS`; `gateway_call` returns before `span(...)` is called.
There is no error status to export until the span moves above the check,
which is the change that makes the trace and the audit log agree.

**FINDING: an exception orphans a span the exporter cannot tell is
unfinished.** `span()` appends on creation and `finish()` runs only on the
success path, so a tool that raises leaves `end == 0` -- **1** such span
after one bad call. Exported literally it ends before it began, which a
receiver cannot distinguish from a clock problem.

**FINDING: an exchange is two `llm.chat` spans and neither is costable.**
Input tokens land on one span and output tokens on another, so **0** of **4**
carry both, and a receiver summing cost per span sees two half records of one
call.

Structure: `to_otlp()` is the wire encoding, `collector()` the receiver, and
`identifiers()` and `parentage()` read only what the receiver holds.
"""

import contextlib
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "23-capstone-tool-ecosystem"
TOKENS = {"gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"}


def attribute(key, value):
    box = {"intValue": str(value)} if isinstance(value, int) else {"stringValue": str(value)}
    return {"key": key, "value": box}


def encode(sp):
    error = sp["attrs"].get("error.type")
    return {"traceId": sp["traceId"], "spanId": sp["spanId"], "name": sp["name"],
            "parentSpanId": sp["parentSpanId"] or "", "kind": 3 if sp["kind"] == "CLIENT" else 1,
            "startTimeUnixNano": str(sp["start"]), "endTimeUnixNano": str(sp["end"]),
            "status": {"code": 2, "message": error} if error else {"code": 1},
            "attributes": [attribute(k, v) for k, v in sp["attrs"].items()]}


def collector(received):
    """A real OTLP/HTTP receiver: the only place the assertions may look."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"partialSuccess":{}}')

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def export(server, spans):
    payload = {"resourceSpans": [{"resource": {"attributes": [
        attribute("service.name", "research-orchestrator")]}, "scopeSpans": [
            {"scope": {"name": "capstone"}, "spans": [encode(sp) for sp in spans]}]}]}
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_address[1]}/v1/traces",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status


def identifiers(spans):
    chat = [sp for sp in spans if sp["name"] == "llm.chat"]
    return {"spans": len(spans), "traces": len({sp["traceId"] for sp in spans}),
            "trace_widths": sorted({len(sp["traceId"]) for sp in spans}),
            "span_widths": sorted({len(sp["spanId"]) for sp in spans}),
            "errors": [sp["status"]["message"] for sp in spans if sp["status"]["code"] == 2],
            "chat_spans": len(chat),
            "costable": sum(TOKENS <= {a["key"] for a in sp["attributes"]} for sp in chat)}


def parentage(spans):
    by_id, tops = {sp["spanId"]: sp for sp in spans}, []
    for span in spans:
        node, levels = span, 1
        while node["parentSpanId"] in by_id:
            node, levels = by_id[node["parentSpanId"]], levels + 1
        tops.append((node["parentSpanId"], levels))
    return {"roots": sum(not sp["parentSpanId"] for sp in spans),
            "orphans": sum(bool(parent) for parent, _ in tops),
            "depth": max(levels for _, levels in tops)}


def perturb(ref, alice, bob):
    root = next(sp for sp in ref.SPANS
                if sp["traceId"] == bob["trace_id"] and sp["parentSpanId"] is None)
    ref.finish(ref.span("mcp.call", "CLIENT", bob["trace_id"], root["spanId"],
                        {"gen_ai.tool.name": "generate_report", "gateway.user": "bob",
                         "error.type": "insufficient_scope"}))
    with contextlib.suppress(KeyError):  # raises after the span, before finish()
        ref.gateway_call("tok_alice", "arxiv_search", {}, alice["trace_id"], None,
                         ref.request_meta())
    return sum(sp["end"] == 0 for sp in ref.SPANS)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    alice = ref.orchestrator("tok_alice", "summarize")
    bob = ref.orchestrator("tok_bob", "generate a report")
    denied = [sp for sp in ref.SPANS if sp["traceId"] == bob["trace_id"]
              and sp["attrs"].get("gen_ai.tool.name") == "generate_report"]
    unfinished = perturb(ref, alice, bob)
    received = []
    server = collector(received)
    status = export(server, [sp for sp in ref.SPANS if sp["end"]])
    server.shutdown()
    spans = received[0]["resourceSpans"][0]["scopeSpans"][0]["spans"]
    return {"status": status, "posts": len(received), "unfinished": unfinished,
            "report_spans": len(denied), **identifiers(spans), **parentage(spans),
            "audit_403": sum(row["decision"] == "403" for row in ref.AUDIT)}


def verify(result):
    return [
        practice.Check(
            "ANSWER: a real collector receives the run and every assertion is receiver-side",
            all([result["status"] == 200, result["posts"] == 1, result["spans"] == 12,
                 result["trace_widths"] == [32], result["span_widths"] == [16],
                 result["traces"] == 2, result["roots"] == 2, result["orphans"] == 0,
                 result["depth"] == 4, len(result["errors"]) == 1]),
            f"{result['spans']} spans arrive in {result['posts']} POST, every traceId "
            f"{result['trace_widths'][0]} hex and every spanId {result['span_widths'][0]}, "
            f"rebuilding {result['traces']} trees, {result['roots']} roots, "
            f"{result['orphans']} orphans, depth {result['depth']}, {result['errors']} at 2",
        ),
        practice.Check(
            "FINDING: the denial never reaches the trace, because the span starts after it",
            all([result["audit_403"] == 1, result["report_spans"] == 0]),
            f"bob's generate_report adds {result['audit_403']} 403 row to AUDIT and "
            f"{result['report_spans']} spans to SPANS, because gateway_call returns before "
            "span(...) is called. There is no error status to export until the span moves "
            "above the check -- the change that makes trace and audit log agree",
        ),
        practice.Check(
            "FINDING: an exception orphans a span the exporter cannot tell is unfinished",
            result["unfinished"] == 1,
            f"span() appends on creation and finish() runs only on the success path, so a "
            f"raising tool leaves end == 0 -- {result['unfinished']} such span. Exported "
            "literally it ends before it began, which no receiver can tell from a bad clock",
        ),
        practice.Check(
            "FINDING: an exchange is two llm.chat spans and neither is costable",
            all([result["chat_spans"] == 4, result["costable"] == 0]),
            f"input tokens land on one span and output tokens on another, so "
            f"{result['costable']} of {result['chat_spans']} carry both. A receiver summing "
            "cost per span sees half records, and the join needs a convention absent here",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
