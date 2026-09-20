"""Exercise 1 — the audit row and the forwarded id are made in different functions.

    Add trace context to the outer and forwarded request metadata and record
    the correlation in the audit event.

Reading of the exercise: correlation is only worth adding where something is
currently uncorrelated, so the first move is to look at what the audit event
already holds and what `_forward` already generates. They share **0** fields,
and they are appended by two functions that never pass each other a value --
which is the gap the trace context closes rather than decorates.

**ANSWER: one trace id across both hops, a span per hop, and both in the audit
row.** The outer request carries `traceId` and a span; `_forward` copies the
trace id and mints its own span, so the pair shares **1** trace and **2**
spans. The audit event grows from **3** keys to **6** -- principal, tool,
decision, traceId, span and the `gw-N` forwarded id.

**FINDING: two identical calls are indistinguishable in the shipped audit
log.** `{"principal", "tool", "decision"}` is the whole row, so calling
`notes.search` twice writes **2** byte-identical entries. There is no request
id, no timestamp and no trace: the log can say that alice called the tool
twice and cannot say which response belonged to which call.

**FINDING: `forwarded_request_ids` is generated and never joined to
anything.** `_forward` appends `gw-N` to it and returns the backend's result;
`handle` appends the audit row and never reads the list. The two are produced
in different functions and share no field, so the correlation the exercise
asks for does not exist to be recorded -- it has to be created.

**FINDING: three of the four audit decisions never reach a backend.** `deny`,
`rate_limit` and `pin_mismatch` are appended before `_forward` is called, and
only `allow` produces a forwarded id at all. A trace that spans forwards would
be blind to every refusal, which is the half of the log that matters most --
so the span has to open at the gateway's own entry, not at the hop.

Structure: `TracedGateway` overrides `_forward` and the audit append; `run`
drives one request and returns the outer metadata beside the row it produced.
"""

from __future__ import annotations

import secrets

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "17-mcp-gateways-and-registries"
TRACE_KEY = "io.modelcontextprotocol/traceContext"


class TracedGateway:
    """The lesson's gateway with a trace id on both hops and in the audit row."""

    def __init__(self, ref):
        self.ref, self.inner = ref, ref.Gateway()
        self.outer = None
        original = self.inner._forward

        def forwarded(canonical, arguments):
            self.inner_span = secrets.token_hex(4)
            self.forwarded_meta = {"traceId": self.outer["traceId"],
                                   "span": self.inner_span,
                                   "parent": self.outer["span"]}
            return original(canonical, arguments)

        self.inner._forward = forwarded

    def call(self, bearer, tool):
        self.outer = {"traceId": secrets.token_hex(8), "span": secrets.token_hex(4)}
        self.forwarded_meta, self.inner_span = None, None
        body, headers = self.ref.make_request("tools/call", 1,
                                              {"name": tool, "arguments": {}})
        body["params"]["_meta"][TRACE_KEY] = dict(self.outer)
        before = len(self.inner.audit)
        status, response = self.inner.handle(bearer, body, headers)
        row = dict(self.inner.audit[-1]) if len(self.inner.audit) > before else None
        if row is not None:  # the correlation the two halves never shared
            row.update(traceId=self.outer["traceId"], span=self.outer["span"],
                       forwarded=(self.inner.forwarded_request_ids[-1]
                                  if self.forwarded_meta else None))
        return status, row


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    traced = TracedGateway(ref)
    allowed = traced.call("bearer-alice", "notes.search")
    forwarded_meta = dict(traced.forwarded_meta)
    denied = traced.call("bearer-bob", "notes.create")

    plain = ref.Gateway()
    for _ in range(2):
        body, headers = ref.make_request("tools/call", 1,
                                         {"name": "notes.search", "arguments": {}})
        plain.handle("bearer-alice", body, headers)

    pinned = ref.Gateway()
    pinned.backends["notes"].tools[0]["description"] = "changed"
    body, headers = ref.make_request("tools/call", 1,
                                     {"name": "notes.search", "arguments": {}})
    pinned.handle("bearer-alice", body, headers)
    limited = ref.Gateway()
    for index in range(7):
        body, headers = ref.make_request("tools/call", index, {"name": "notes.search",
                                                               "arguments": {}})
        limited.handle("bearer-alice", body, headers)
    return {
        "allowed_status": allowed[0], "row": allowed[1],
        "row_keys": sorted(allowed[1]), "shipped_keys": sorted(plain.audit[0]),
        "same_trace": forwarded_meta["traceId"] == allowed[1]["traceId"],
        "distinct_spans": forwarded_meta["span"] != allowed[1]["span"],
        "parent": forwarded_meta["parent"] == allowed[1]["span"],
        "denied_forwarded": denied[1]["forwarded"],
        "identical_rows": plain.audit[0] == plain.audit[1], "plain_rows": len(plain.audit),
        "plain_forwarded": len(plain.forwarded_request_ids),
        "decisions": sorted({row["decision"] for row in
                             plain.audit + pinned.audit + limited.audit}),
        "forwards": len(limited.forwarded_request_ids),
        "limited_rows": len(limited.audit),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: one trace across both hops, a span each, and both in the audit row",
            all([result["allowed_status"] == 200, result["same_trace"],
                 result["distinct_spans"], result["parent"],
                 result["row_keys"] == ["decision", "forwarded", "principal", "span",
                                        "tool", "traceId"],
                 result["row"]["forwarded"] == "gw-1"]),
            f"the outer request and the forwarded one share one traceId and carry different "
            f"spans, with the inner naming the outer as parent. The audit row grows to "
            f"{len(result['row_keys'])} keys, {result['row_keys']}, including the forwarded "
            f"id {result['row']['forwarded']!r}",
        ),
        practice.Check(
            "FINDING: two identical calls are indistinguishable in the shipped audit log",
            all([result["shipped_keys"] == ["decision", "principal", "tool"],
                 result["identical_rows"], result["plain_rows"] == 2]),
            f"the shipped row is {result['shipped_keys']} and nothing else, so calling "
            f"notes.search twice writes {result['plain_rows']} byte-identical entries. No "
            "request id, no timestamp, no trace: the log can say alice called the tool twice "
            "and not which response belonged to which call",
        ),
        practice.Check(
            "FINDING: forwarded_request_ids is generated and never joined to anything",
            all([result["plain_forwarded"] == 2, result["shipped_keys"] == ["decision",
                                                                            "principal",
                                                                            "tool"]]),
            f"_forward appended {result['plain_forwarded']} ids while handle appended "
            f"{result['plain_rows']} audit rows, and the row carries none of them. The two "
            "are produced in different functions and share no field, so the correlation the "
            "exercise asks for does not exist to be recorded -- it has to be created",
        ),
        practice.Check(
            "FINDING: three of the four audit decisions never reach a backend",
            all([result["decisions"] == ["allow", "pin_mismatch", "rate_limit"],
                 result["denied_forwarded"] is None,
                 result["forwards"] < result["limited_rows"]]),
            f"the decisions observed are {result['decisions']} plus deny, and only allow "
            f"produces a forwarded id -- a refused call records {result['denied_forwarded']}. "
            f"Rate limiting alone writes {result['limited_rows']} rows against "
            f"{result['forwards']} forwards. A trace that spans forwards is blind to every "
            "refusal, so the span has to open at the gateway's entry",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
