"""Exercise 7 -- a new-ID retry never resumes, and after a late break it creates the Jira issue twice.

    Break a request-scoped SSE connection and retry with a new JSON-RPC request ID. Verify that no `Last-Event-ID` recovery path is used.

Reading of the exercise: the lesson has no transport, so a request-scoped
SSE response is modelled as a Python generator around the lesson's
`dispatch`: a `notifications/progress` event, then the tool runs, a second
progress event, then the JSON-RPC response carrying the request's id.
"Break" is the client closing the generator; the server treats that close as
cancellation, as the spec requires. The break is tried early (after event 1,
before the tool) and late (after event 2, tool done, response lost), each
retried with a new id. "No `Last-Event-ID` path" is checked three ways: the
retry sends no such header, the server writes no SSE `id:` lines to resume
from, and a retry that does send `Last-Event-ID: 1` (as an older client
would) gets the same stream as one that does not.

**ANSWER: the retry with id 8 is a fresh request, never a resumption.** The
server logs id 7 as cancelled and sends nothing more on it; the retry's
stream starts again at "started" and ends with a response whose id is 8.
There are 0 `id:` lines in any stream. The `Last-Event-ID: 1` retry yields
the same 3 events and runs the tool once more. The spec says "Resumable SSE
streams via `Last-Event-ID` are not supported" and a server should ignore the
header
(modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http,
read 2026-09-29).

**FINDING: a late break makes the retry re-run the mutation.** Broken
early, `jira.create` runs once in total (0, then 1 on retry). Broken late, it
runs twice: the cancellation arrives after the handler, and nothing in the
lesson code carries an idempotency key. Both calls return the same hard-coded
`PROJ-99`, so the duplicate is invisible in the reply. The read-only tool is
safe to retry: its two replies are identical.
"""

from __future__ import annotations

import json
import time

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "13-mcp-server-with-registry"


def post(ref, server, token, headers, body, approval, cancelled):
    """One POST, one request-scoped SSE stream; closing it cancels body['id']. `headers` (Last-Event-ID
    included) are never consulted: there is no resume path to take."""
    p = body["params"]
    note = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 0}}
    try:
        yield f"event: message\ndata: {json.dumps(note | {'params': {'progress': 0, 'message': 'started'}})}\n\n"
        out = ref.dispatch(server, token, p["name"], p["arguments"], p["_meta"], [], approval)
        yield f"event: message\ndata: {json.dumps(note | {'params': {'progress': 1, 'message': 'done'}})}\n\n"
        yield f"event: message\ndata: {json.dumps({'jsonrpc': '2.0', 'id': body['id'], **out})}\n\n"
    except GeneratorExit:
        cancelled.append(body["id"])
        raise


def read(stream, limit=None):
    events = []
    for event in stream:
        events.append(event)
        if limit is not None and len(events) == limit:
            stream.close()
            break
    return events


def scenario(ref, server, token, tool, args, approval, break_after):
    calls, cancelled = [], []
    real = server.handlers[tool]
    server.handlers[tool] = lambda a: calls.append(1) or real(a)
    headers = {"Accept": "application/json, text/event-stream", "Mcp-Method": "tools/call", "Mcp-Name": tool}

    def body(i):
        return {"jsonrpc": "2.0", "id": i, "method": "tools/call",
                "params": {"name": tool, "arguments": args, "_meta": ref.request_meta()}}

    first = read(post(ref, server, token, headers, body(7), approval, cancelled), break_after)
    before = len(calls)
    retry = read(post(ref, server, token, headers, body(8), approval, cancelled))
    after = len(calls)
    stale = read(post(ref, server, token, headers | {"Last-Event-ID": "1"}, body(9), approval, cancelled))
    final = json.loads(retry[-1].split("data: ", 1)[1])
    server.handlers[tool] = real
    return {
        "first_events": len(first), "cancelled": cancelled, "calls_before_retry": before,
        "calls_after_retry": after,
        "stale_reran": len(calls) - after, "retry_id": final["id"],
        "retry_starts_fresh": '"started"' in retry[0], "id_lines": sum("\nid:" in "\n" + e for e in first + retry),
        "stale_same_shape": [e.replace('"id": 9', '"id": 8') for e in stale] == retry,
        "retry_headers": sorted(headers), "reply": final.get("structuredContent"),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ro, rw = ref.build_readonly_server(), ref.build_destructive_server()
    later = time.time() + 3600
    ro_tok = ref.Token("u42", ro.trusted_issuer, ro.url, frozenset({"postgres:query:readonly"}), later)
    rw_tok = ref.Token("u42", rw.trusted_issuer, rw.url, frozenset({"jira:write"}), later)
    args = {"title": "new bug"}
    approval = ref.ApprovalRecord.for_action("u42", "jira.create", args, rw.url, later)
    code = parity.lesson_dir(PHASE, LESSON) / "code"
    text = "".join(f.read_text() for f in [code / "main.py", *sorted((code / "ts" / "src").glob("*.ts"))])
    return {
        "ro_early": scenario(ref, ro, ro_tok, "postgres.readonly", {"sql": "SELECT 1"}, None, 1),
        "ro_late": scenario(ref, ro, ro_tok, "postgres.readonly", {"sql": "SELECT 1"}, None, 2),
        "rw_early": scenario(ref, rw, rw_tok, "jira.create", args, approval, 1),
        "rw_late": scenario(ref, rw, rw_tok, "jira.create", args, approval, 2),
        "mentions": {k: text.lower().count(k) for k in ("last-event-id", "event-stream", "idempot")},
    }


def is_fresh(s):
    shape = (s["cancelled"], s["retry_id"], s["retry_starts_fresh"], s["id_lines"], s["stale_same_shape"])
    return shape == ([7], 8, True, 0, True) and s["stale_reran"] == 1 and "Last-Event-ID" not in s["retry_headers"]


def verify(result):
    r = result
    runs = {k: (r[k]["calls_before_retry"], r[k]["calls_after_retry"]) for k in ("rw_early", "rw_late")}
    fresh = all(is_fresh(r[k]) for k in ("ro_early", "ro_late", "rw_early", "rw_late"))
    return [
        practice.Check(
            "ANSWER: the broken id-7 stream is cancelled and the id-8 retry starts fresh, with no Last-Event-ID",
            (fresh, [r[k]["first_events"] for k in ("ro_early", "ro_late")]) == (True, [1, 2]),
            f"4 scenarios: cancelled {[r[k]['cancelled'] for k in ('ro_early', 'rw_late')]}, retry id "
            f"{r['ro_early']['retry_id']}, SSE id lines {r['ro_early']['id_lines']}, stale Last-Event-ID "
            f"stream identical {r['ro_early']['stale_same_shape']}; reference mentions {r['mentions']}",
        ),
        practice.Check(
            "FINDING: after a late break the new-id retry runs jira.create a second time",
            (runs, r["rw_late"]["reply"]["id"], r["ro_late"]["reply"], r["ro_early"]["reply"], r["mentions"])
            == (
                {"rw_early": (0, 1), "rw_late": (1, 2)},
                "PROJ-99",
                {"rows": [[1]], "sql": "SELECT 1"},
                {"rows": [[1]], "sql": "SELECT 1"},
                {"last-event-id": 0, "event-stream": 0, "idempot": 0},
            ),
            f"(handler runs before retry, after retry): {runs}; late retry reply {r['rw_late']['reply']}; "
            f"reference code mentions {r['mentions']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
