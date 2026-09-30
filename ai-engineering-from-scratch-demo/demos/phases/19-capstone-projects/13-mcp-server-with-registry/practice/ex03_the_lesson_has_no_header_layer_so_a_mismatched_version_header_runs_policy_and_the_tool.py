"""Exercise 3 -- the lesson has no header layer, so a mismatched MCP-Protocol-Version header runs policy and the tool.

    Send a valid body with a different `MCP-Protocol-Version` header. Return `-32020` and do not invoke policy or the tool.

Reading of the exercise: the lesson's `dispatch` takes a body's params and
`_meta` but no HTTP headers, so the check has to live in a Streamable HTTP
front (`post` below) that compares the mirrored headers with the body,
case-insensitively by name, before it hands the body to the lesson's
`dispatch`. Policy and tool are counted by wrapping the loaded module's
`policy_decide` and the server's handler; nothing is patched in the source.

**ANSWER: `post` returns HTTP 400 with JSON-RPC error `-32020`, and policy
and the tool run 0 times.** The body is a valid `tools/call` for
`postgres.readonly` at `2026-07-28`; the header says `2025-11-25`. The same
holds with the header naming a version
the server does not support against a supported body (still `-32020`, not
`-32022`: the mismatch is checked first), and with a mismatched `Mcp-Name`.
With matching headers, whether the names are sent as written or in lower case,
the call succeeds and policy and the tool each run once.

**FINDING: without the front, the lesson executes the mismatched request.**
Handing the same body to `dispatch` directly returns `resultType:
"complete"` with policy and tool each invoked once; the header never reaches
it. `-32020` appears nowhere in `code/main.py` or `code/ts/src`, although
the doc asks for it and the spec requires 400 + `-32020` (HeaderMismatch)
when "the header value MUST match" fails
(modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http,
read 2026-09-29).
"""

from __future__ import annotations

import time

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "13-mcp-server-with-registry"
PV = "io.modelcontextprotocol/protocolVersion"


def mismatch(headers, body):
    h = {k.lower(): v for k, v in headers.items()}
    params = body["params"]
    want = {"mcp-protocol-version": params["_meta"].get(PV), "mcp-method": body["method"]}
    if body["method"] == "tools/call":
        want["mcp-name"] = params.get("name")
    return next((f"{k} header {h.get(k)!r} != body {v!r}" for k, v in want.items() if h.get(k) != v), None)


def post(ref, server, token, headers, body, audit):
    """A Streamable HTTP front: mirrored headers first, then the lesson's dispatch."""
    reason = mismatch(headers, body)
    if reason:
        return 400, {"jsonrpc": "2.0", "id": body["id"], "error": {"code": -32020, "message": reason}}
    p = body["params"]
    out = ref.dispatch(server, token, p["name"], p["arguments"], p["_meta"], audit)
    return (400 if out.get("error", {}).get("code") == -32022 else 200), {"jsonrpc": "2.0", "id": body["id"], **out}


def request(ref, version="2026-07-28", name="postgres.readonly", method_header="tools/call"):
    body = {"jsonrpc": "2.0", "id": 42, "method": "tools/call",
            "params": {"name": "postgres.readonly", "arguments": {"sql": "SELECT 1"}, "_meta": ref.request_meta()}}
    headers = {"MCP-Protocol-Version": version, "Mcp-Method": method_header, "Mcp-Name": name,
               "Accept": "application/json, text/event-stream"}
    return headers, body


def instrument(ref, server, originals):
    """Fresh counters around the original policy and handler (never around an earlier wrapper)."""
    counts = {"policy": 0, "tool": 0}
    real_policy, real_tool = originals

    def policy(*a, **k):
        counts["policy"] += 1
        return real_policy(*a, **k)

    def tool(args):
        counts["tool"] += 1
        return real_tool(args)

    ref.policy_decide, server.handlers["postgres.readonly"] = policy, tool
    return counts


def run(ref, server, token, headers, body, originals):
    counts = instrument(ref, server, originals)
    status, reply = post(ref, server, token, headers, body, [])
    return status, reply.get("error", {}).get("code", reply.get("resultType")), dict(counts)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    server = ref.build_readonly_server()
    originals = (ref.policy_decide, server.handlers["postgres.readonly"])
    token = ref.Token("u42", server.trusted_issuer, server.url, frozenset({"postgres:query:readonly"}),
                      time.time() + 3600)
    cases = {"version header differs": request(ref, version="2025-11-25"),
             "unsupported header version": request(ref, version="2099-01-01"),
             "name header differs": request(ref, name="s3.list"), "headers match": request(ref)}
    lower_h, lower_b = request(ref)
    cases["lower-case header name"] = ({k.lower(): v for k, v in lower_h.items()}, lower_b)
    rows = {k: run(ref, server, token, h, b, originals) for k, (h, b) in cases.items()}
    counts = instrument(ref, server, originals)
    _, body = request(ref, version="2025-11-25")
    p = body["params"]
    direct = ref.dispatch(server, token, p["name"], p["arguments"], p["_meta"], [])
    code = parity.lesson_dir(PHASE, LESSON) / "code"
    sources = [code / "main.py", *sorted((code / "ts" / "src").glob("*.ts"))]
    return {"rows": rows, "direct": (direct.get("resultType"), dict(counts)),
            "mentions": sum(s.read_text().count("32020") for s in sources), "sources": len(sources)}


def verify(result):
    rows = result["rows"]
    rejected = ["version header differs", "unsupported header version", "name header differs"]
    return [
        practice.Check(
            "ANSWER: a mismatched MCP-Protocol-Version header gets 400/-32020 with 0 policy and 0 tool runs",
            all(rows[k] == (400, -32020, {"policy": 0, "tool": 0}) for k in rejected)
            and rows["headers match"] == rows["lower-case header name"] == (200, "complete", {"policy": 1, "tool": 1}),
            f"{rows}",
        ),
        practice.Check(
            "FINDING: without a header front the lesson executes the mismatched request",
            result["direct"] == ("complete", {"policy": 1, "tool": 1})
            and (result["mentions"], result["sources"]) == (0, 6),
            f"direct dispatch -> {result['direct']}; '32020' occurs {result['mentions']} times in "
            f"{result['sources']} reference source files",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
