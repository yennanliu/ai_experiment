"""Exercise 2 -- tools/list is byte-stable, but its "private" cache hands a 1-scope caller 2 tools.

    Send `tools/list` twice with identical inputs and prove byte-stable tool order. Then expire `ttlMs` and refresh.

Reading of the exercise: "byte-stable" is read at the wire: the result of
the lesson's `MCPServer.tools_list` is serialized with `json.dumps` and the
bytes compared, for two calls on one server and for a second server
registered in the reverse order. "Expire and refresh" needs a client cache,
so a small TTL cache with an injected clock (milliseconds, no sleeping) keys
entries by `cacheScope` and honours the result's own `ttlMs`.

**ANSWER: two calls give identical 744-byte results, and so does a server
registered in reverse order.** Order is `postgres.readonly`, `s3.list`.
The cache serves the list from memory at t = 0 and t = 59,999 ms, refetches
at t = 60,000 ms (the result's `ttlMs`), and the refreshed bytes are the same
744 bytes. The server is called 2 times over 4 reads.

**FINDING: the "private" list is not per-caller.** `tools_list` takes no
token, so it cannot vary by authorization, yet it says `cacheScope:
"private"`. A token holding only `s3:list` gets both tools; `dispatch` then
denies 1 of the 2 (`missing scope: postgres:query:readonly`). The spec lets
the set "vary by the authorization presented on the request"
(modelcontextprotocol.io/specification/2026-07-28/server/tools, read 2026-09-29).

**FINDING: the lesson's two reference models disagree on the cache hint.**
The Python model lists with `ttlMs` 60,000 and `private`; the TypeScript
`handleToolsList` uses 300,000 and `public`, for the same kind of list.
"""

from __future__ import annotations

import json
import re
import time

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "13-mcp-server-with-registry"


def wire(ref, server):
    return json.dumps(server.tools_list(ref.request_meta()))


def reversed_server(ref):
    """Same tools, registered in the opposite order."""
    src = ref.build_readonly_server()
    dst = ref.MCPServer(src.name, src.title, src.description, src.version, src.url, src.trusted_issuer)
    for name in reversed(list(src.tools)):
        dst.register(src.tools[name], src.handlers[name])
    return dst


def cached_reads(fetch, times_ms):
    """A client cache: reuse the body until its own ttlMs has elapsed."""
    entry, calls, log = None, 0, []
    for now in times_ms:
        if entry is None or now >= entry["at"] + entry["ttl"]:
            body = fetch()
            calls += 1
            parsed = json.loads(body)
            entry = {"at": now, "ttl": parsed["ttlMs"], "scope": parsed["cacheScope"], "body": body}
            log.append((now, "fetch"))
        else:
            log.append((now, "hit"))
    return entry, calls, log


def scope_gap(ref, server):
    token = ref.Token("u7", server.trusted_issuer, server.url, frozenset({"s3:list"}), time.time() + 3600)
    listed = [t["name"] for t in server.tools_list(ref.request_meta())["tools"]]
    args = {"postgres.readonly": {"sql": "SELECT 1"}, "s3.list": {"bucket": "b"}}
    denied = [
        (n, r["error"]["message"])
        for n in listed
        if "error" in (r := ref.dispatch(server, token, n, args[n], ref.request_meta(), []))
    ]
    return listed, denied


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    server = ref.build_readonly_server()
    first, second, rev = wire(ref, server), wire(ref, server), wire(ref, reversed_server(ref))
    entry, calls, log = cached_reads(lambda: wire(ref, server), [0, 59_999, 60_000, 60_001])
    ts = (parity.lesson_dir(PHASE, LESSON) / "code" / "ts" / "src" / "protocol.ts").read_text()
    ts_hint = re.search(r"handleToolsList[\s\S]*?ttlMs: ([\d_]+), cacheScope: \"(\w+)\"", ts)
    parsed = json.loads(first)
    return {
        "same": first == second, "same_rev": first == rev, "bytes": len(first.encode()),
        "order": [t["name"] for t in parsed["tools"]],
        "py_hint": (parsed["ttlMs"], parsed["cacheScope"]),
        "log": log, "calls": calls, "refreshed_same": entry["body"] == first,
        "gap": scope_gap(ref, server),
        "ts_hint": (int(ts_hint.group(1).replace("_", "")), ts_hint.group(2)),
    }


def verify(result):
    r = result
    listed, denied = r["gap"]
    return [
        practice.Check(
            "ANSWER: two tools/list results are byte-identical; ttlMs expiry triggers one refetch",
            (r["same"], r["same_rev"], r["bytes"], r["order"]) == (True, True, 744, ["postgres.readonly", "s3.list"])
            and r["log"] == [(0, "fetch"), (59_999, "hit"), (60_000, "fetch"), (60_001, "hit")]
            and (r["calls"], r["refreshed_same"]) == (2, True),
            f"identical {r['same']}, reverse-registered identical {r['same_rev']}, {r['bytes']} bytes, "
            f"order {r['order']}; cache log {r['log']}; {r['calls']} server calls; refresh identical "
            f"{r['refreshed_same']}",
        ),
        practice.Check(
            "FINDING: the 'private' tools/list is not per-caller: a 1-scope token sees 2 tools",
            r["py_hint"] == (60_000, "private")
            and listed == ["postgres.readonly", "s3.list"]
            and denied == [("postgres.readonly", "missing scope: postgres:query:readonly")],
            f"hint {r['py_hint']}; an s3:list-only token lists {listed}, dispatch denies {denied}",
        ),
        practice.Check(
            "FINDING: the Python and TypeScript reference models disagree on the tools/list cache hint",
            r["ts_hint"] == (300_000, "public") and r["py_hint"] != r["ts_hint"],
            f"Python {r['py_hint']} vs TypeScript {r['ts_hint']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
