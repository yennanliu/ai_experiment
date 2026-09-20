"""Exercise 2 — the pin manifest already has a namespace the tool list does not.

    Add a second static backend and define the collision rule for two tools
    with the same name. Then replace both lists with real `tools/list` calls.

Reading of the exercise: "then replace both lists with real calls" is the part
that decides whether the collision rule is worth anything, so both backends are
served over loopback HTTP and the catalogue is built from two real `tools/list`
responses rather than from two literals. The rule is then applied to names that
arrived over a wire, which is where a namespace has to come from.

**ANSWER: qualify every tool as `server::name`, and refuse a bare name that
two servers answer to.** Two real POSTs return **4** tools under **3**
distinct bare names; qualified, all **4** are distinct. A call for bare
`generate_report` is refused as ambiguous rather than routed.

**FINDING: the pinned manifest is already keyed `research::name`, and nothing
else is.** `PINNED` uses the qualified form, `TOOLS`, `REQUIRED_SCOPE` and
`gateway_call` all use the bare one. So the namespace the collision rule needs
exists in the shipped code, in exactly one of four places.

**FINDING: the pin check authenticates the description, not the server.** The
archive backend ships `generate_report` with a byte-identical description, and
`pin_ok` returns True for it -- the archive's tool passes a manifest it was
never in. Hashing the text without the origin makes the pin transferable
between backends.

**FINDING: the scope table silently lends its answer to the newcomer.**
`REQUIRED_SCOPE["generate_report"]` is `research:write`, so the archive tool
inherits a scope nobody chose for it, and `gateway_call`'s `next(...)` lookup
returns the first match without noticing a second exists.

Structure: `serve()` starts a real `tools/list` endpoint per backend;
`catalogue()` merges two live responses; `resolve()` is the collision rule.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "23-capstone-tool-ecosystem"
SHARED = "Use when the user wants a full report."
ARCHIVE = [{"name": "generate_report", "description": SHARED},
           {"name": "fetch_pdf", "description": "Use when the user wants a PDF."}]


def serve(name, tools):
    """A real HTTP endpoint answering tools/list for one backend."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps({
                "jsonrpc": "2.0", "id": request["id"],
                "result": {"resultType": "complete", "tools": tools,
                           "_meta": {"io.modelcontextprotocol/serverInfo": {"name": name}}},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def tools_list(server, meta):
    """A real POST: JSON-RPC over a loopback socket, not a dict lookup."""
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list",
                          "params": {"_meta": meta}}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_address[1]}/mcp", data=payload,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())["result"]


def catalogue(responses):
    """Every tool qualified by the server that answered for it."""
    return {f"{origin}::{tool['name']}": tool
            for origin, listing in responses for tool in listing["tools"]}


def resolve(catalogue_, name):
    """The collision rule: qualified wins, ambiguous bare names are refused."""
    if "::" in name:
        return name if name in catalogue_ else "unknown tool"
    matches = [key for key in catalogue_ if key.rsplit("::", 1)[1] == name]
    if len(matches) > 1:
        return f"ambiguous: {sorted(matches)}"
    return matches[0] if matches else "unknown tool"


def bare_names(catalogue_):
    return [key.rsplit("::", 1)[1] for key in catalogue_]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    servers = {"research": serve("research", ref.TOOLS), "archive": serve("archive", ARCHIVE)}
    meta = ref.request_meta()
    listings = [(name, tools_list(server, meta)) for name, server in servers.items()]
    merged = catalogue(listings)
    for server in servers.values():
        server.shutdown()

    origins = [listing["_meta"]["io.modelcontextprotocol/serverInfo"]["name"]
               for _, listing in listings]
    archive_report = next(t for t in ARCHIVE if t["name"] == "generate_report")
    return {
        "requests": len(listings), "origins": origins,
        "ports": sorted({server.server_address[1] for server in servers.values()}),
        "tools": len(merged), "qualified": sorted(merged),
        "bare_distinct": len(set(bare_names(merged))),
        "ambiguous": resolve(merged, "generate_report"),
        "qualified_ok": resolve(merged, "archive::generate_report"),
        "unknown": resolve(merged, "archive::arxiv_search"),
        "pin_keys": sorted(ref.PINNED),
        "pin_accepts_archive": ref.pin_ok("generate_report", archive_report["description"]),
        "descriptions_identical": archive_report["description"] == ref.TOOLS[1]["description"],
        "scope_keys": sorted(ref.REQUIRED_SCOPE),
        "inherited_scope": ref.REQUIRED_SCOPE["generate_report"],
        "first_wins": next(t for t in ref.TOOLS + ARCHIVE
                           if t["name"] == "generate_report")["description"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: qualify as server::name, and refuse a bare name two servers answer to",
            all([result["requests"] == 2, result["origins"] == ["research", "archive"],
                 len(result["ports"]) == 2, result["tools"] == 4,
                 result["bare_distinct"] == 3,
                 result["ambiguous"].startswith("ambiguous:"),
                 result["qualified_ok"] == "archive::generate_report",
                 result["unknown"] == "unknown tool"]),
            f"{result['requests']} real POSTs to {result['ports']} return {result['tools']} "
            f"tools under {result['bare_distinct']} distinct bare names, {result['qualified']}. "
            f"A bare generate_report answers {result['ambiguous']} instead of routing, and "
            "the qualified form resolves",
        ),
        practice.Check(
            "FINDING: the pinned manifest is already namespaced and nothing else is",
            all([all(key.startswith("research::") for key in result["pin_keys"]),
                 result["scope_keys"] == ["arxiv_search", "generate_report"]]),
            f"PINNED is keyed {result['pin_keys']} while REQUIRED_SCOPE is keyed "
            f"{result['scope_keys']} and gateway_call matches t['name'] directly. The "
            "namespace the collision rule needs is already in the shipped code, in one of "
            "the four places that would have to agree",
        ),
        practice.Check(
            "FINDING: the pin check authenticates the description, not the server",
            all([result["pin_accepts_archive"], result["descriptions_identical"]]),
            "the archive backend ships generate_report with a byte-identical description and "
            f"pin_ok returns {result['pin_accepts_archive']} for it -- a tool passing a "
            "manifest it was never in. Hashing the text without the origin makes the pin "
            "transferable between backends, which is the opposite of what pinning is for",
        ),
        practice.Check(
            "FINDING: the scope table lends its answer to the newcomer",
            all([result["inherited_scope"] == "research:write",
                 result["first_wins"] == SHARED]),
            f"REQUIRED_SCOPE['generate_report'] is {result['inherited_scope']!r}, so the "
            "archive tool inherits a scope nobody chose for it, and gateway_call's next(...) "
            "lookup returns the first match without noticing a second exists -- a routing "
            "decision and an authorization decision made by list order",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
