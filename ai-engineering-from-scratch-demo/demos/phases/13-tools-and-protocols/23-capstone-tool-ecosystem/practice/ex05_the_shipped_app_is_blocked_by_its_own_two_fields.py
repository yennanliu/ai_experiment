"""Exercise 5 — the shipped App is blocked by its own two fields.

    Build a minimal MCP App and verify `app.callServerTool` in a browser with
    a restrictive CSP and explicit permissions.

Reading of the exercise: a browser is out of reach at this tier, so the App is
split at the boundary a browser actually owns. The CSP is emitted as a real
response header and evaluated against the real script bytes, and
`app.callServerTool` is exercised against the host side that receives it --
both fully checkable. What is left for a browser is enforcement, and saying so
is only worth anything once the parts either side of it are exercised, because
the shipped App fails on this side first.

**ANSWER: a minimal App whose CSP names the script by hash and whose
permissions name the tool by name.** The served response carries
`script-src 'sha256-...'` as a real header over loopback, the inline script's
digest matches, and the bridge allows `arxiv_search` and refuses
`generate_report` -- **1** allowed of **2** attempted. The remaining claim is
enforcement: a browser deciding to honour the header, which no assertion here
can make.

**FINDING: the shipped App is blocked by its own two fields.** Its `ui` block
declares `default-src 'self'` with no `script-src`, no `'unsafe-inline'` and
no hash, while its HTML embeds an inline `<script>`; a conforming renderer
refuses to run it. Its `permissions` list is empty, so `app.callServerTool`
has **0** tools it may call. Both fields are correct-looking and together
they describe an App that can neither run nor call.

**FINDING: hashing the script binds its bytes, so a one-character edit is a
silent outage.** Adding a single space flips the CSP verdict from allow to
refuse, with no error the server can see. The hash is the tightest of the
three options and it moves the failure to deploy time, which is the trade a
nonce is usually bought to avoid.

**FINDING: the `ui://` reference names nothing resolvable.** The result
carries `ui://report/current` in its content and the HTML beside it in a
sibling key, and the module defines **0** ways to read a resource. The URI is
a label on a field that was already delivered.

Structure: `serve_app()` emits the real header; `allows_inline()` is the CSP
arithmetic; `call_server_tool()` is the host half of the bridge.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "23-capstone-tool-ecosystem"
SCRIPT = "window.app.callServerTool('arxiv_search', {query: 'agent'});"
PERMISSIONS = ["arxiv_search"]


def digest(script):
    return "'sha256-" + base64.b64encode(hashlib.sha256(script.encode()).digest()).decode() + "'"


def allows_inline(csp, script):
    """Would a conforming renderer run this inline script under this policy?"""
    sources = csp.get("script-src", csp.get("default-src", "")).split()
    return "'unsafe-inline'" in sources or digest(script) in sources


def app_html(script):
    return f"<!doctype html><html><body><div id=r></div><script>{script}</script></body></html>"


def serve_app(html, csp):
    """The App resource with its policy on the response, where a browser reads it."""
    header = "; ".join(f"{name} {value}" for name, value in csp.items())

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Security-Policy", header)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def call_server_tool(ref, permissions, name, args):
    """The host half of app.callServerTool: the allowlist is checked before the gateway."""
    if name not in permissions:
        return {"error": "permission_denied", "permission": name}
    meta = ref.request_meta(tasks=name == "generate_report")
    return ref.gateway_call("tok_alice", name, args, ref._hex(16), None, meta)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    csp = {"default-src": "'self'", "script-src": digest(SCRIPT)}
    html = app_html(SCRIPT)
    server = serve_app(html, csp)
    with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_address[1]}/ui", timeout=5) as response:
        served, header = response.read().decode(), response.headers["Content-Security-Policy"]
    server.shutdown()

    allowed = call_server_tool(ref, PERMISSIONS, "arxiv_search", {"query": "agent"})
    refused = call_server_tool(ref, PERMISSIONS, "generate_report", {})
    shipped = ref.tasks_get(
        ref.research_generate_report({}, ref._hex(16), None)["taskId"],
        ref.request_meta(tasks=True))["result"]
    return {
        "header": header, "served_matches": served == html,
        "hash_in_header": digest(SCRIPT) in header,
        "allows": allows_inline(csp, SCRIPT),
        "edited_allows": allows_inline(csp, SCRIPT + " "),
        "attempted": 2, "allowed_tools": PERMISSIONS,
        "allowed_ok": json.loads(allowed["content"][0]["text"])[0]["arxiv_id"],
        "refused": refused.get("error"),
        "shipped_csp": shipped["ui"]["csp"], "shipped_permissions": shipped["ui"]["permissions"],
        "shipped_inline": "<script>" in shipped["html"],
        "shipped_allows": allows_inline(shipped["ui"]["csp"], "/* anything */"),
        "ui_uri": shipped["content"][1]["uri"],
        "readers": [name for name in dir(ref) if "resource" in name.lower()],
        "html_sibling": "html" in shipped,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the CSP names the script by hash and the permissions name the tool",
            all([result["hash_in_header"], result["served_matches"], result["allows"],
                 result["allowed_ok"] == "2604.01055", result["refused"] == "permission_denied",
                 result["allowed_tools"] == ["arxiv_search"]]),
            f"the served response carries {result['header']!r} over loopback and the inline "
            f"script's digest matches, so the policy admits it. The bridge allows "
            f"{result['allowed_tools']} and refuses the other with {result['refused']!r} -- "
            f"1 of {result['attempted']}. What no assertion here reaches is enforcement: a "
            "browser choosing to honour the header",
        ),
        practice.Check(
            "FINDING: the shipped App is blocked by its own two fields",
            all([result["shipped_csp"] == {"default-src": "'self'"},
                 result["shipped_permissions"] == [], result["shipped_inline"],
                 not result["shipped_allows"]]),
            f"the shipped ui block declares {result['shipped_csp']} with no script-src, no "
            f"unsafe-inline and no hash while its HTML embeds an inline script, so a "
            f"conforming renderer refuses to run it; its permissions list is "
            f"{result['shipped_permissions']}, so callServerTool has nothing it may call. "
            "Both fields look right and together they describe an App that can neither run "
            "nor call",
        ),
        practice.Check(
            "FINDING: hashing the script binds its bytes, so a one-character edit is an outage",
            all([result["allows"], not result["edited_allows"]]),
            "adding one space to the script flips the verdict from allow to refuse, with no "
            "error the server can see. The hash is the tightest of the three options and it "
            "moves the failure to deploy time -- which is the trade a nonce is usually "
            "bought to avoid",
        ),
        practice.Check(
            "FINDING: the ui:// reference names nothing resolvable",
            all([result["ui_uri"] == "ui://report/current", result["readers"] == [],
                 result["html_sibling"]]),
            f"the result carries {result['ui_uri']!r} in its content and the HTML beside it "
            f"in a sibling key, and the module defines {len(result['readers'])} ways to read "
            "a resource. The URI is a label on a field that was already delivered, which is "
            "why nothing breaks when it points nowhere",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
