"""Exercise 5 — stdio beats StreamableHTTP per call and at cold start, and the lesson has neither: its tools are a dict.

    Swap MCP StreamableHTTP transport for stdio. Benchmark cold-start and per-call latency. Pick a winner for local-only use.

Reading of the exercise: both transports serve the lesson's own `TOOLS`
dict as an MCP-shaped JSON-RPC server, from this file re-run as a
subprocess. Over stdio, the messages are newline-delimited JSON on the
pipe. Over StreamableHTTP, each message is a POST to `/mcp` on 127.0.0.1
with a keep-alive connection and the spec's `Accept` header. Cold start is
spawn to first `initialize` result. Per-call latency is the median of 300
`tools/call read_file` round trips; cold start is the best of 7
fresh spawns. The in-process call, which is what
`run_agent` actually does, is the floor. Wall-clock numbers vary by
machine, so the checks assert orderings and the details print the numbers.

**ANSWER: stdio wins for local-only use.** It has the lower per-call median
and the lower cold start. It needs no port, and the server dies with the
harness's pipe. On the machine this was written on, over 6 runs, the
per-call median was 0.047-0.050 ms for stdio and 0.135-0.149 ms for HTTP,
about 3x; the in-process call was 0.030 ms. Cold start was 32-33 ms against
34-35 ms. Most of that is the Python interpreter starting, and the
transport adds only the last 2-3 ms. All 900 responses are byte-identical
across the three paths.

**FINDING: the lesson ships neither transport.** `main.py` never imports a
socket, HTTP or subprocess-pipe server; `run_agent` indexes `TOOLS[name]`
in-process. The "MCP StreamableHTTP client" in the lesson's architecture
diagram, and step 3's "transport-agnostic" claim, have no code behind them.
"""

from __future__ import annotations

import http.client
import json
import pathlib
import statistics
import subprocess
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "01-terminal-native-coding-agent"
CALLS, COLD = 300, 7
CALL = {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "read_file", "arguments": {"path": "a.txt"}}}
INIT = {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}


def handle(ref, sandbox, msg):
    p = msg["params"]
    result = ({"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}} if msg["method"] == "initialize"
              else {"content": [{"type": "text", "text": ref.TOOLS[p["name"]](sandbox, **p["arguments"])}]})
    return json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result})


def serve(mode, sandbox):
    ref = parity.load_reference(PHASE, LESSON, "main")
    if mode == "--stdio":
        for line in sys.stdin:
            print(handle(ref, sandbox, json.loads(line)), flush=True)
        return

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        log_message = lambda *args: None

        def do_POST(self):
            body = handle(ref, sandbox, json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            for k, v in {"Content-Type": "application/json", "Content-Length": str(len(body))}.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body.encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print(server.server_address[1], flush=True)
    server.serve_forever()


def timed(fn, n):
    """Median wall time of fn(0..n-1) in ms, and the outputs; a tuple evaluates left to right."""
    stamps = [(time.perf_counter(), fn(i), time.perf_counter()) for i in range(n)]
    return statistics.median([b - a for a, _, b in stamps] or [0]) * 1e3, [o for _, o, _ in stamps]


def client(mode, proc):
    if mode == "--stdio":
        return lambda msg: proc.stdin.write(json.dumps(msg) + "\n") and proc.stdout.readline().strip()
    conn = http.client.HTTPConnection("127.0.0.1", int(proc.stdout.readline()))
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    return lambda msg: conn.request("POST", "/mcp", json.dumps(msg), headers) or conn.getresponse().read().decode()


def bench(mode, sandbox, calls=CALLS):
    t0 = time.perf_counter()
    proc = subprocess.Popen([sys.executable, __file__, mode, sandbox], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, bufsize=1)
    try:
        rpc = client(mode, proc)
        rpc(INIT)
        cold = (time.perf_counter() - t0) * 1e3
        call_ms, bodies = timed(lambda i: rpc({**CALL, "id": i + 1}), calls)
        return {"cold_ms": cold, "call_ms": call_ms,
                "texts": [json.loads(b)["result"]["content"][0]["text"] for b in bodies]}
    finally:
        proc.kill()
        proc.wait()


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as tmp:
        pathlib.Path(tmp, "a.txt").write_text("def f(x):\n    return x + 1\n" * 20)
        rows = {m: {**bench(m, tmp), "cold_ms": min(bench(m, tmp, 0)["cold_ms"] for _ in range(COLD))}
                for m in ("--stdio", "--http")}
        direct_ms, direct = timed(lambda i: ref.TOOLS["read_file"](tmp, path="a.txt"), CALLS)
    source = (parity.lesson_dir(PHASE, LESSON) / "code" / "main.py").read_text()
    return {"stdio": rows["--stdio"], "http": rows["--http"], "direct_ms": direct_ms,
            "identical": rows["--stdio"]["texts"] == rows["--http"]["texts"] == direct,
            "servers": [m for m in ("socket", "http", "asyncio", "selectors", "Popen(") if m in source],
            "dispatch": "TOOLS[name](sandbox, **args)" in source}


def verify(result):
    s, h, d = result["stdio"], result["http"], result["direct_ms"]
    return [
        practice.Check(
            "ANSWER: stdio wins for local-only use, on both per-call and cold start",
            result["identical"] and d < s["call_ms"] < h["call_ms"] and s["cold_ms"] < h["cold_ms"],
            f"per call: in-process {d:.4f} ms, stdio {s['call_ms']:.3f} ms, HTTP {h['call_ms']:.3f} ms; cold "
            f"start: stdio {s['cold_ms']:.0f} ms, HTTP {h['cold_ms']:.0f} ms; identical: {result['identical']}",
        ),
        practice.Check(
            "FINDING: the lesson ships no transport; tools are dispatched from an in-process dict",
            result["servers"] == [] and result["dispatch"],
            f"server/socket markers in main.py: {result['servers']}; in-process dispatch: {result['dispatch']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in ("--stdio", "--http"):
        raise SystemExit(serve(sys.argv[1], sys.argv[2]))
    raise SystemExit(practice.selfcheck(globals()))
