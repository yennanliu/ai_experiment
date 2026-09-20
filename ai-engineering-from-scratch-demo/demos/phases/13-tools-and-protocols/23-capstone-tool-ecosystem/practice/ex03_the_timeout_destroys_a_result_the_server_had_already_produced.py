"""Exercise 3 — the timeout destroys a result the server had already produced.

    Replace the writer stub with an A2A test server. Record the Agent Card,
    message request, timeout path, and returned artifact.

Reading of the exercise: "record the timeout path" is the item the stub cannot
have, because a function call has no timeout, so the test server is given a
slow route and the client a deadline shorter than it. What the recording then
shows is not that the call failed but that the client and the server disagree
about whether it did -- which is the reason A2A returns a task id instead of a
body.

**ANSWER: a real Agent Card over HTTP, a real `message/send`, a real timeout,
and an artifact the caller did not build.** The card declares **1** skill at a
loopback URL; `message/send` returns an artifact of **2** parts; the slow
route exceeds a 150ms deadline and raises. **3** requests leave the client,
**3** reach the server, **2** answers come back.

**FINDING: the timeout destroys a result the server had already produced.**
The slow call completes server-side and the client has abandoned the socket,
so `server_completed` is **3** where `client_received` is **2**. Nothing in
the response shape lets the client ask again -- which is exactly what a task
id would have given it, and what the stub's synchronous return can never
need.

**FINDING: the capstone has no failure path for delegation at all.** The
module contains **0** `except` handlers and **0** timeouts, so
`research_generate_report` cannot observe a writer that is slow, absent or
wrong. Adding the wire adds the first way for the step to fail.

**FINDING: the skill name is asserted in a span attribute and validated
nowhere.** The stub records `a2a.skill = summarize_papers`; the real card
declares that id, and the two agreeing is a coincidence the code cannot
check. A card fetch turns the attribute into a claim with a source.

Structure: `serve_writer()` is the A2A test server, `card()` and `send()` the
two client calls, and `record` keeps both sides' counts so they can disagree.
"""

from __future__ import annotations

import json
import pathlib
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "23-capstone-tool-ecosystem"
SKILL, CARD_PATH = "summarize_papers", "/.well-known/agent-card.json"
DEADLINE, SLOW = 0.15, 0.45


def serve_writer(record):
    """A real A2A peer: an Agent Card at the well-known path and message/send."""

    class Handler(BaseHTTPRequestHandler):
        def reply(self, body):
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self.reply({"protocolVersion": "0.3.0", "name": "writer-agent",
                        "url": f"{base(self.server)}/a2a", "preferredTransport": "JSONRPC",
                        "skills": [{"id": SKILL, "name": "Summarize papers"}]})

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if request["params"].get("slow"):
                time.sleep(SLOW)
            record["completed"] += 1
            self.reply({"jsonrpc": "2.0", "id": request["id"], "result": {
                "kind": "message", "role": "agent", "artifacts": [
                    {"artifactId": uuid.uuid4().hex, "name": "report.html", "parts": [
                        {"kind": "text", "text": "3 papers summarized."},
                        {"kind": "text", "text": "<h1>Report</h1>"}]}]}})

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def settle(record, target, deadline=10):
    limit = time.monotonic() + deadline
    while record["completed"] < target and time.monotonic() < limit:
        time.sleep(0.01)


def base(server):
    return f"http://127.0.0.1:{server.server_address[1]}"


def send(server, text, *, slow=False, timeout=5.0):
    """message/send over the wire; returns the result or the failure name."""
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {
        "message": {"role": "user", "parts": [{"kind": "text", "text": text}]},
        "slow": slow}}).encode()
    request = urllib.request.Request(base(server) + "/a2a", data=body,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())["result"]
    except (TimeoutError, urllib.error.URLError) as exc:
        return {"failure": type(exc).__name__}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    source = pathlib.Path(ref.__file__).read_text(encoding="utf-8")
    server = serve_writer(record := {"completed": 0})
    with urllib.request.urlopen(base(server) + CARD_PATH, timeout=5) as page:
        agent_card = json.loads(page.read())  # the well-known Agent Card, over the wire
    attempts = [send(server, "summarize the papers"),
                send(server, "summarize the papers", slow=True, timeout=DEADLINE),
                send(server, "summarize the papers")]
    settle(record, len(attempts))
    server.shutdown()

    artifact, out = attempts[0]["artifacts"][0], ref.orchestrator("tok_alice", "x")
    stub = next(sp for sp in ref.SPANS if sp["name"] == "a2a.SendMessage")
    return {
        "card_skills": [skill["id"] for skill in agent_card["skills"]],
        "card_url": agent_card["url"], "card_transport": agent_card["preferredTransport"],
        "artifact_parts": len(artifact["parts"]), "artifact_named": artifact["name"],
        "artifact_has_id": bool(artifact["artifactId"]), "sent": len(attempts),
        "failure": attempts[1].get("failure"), "server_completed": record["completed"],
        "client_received": sum("failure" not in reply for reply in attempts),
        "stub_skill": stub["attrs"]["a2a.skill"], "stub_keys": sorted(stub["attrs"]),
        "handlers": source.count("except "), "timeouts": source.count("timeout"),
        "artifact_from_peer": artifact["parts"][0]["text"] == "3 papers summarized.",
        "stub_built_locally": ref.PAPERS[0]["arxiv_id"] in out["task"]["result"]["html"]}


def verify(result):
    return [
        practice.Check(
            "ANSWER: a real card, a real message/send, a real timeout, a peer-built artifact",
            all([result["card_skills"] == [SKILL], result["artifact_parts"] == 2,
                 result["card_url"].startswith("http://127.0.0.1:"),
                 result["artifact_has_id"], result["card_transport"] == "JSONRPC",
                 result["failure"] is not None, result["artifact_from_peer"]]),
            f"the card declares {result['card_skills']} at {result['card_url']} over "
            f"{result['card_transport']}, message/send returns an identified "
            f"{result['artifact_parts']}-part {result['artifact_named']!r}, and the slow "
            f"route raises {result['failure']}",
        ),
        practice.Check(
            "FINDING: the timeout destroys a result the server had already produced",
            all([result["server_completed"] == 3, result["client_received"] == 2,
                 result["sent"] == 3]),
            f"{result['sent']} requests leave and {result['server_completed']} complete "
            f"server-side, but {result['client_received']} answers return -- the abandoned "
            "work exists and is unreachable, hence a task id instead of a body",
        ),
        practice.Check(
            "FINDING: the capstone has no failure path for delegation at all",
            all([result["handlers"] == 0, result["timeouts"] == 0]),
            f"the module contains {result['handlers']} except handlers and "
            f"{result['timeouts']} mentions of a timeout, so research_generate_report "
            "cannot observe a writer that is slow, absent or wrong -- the wire adds the "
            "step's first possible failure",
        ),
        practice.Check(
            "FINDING: the skill name is asserted in a span attribute and validated nowhere",
            all([result["stub_skill"] == SKILL, result["stub_built_locally"],
                 result["stub_keys"] == ["a2a.peer", "a2a.skill"]]),
            f"the stub records {result['stub_keys']} with skill {result['stub_skill']!r} "
            "and builds the HTML from its own PAPERS list; the card declares the same id "
            "and nothing checks that, so a card fetch is what gives the claim a source",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
