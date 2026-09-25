"""Exercise 4 — five of the six core operations are 404.

    Read the A2A spec (https://a2a-protocol.org/latest/specification/).
    Identify three things the spec mandates that this demo does not implement.

Reading of the exercise: take the mandates from the spec's current release
(1.0.0 -- the demo's card claims `a2a-0.3`) and check each against the
running demo rather than against its source, so every "does not implement"
is a status code or a missing key. Required fields come from the spec's
normative `a2a.proto`; operations from its §5.3 method-mapping table.

**ANSWER: discovery, the core operations, and error handling.**
(1) *Discovery*: the well-known URI is `/.well-known/agent-card.json` (§8.2,
§14.3); the demo serves `agent.json`, and the spec path returns 404. The card
it does serve lacks 5 of the 8 AgentCard fields the proto marks REQUIRED --
description, supportedInterfaces, capabilities, defaultInputModes,
defaultOutputModes -- and its `skills` are strings where AgentSkill requires
id, name, description and tags. (2) *Core operations*, which §3.1 says "all
A2A implementations must support": of the 6 REST endpoints for send, stream,
get, list, cancel and subscribe, the demo answers 1 -- `GET /tasks/{id}` --
and 404s the other 5. No code path ever assigns `canceled`. (3) *Errors*:
§3.3.2 says servers MUST return appropriate errors; a malformed JSON body
makes `do_POST` raise, and the client gets no response at all, only a closed
connection. And §3.3.4 requires a streaming call to an agent that does not
declare streaming to answer UnsupportedOperationError (HTTP 400); the demo
answers 404.

**FINDING: the task on the wire has neither of the spec's required shapes.**
The spec's Task requires `id` and `status` (with `status.state`); the demo
returns a flat `state`, plus `skill`, `payload` and `created_at`. Its one
`artifact` is `{type, data}`, where an Artifact requires `artifactId` and
`parts`. A spec client reading `status.state` finds nothing to poll on.

Structure: `probe()` issues one request and returns its status, or the
exception name when the server sends nothing back.
"""

from __future__ import annotations

import http.client
import inspect
import threading
import time
from http.server import HTTPServer

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "12-a2a-protocol"
CARD_REQUIRED = ("name", "description", "supportedInterfaces", "version", "capabilities",
                 "defaultInputModes", "defaultOutputModes", "skills")
SKILL_REQUIRED = ("id", "name", "description", "tags")
CORE = (("POST", "/message:send"), ("POST", "/message:stream"), ("GET", "/tasks/{id}"),
        ("GET", "/tasks"), ("POST", "/tasks/{id}:cancel"), ("POST", "/tasks/{id}:subscribe"))


class Quiet(HTTPServer):
    def handle_error(self, request, client_address):
        """The demo's traceback is the finding; keep it out of the grader's output."""


def probe(port, method, path, body=b"{}"):
    conn = http.client.HTTPConnection("localhost", port, timeout=5)
    try:
        conn.request(method, path, body=body if method == "POST" else None,
                     headers={"Content-Type": "application/json"})
        return conn.getresponse().status
    except (http.client.RemoteDisconnected, ConnectionError) as exc:
        return type(exc).__name__
    finally:
        conn.close()


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    server = Quiet(("localhost", 0), ref.A2AHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_port
    try:
        tid = ref.http_json("POST", f"http://localhost:{port}/tasks",
                            {"skill": "review-python", "payload": {"code": "x"}})["task_id"]
        core = {f"{m} {p}": probe(port, m, p.replace("{id}", tid)) for m, p in CORE}
        task = ref.http_json("GET", f"http://localhost:{port}/tasks/{tid}")
        while task["state"] != "completed":
            time.sleep(0.02)
            task = ref.http_json("GET", f"http://localhost:{port}/tasks/{tid}")
        return {
            "spec_path": probe(port, "GET", "/.well-known/agent-card.json"),
            "demo_path": probe(port, "GET", "/.well-known/agent.json"),
            "missing": [f for f in CARD_REQUIRED if f not in ref.AGENT_CARD],
            "skill_type": type(ref.AGENT_CARD["skills"][0]).__name__,
            "claims": ref.AGENT_CARD["protocol_version"], "core": core,
            "canceled": "canceled" in inspect.getsource(ref),
            "malformed": probe(port, "POST", "/tasks", body=b"{not json"),
            "task_keys": sorted(task), "artifact_keys": sorted(task["artifact"]),
        }
    finally:
        server.shutdown()


def verify(result):
    served = [k for k, v in result["core"].items() if v == 200]
    return [
        practice.Check(
            "ANSWER (1): discovery -- wrong well-known path, 5 of 8 required card fields missing",
            all([result["spec_path"] == 404, result["demo_path"] == 200,
                 len(result["missing"]) == 5, result["skill_type"] == "str"]),
            f"agent-card.json -> {result['spec_path']}, agent.json -> {result['demo_path']}; "
            f"the card claims {result['claims']} and lacks {result['missing']}; skills are "
            f"{result['skill_type']}s where AgentSkill requires {list(SKILL_REQUIRED)}",
        ),
        practice.Check(
            "ANSWER (2): 5 of the 6 core operations are 404",
            served == ["GET /tasks/{id}"] and not result["canceled"],
            f"status by endpoint: {result['core']}; the string 'canceled' appears nowhere "
            "in the module",
        ),
        practice.Check(
            "ANSWER (3): errors -- a malformed body gets no response, streaming gets 404 not 400",
            all([result["malformed"] == "RemoteDisconnected",
                 result["core"]["POST /message:stream"] == 404]),
            f"POST /tasks with '{{not json' -> {result['malformed']}; POST /message:stream "
            f"-> {result['core']['POST /message:stream']} where §3.3.4 requires 400",
        ),
        practice.Check(
            "FINDING: the task on the wire has neither of the spec's required shapes",
            all(["status" not in result["task_keys"], "state" in result["task_keys"],
                 result["artifact_keys"] == ["data", "type"]]),
            f"task keys {result['task_keys']} (spec: id, status.state); artifact keys "
            f"{result['artifact_keys']} (spec: artifactId, parts)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
