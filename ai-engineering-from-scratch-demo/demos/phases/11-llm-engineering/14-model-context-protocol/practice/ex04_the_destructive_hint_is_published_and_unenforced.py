"""Exercise 4 — the destructive hint is published and unenforced.

    Return `input_required` from a tool that needs user confirmation. Retry the
    original call with a new ID, an `inputResponses` entry, and the exact
    `requestState` instead of inventing a server-to-client JSON-RPC request.

Reading of the exercise: `delete_user` is the tool that needs confirmation --
the lesson already marks it `destructive=True` -- so the flow is built on it.
Elicitation is added where it has to be added rather than where it would be
convenient, and the three rules of the retry are each tested for whether the
server can enforce them or only hope for them.

**ANSWER: the ask deletes nothing and the retry deletes once.** The first
`tools/call` comes back `resultType: "input_required"` carrying one
`inputRequests` entry and a `requestState`, with the delete list still empty;
the retry, with a fresh JSON-RPC id, an `inputResponses` entry and the same
`requestState`, comes back `complete` and performs the delete.

**FINDING: `input_required` cannot come from a tool handler.** `_complete`
writes `"resultType": "complete"` into every result it builds, and all six
methods go through it -- `server/discover`, `tools/list`, `tools/call`,
`resources/list`, `resources/read`, `prompts/list` all answer `complete`.
`tools/call` additionally hardcodes `isError: False` and JSON-encodes the
handler's return value into `content[0].text`. A handler has no way to reach
the field the exercise asks it to set; elicitation is a change to the result
builder, not to the tool.

**FINDING: the server never validates the extra params.** Sent to the
unmodified server, a `tools/call` carrying `inputResponses` and `requestState`
returns an ordinary `complete` result: `handle` reads `name`, `arguments` and
`_meta` and ignores everything else in `params`. The retry contract is entirely
application convention -- which is the right answer, and also means a typo in
`requestState` is not an error.

**FINDING: "a new ID" is a convention the server cannot enforce.** Because the
server keeps no per-request state, replaying the retry with the *original* id
succeeds identically and deletes a second time. And the alternative the exercise warns against is not
merely discouraged: `MCPClient` exposes one method, `request`, and `handle`
returns exactly one response per message, so there is no channel on which a
server-to-client JSON-RPC request could travel.

**FINDING: the destructive annotation is decoration.** `tools/list` publishes
`{"destructiveHint": true}` for `delete_user`, and the `tools/call` branch
never reads `tool.destructive` -- so an unconfirmed delete straight to the
unmodified server returns `complete` and reports the deletion. The confirmation
gate has to be written into the wrapper because the flag does nothing.

Structure: `CONFIRM` is the sentinel a handler returns to ask for input and
`guarded_delete` is `delete_user` behind that gate. `Elicit` is a server
wrapper that owns the one field the lesson's result builder lacks: it turns the
sentinel into an `input_required` result and folds a retry's `inputResponses`
back into the arguments. `flow` walks ask, retry and replay; `unguarded` asks
the lesson's own server, unmodified, for the same things; `METHODS` is every
method it implements.
"""

from __future__ import annotations

import json

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "14-model-context-protocol"
CONFIRM, DELETED = "__input_required__", []
SCHEMA = {"type": "object", "properties": {"user_id": {"type": "integer"}},
          "required": ["user_id"]}


def guarded_delete(user_id, confirmed=False):
    if confirmed:
        DELETED.append(user_id)
        return {"deleted": user_id}
    return {CONFIRM: {"name": "confirmed", "prompt": f"Delete user {user_id}?"},
            "requestState": {"tool": "delete_user", "user_id": user_id, "step": 1}}


class Elicit:
    def __init__(self, server):
        self.server = server

    def handle(self, message):
        sent = message.get("params", {}).get("inputResponses", ())
        if sent:
            message = json.loads(json.dumps(message))
            message["params"]["arguments"].update({e["name"]: e["value"] for e in sent})
        response = self.server.handle(message)
        result = response.get("result") if response else None
        payload = json.loads(result["content"][0]["text"]
                             if result and result.get("content") else "{}")
        if isinstance(payload, dict) and CONFIRM in payload:
            response["result"] = {"resultType": "input_required", "_meta": result["_meta"],
                                  "inputRequests": [payload[CONFIRM]],
                                  "requestState": payload["requestState"]}
        return response


def build(ref):
    DELETED.clear()
    server = ref.MCPServer("practice")
    for name in ("tools", "resources", "prompts"):
        getattr(server, name).update(getattr(ref.server, name))
    server.tools["delete_user"] = ref.Tool("delete_user", "Delete a user.", SCHEMA,
                                           guarded_delete, destructive=True)
    return server


def send(ref, target, params, request_id):
    return target.handle({"jsonrpc": "2.0", "id": request_id, "method": "tools/call",
                          "params": {**params, "_meta": ref.request_metadata()}})


def flow(ref, server):
    gate = Elicit(server)
    asked = send(ref, gate, {"name": "delete_user", "arguments": {"user_id": 7}}, 1)["result"]
    answer = {"name": "delete_user", "arguments": {"user_id": 7},
              "inputResponses": [{"name": "confirmed", "value": True}],
              "requestState": asked["requestState"]}
    asked_deleted = list(DELETED)
    retry = send(ref, gate, answer, 2)["result"]
    retry_deleted = list(DELETED)
    replayed = send(ref, gate, answer, 2)["result"]
    return {"asked_type": asked["resultType"], "requests": asked["inputRequests"],
            "state": asked["requestState"], "retry_type": retry["resultType"],
            "retry_text": retry["content"][0]["text"], "asked_deleted": asked_deleted,
            "retry_deleted": retry_deleted, "replay_type": replayed["resultType"],
            "replay_deleted": list(DELETED)}


METHODS = (("server/discover", {}), ("tools/list", {}), ("resources/list", {}),
           ("prompts/list", {}), ("resources/read", {"uri": "config://app"}),
           ("tools/call", {"name": "add", "arguments": {"a": 1, "b": 2}}))


def unguarded(ref, server):
    results = [server.handle({"jsonrpc": "2.0", "id": 4, "method": method,
                              "params": {**params, "_meta": ref.request_metadata()}})["result"]
               for method, params in METHODS]
    extra = send(ref, server, {"name": "add", "arguments": {"a": 1, "b": 2},
                               "inputResponses": [{"name": "x", "value": 1}],
                               "requestState": {"step": 9}}, 5)["result"]
    hints = {t["name"]: t.get("annotations", {}) for t in results[1]["tools"]}
    return {"result_types": sorted({r["resultType"] for r in results}),
            "is_error": results[-1]["isError"], "extra_ok": extra["resultType"],
            "hint": hints["delete_user"],
            "client_api": [n for n in dir(ref.MCPClient) if not n.startswith("_")]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    server = build(ref)
    steps = flow(ref, server)
    plain = send(ref, server,
                 {"name": "delete_user", "arguments": {"user_id": 99, "confirmed": True}}, 7)
    return {**steps, **unguarded(ref, server), "plain_delete": plain["result"]["resultType"]}


def verify(result):
    return [
        practice.Check(
            "ANSWER: the ask deletes nothing and the retry deletes once",
            all([result["asked_type"] == "input_required", result["retry_type"] == "complete",
                 result["asked_deleted"] == [], result["retry_deleted"] == [7]]),
            f"the ask returns {result['asked_type']!r} with {result['requests']} and "
            f"requestState {result['state']}, deleting {result['asked_deleted']}; the retry "
            f"with a fresh id and the same state returns {result['retry_text']}, deleting "
            f"{result['retry_deleted']}",
        ),
        practice.Check(
            "FINDING: input_required cannot come from a tool handler",
            all([result["result_types"] == ["complete"], not result["is_error"]]),
            "_complete writes resultType into every result and all six methods answer "
            f"{result['result_types']}; tools/call hardcodes isError {result['is_error']} "
            "and JSON-encodes the return into content[0]. Elicitation changes the builder",
        ),
        practice.Check(
            "FINDING: the server never validates the extra params",
            result["extra_ok"] == "complete",
            "sent to the unmodified server, a tools/call carrying inputResponses and "
            f"requestState returns an ordinary {result['extra_ok']!r} result: handle reads "
            "name, arguments and _meta and ignores the rest. The contract is convention",
        ),
        practice.Check(
            "FINDING: 'a new ID' is a convention the server cannot enforce",
            all([result["replay_type"] == "complete", result["client_api"] == ["request"],
                 result["replay_deleted"] == [7, 7]]),
            f"replaying the retry with the original id deletes again -- "
            f"{result['replay_deleted']} -- because the server keeps no per-request state. "
            f"MCPClient exposes {result['client_api']} and handle returns one response per "
            "message: no channel exists for a reverse request",
        ),
        practice.Check(
            "FINDING: the destructive annotation is decoration",
            all([result["hint"] == {"destructiveHint": True},
                 result["plain_delete"] == "complete"]),
            f"tools/list publishes {result['hint']} for delete_user and the tools/call "
            "branch never reads tool.destructive, so a delete straight to the server "
            f"returns {result['plain_delete']!r}. The gate has to live in the wrapper",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
