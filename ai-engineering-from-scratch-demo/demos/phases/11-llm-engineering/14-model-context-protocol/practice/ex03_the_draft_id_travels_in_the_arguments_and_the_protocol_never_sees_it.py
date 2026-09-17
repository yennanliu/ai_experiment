"""Exercise 3 — the draftId travels in the arguments and the protocol never sees it.

    Add a server-minted `draftId` to a create operation, then require it as an
    argument to update. Explain why that is application state rather than a
    protocol session.

Reading of the exercise: the two tools are built and the round trip walked, and
then the "why" is answered by measurement rather than assertion -- by looking
for the places a protocol session would have to live and finding them fixed or
absent. Two independent clients are used throughout, because "session" is a
claim about who the state belongs to.

**ANSWER: `create_draft` mints `draft-0001` and `update_draft` requires it.**
The id appears in exactly one place on the wire: `params.arguments.draftId`. It
is minted by the handler, interpreted by the handler, and the protocol layer --
`handle`, `_validate_metadata`, `_complete` -- never reads it.

**FINDING: the id is unscoped, which is what makes it application state.**
Client A mints `draft-0001`; client B, a separate `MCPClient` with its own id
counter, updates it and is accepted. Both clients send request id 1 first,
because each counts from zero, so the server could not tell them apart even if
it wanted to. The draft belongs to the server's dict, not to a caller.

**FINDING: there is nowhere to put a session.** A request's `_meta` carries
exactly three keys -- `protocolVersion`, `clientCapabilities`, `clientInfo` --
and a response's carries exactly one, `serverInfo`. `MCPServer.__init__` stores
`name`, `server_info`, `tools`, `resources`, `prompts` and no per-connection
state, and `MCPClient` exposes one method, `request`, which rebuilds
`request_metadata()` from scratch on every call. A session would need a new
`_meta` key, a store to hold it and a check in `_validate_metadata`; none of
the three exists.

**FINDING: `resources/read` advertises mutable state as cacheable.**
`_complete(cacheable=True)` is applied to `resources/read` unconditionally, so
a draft exposed as a resource comes back with `ttlMs: 30000` and
`cacheScope: "private"` -- a client obeying that advice shows a 30-second-stale
draft. `tools/call` is the one method that is not cacheable, so the mutation is
fresh and the read of it is not.

**FINDING: the required argument is enforced by Python, not by the protocol.**
Omitting `draftId` raises `TypeError` and passing an unknown one raises
`KeyError`, and both arrive as **-32603 "tool handler failed"** -- the same
code, with no data. A client cannot tell "you forgot the id" from "that draft
is gone", and neither is -32602.

Structure: `DRAFTS` is the application state, `create_draft` and `update_draft`
the two tools, `build` registers them on a fresh server, `round_trip` walks
create-then-update across two clients, and `places` looks for somewhere a
session could live.
"""

from __future__ import annotations

import json

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "14-model-context-protocol"
DRAFTS: dict[str, dict] = {}
OBJECT_SCHEMA = {"type": "object", "properties": {}, "required": []}


def create_draft(title):
    draft_id = f"draft-{len(DRAFTS) + 1:04d}"
    DRAFTS[draft_id] = {"title": title, "body": ""}
    return {"draftId": draft_id}


def update_draft(draftId, body):  # noqa: N803 - the wire name is the argument name
    DRAFTS[draftId]["body"] = body
    return {"draftId": draftId, "length": len(body)}


def build(ref):
    DRAFTS.clear()
    server = ref.MCPServer("drafts")
    server.tools["create_draft"] = ref.Tool("create_draft", "Create a draft, minting its id.",
                                            OBJECT_SCHEMA, create_draft)
    server.tools["update_draft"] = ref.Tool("update_draft", "Update a draft by id.",
                                            OBJECT_SCHEMA, update_draft)
    server.resources["draft://latest"] = ref.Resource(
        "draft://latest", "latest", "The newest draft.",
        lambda: json.dumps(DRAFTS.get("draft-0001", {})))
    return server


def tool_call(server, name, arguments, ref, request_id=1):
    return server.handle({"jsonrpc": "2.0", "id": request_id, "method": "tools/call",
                          "params": {"name": name, "arguments": arguments,
                                     "_meta": ref.request_metadata()}})


def round_trip(ref, server):
    """Client A mints the id; client B, with its own counter, spends it."""
    author, editor = ref.MCPClient(server), ref.MCPClient(server)
    made = author.request("tools/call", {"name": "create_draft", "arguments": {"title": "notes"}})
    draft_id = json.loads(made["content"][0]["text"])["draftId"]
    edited = editor.request("tools/call", {"name": "update_draft",
                                           "arguments": {"draftId": draft_id, "body": "hello"}})
    return {"draft_id": draft_id, "cross_client": json.loads(edited["content"][0]["text"]),
            "first_ids": [ref.MCPClient(server)._id + 1, ref.MCPClient(server)._id + 1],
            "call_result_keys": sorted(made), "call_cacheable": "ttlMs" in made}


def places(ref, server):
    """Everywhere a protocol session could live, and what is actually there."""
    read = server.handle({"jsonrpc": "2.0", "id": 9, "method": "resources/read",
                          "params": {"uri": "draft://latest",
                                     "_meta": ref.request_metadata()}})["result"]
    missing = tool_call(server, "update_draft", {"body": "x"}, ref)["error"]
    unknown = tool_call(server, "update_draft", {"draftId": "draft-9999", "body": "x"}, ref)["error"]
    return {"request_meta": sorted(ref.request_metadata()),
            "response_meta": sorted(read["_meta"]), "server_state": sorted(vars(server)),
            "client_api": [n for n in dir(ref.MCPClient) if not n.startswith("_")],
            "read_ttl": read.get("ttlMs"), "read_scope": read.get("cacheScope"),
            "missing_code": missing["code"], "unknown_code": unknown["code"],
            "missing_message": missing["message"], "error_data": "data" in missing}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    server = build(ref)
    trip = round_trip(ref, server)
    return {"drafts": len(DRAFTS), **trip, **places(ref, server)}


def verify(result):
    return [
        practice.Check(
            "ANSWER: create_draft mints draft-0001 and update_draft requires it",
            all([result["draft_id"] == "draft-0001", result["drafts"] == 1,
                 result["cross_client"] == {"draftId": "draft-0001", "length": 5}]),
            f"the create call returns {result['draft_id']!r} and the update call spends it, "
            f"returning {result['cross_client']}. The id is on the wire in exactly one "
            "place, params.arguments.draftId, minted and interpreted by the handler",
        ),
        practice.Check(
            "FINDING: the id is unscoped, which is what makes it application state",
            all([result["first_ids"] == [1, 1], result["cross_client"]["length"] == 5]),
            "a separate MCPClient with its own counter updates the first client's draft and "
            f"is accepted, and both clients send request id {result['first_ids'][0]} first "
            "because each counts from zero. The draft belongs to the server's dict, not to a "
            "caller, and the server could not tell them apart if it wanted to",
        ),
        practice.Check(
            "FINDING: there is nowhere to put a session",
            all([len(result["request_meta"]) == 3, result["response_meta"] ==
                 ["io.modelcontextprotocol/serverInfo"], result["client_api"] == ["request"],
                 result["server_state"] == ["name", "prompts", "resources", "server_info",
                                            "tools"]]),
            f"a request's _meta carries {len(result['request_meta'])} keys and a response's "
            f"{result['response_meta']}; MCPServer holds {result['server_state']} and no "
            f"per-connection state; MCPClient exposes {result['client_api']} and rebuilds "
            "request_metadata() every call. A session needs a new _meta key, a store and a "
            "check in _validate_metadata -- none of the three exists",
        ),
        practice.Check(
            "FINDING: resources/read advertises mutable state as cacheable",
            all([result["read_ttl"] == 30000, result["read_scope"] == "private",
                 not result["call_cacheable"]]),
            f"_complete(cacheable=True) is applied to resources/read unconditionally, so a "
            f"draft read back carries ttlMs {result['read_ttl']} and cacheScope "
            f"{result['read_scope']!r}. tools/call is the one method that is not cacheable "
            f"({sorted(result['call_result_keys'])}), so the mutation is fresh and the read "
            "of it is not",
        ),
        practice.Check(
            "FINDING: the required argument is enforced by Python, not by the protocol",
            all([result["missing_code"] == -32603, result["unknown_code"] == -32603,
                 not result["error_data"]]),
            f"omitting draftId raises TypeError and an unknown one raises KeyError, and both "
            f"arrive as {result['missing_code']} {result['missing_message']!r} with no data. "
            "A client cannot tell 'you forgot the id' from 'that draft is gone', and neither "
            "is -32602",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
