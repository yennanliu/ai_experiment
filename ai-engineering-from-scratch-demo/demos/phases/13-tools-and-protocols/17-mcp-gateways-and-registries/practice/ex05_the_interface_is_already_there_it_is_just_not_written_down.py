"""Exercise 5 — the interface is already there; it is just not written down.

    Write a legacy adapter interface without adding any legacy state to the
    modern `Gateway` class.

Reading of the exercise: "without adding state" is the constraint to measure,
so the gateway's field set is captured before and after and compared. Writing
the interface then turns out to be less work than it sounds, because
`Gateway._forward` already calls exactly one method on whatever sits in
`self.backends` -- so the interface exists as a de facto contract and the
adapter's job is to satisfy it rather than to introduce a new one.

**ANSWER: a `Backend` protocol of one method, and an adapter that satisfies
it.** `handle(body, headers) -> dict` is the whole surface `_forward` uses;
`BackendServer` matches it, `LegacyAdapter` matches it, and the gateway routes
to either with **0** changes. The adapter answers a modern `tools/call` by
running the legacy handshake, keeping the session, and translating the reply.

**FINDING: the gateway's fields are unchanged, and the session is the
adapter's.** Before and after, `Gateway` carries `audit`, `backends`,
`buckets`, `forwarded_request_ids`, `pins` -- **5**, none per-connection --
while the adapter carries a session id that outlives a request. The constraint
holds because the adapter is a backend, not a branch.

**FINDING: the real interface is wider than the protocol.** Registering the
adapter as a backend and pinning its descriptor is not enough: `tools/list`
answers **400** until `REGISTRY_SERVER_JSON` and `VERIFIED_ADMISSION_STATE`
each carry a row under the same key, because `_visible_tools` indexes both by
backend name and a missing key raises straight into the handler's
`except KeyError`. The structural interface is a method and an attribute; the
working one is those plus two module dictionaries nothing declares.

**FINDING: the handshake cost is invisible from the gateway's side.** The
first forwarded call makes the adapter send **2** legacy messages and later
calls **1**, while the gateway records **1** forwarded id either way. A
per-request latency budget measured at the gateway would not see the
handshake -- which is the argument for the adapter owning it rather than the
router.

Structure: `LegacyBackend` is the old server, `LegacyAdapter` is the
compatibility branch that satisfies the protocol, and `routes` checks the two
backend kinds against the same structural check.
"""

from __future__ import annotations

import secrets
from typing import Any, Protocol, runtime_checkable

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "17-mcp-gateways-and-registries"


@runtime_checkable
class Backend(Protocol):
    """Everything Gateway._forward uses, written down."""

    tools: list[dict[str, Any]]

    def handle(self, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]: ...


class LegacyBackend:
    """A 2025-11-25 server: nothing answers until initialize does."""

    def __init__(self):
        self.sessions, self.received = set(), []

    def request(self, method, session=None, name=None):
        self.received.append(method)
        if method == "initialize":
            self.sessions.add(issued := secrets.token_hex(4))
            return {"sessionId": issued}
        if session not in self.sessions:
            return {"error": {"code": -32002, "message": "Server not initialized"}}
        return {"text": f"legacy {name} done"}


class LegacyAdapter:
    """A Backend that hides a handshake and a session behind one modern method."""

    def __init__(self, backend, tools):
        self.backend, self.tools, self.session = backend, tools, None

    def handle(self, body, headers):
        if self.session is None:
            self.session = self.backend.request("initialize")["sessionId"]
        answer = self.backend.request("tools/call", session=self.session,
                                      name=body["params"]["name"])
        return {"jsonrpc": "2.0", "id": body.get("id"), "result": {
            "resultType": "complete", "isError": False, "_meta": {},
            "content": [{"type": "text", "text": answer["text"]}]}}


def enrol(ref, key):
    """The rows `_visible_tools` needs beyond the backend object itself."""
    name = f"com.example/{key}"
    ref.REGISTRY_SERVER_JSON[key] = {**ref.REGISTRY_SERVER_JSON["notes"], "name": name}
    ref.VERIFIED_ADMISSION_STATE[key] = {
        **ref.VERIFIED_ADMISSION_STATE["notes"], "registryName": name}


def descriptor(name):
    return {"name": name, "description": f"Legacy {name}.",
            "inputSchema": {"type": "object"}}


def archive(ref, gateway, index):
    body, headers = ref.make_request("tools/call", index, {"name": "legacy.archive"})
    status, response = gateway.handle("bearer-alice", body, headers)
    return status, response["result"]["content"][0]["text"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    before, gateway = sorted(vars(ref.Gateway())), ref.Gateway()
    legacy = LegacyBackend()
    adapter = LegacyAdapter(legacy, [descriptor("archive")])
    gateway.backends["legacy"] = adapter
    gateway.pins["legacy.archive"] = ref.descriptor_digest(adapter.tools[0])
    unregistered = gateway.handle("bearer-alice", *ref.make_request("tools/list", 8))[0]
    enrol(ref, "legacy")
    ref.RBAC["alice"] = set(ref.RBAC["alice"]) | {"legacy.archive"}
    try:
        results = [archive(ref, gateway, index) for index in range(2)]
        listed = [t["name"] for t in gateway.handle("bearer-alice", *ref.make_request(
            "tools/list", 9))[1]["result"]["tools"]]
    finally:  # the reference module is shared, so every edit is undone
        ref.RBAC["alice"] = set(ref.RBAC["alice"]) - {"legacy.archive"}
        for table in (ref.REGISTRY_SERVER_JSON, ref.VERIFIED_ADMISSION_STATE):
            table.pop("legacy", None)
    return {
        "before": before, "after": sorted(vars(gateway)), "results": results,
        "adapter_fields": sorted(vars(adapter)), "listed": listed,
        "modern_is_backend": isinstance(ref.BackendServer("n", []), Backend),
        "adapter_is_backend": isinstance(adapter, Backend),
        "members": sorted(Backend.__protocol_attrs__),
        "first_hop": legacy.received[:2], "later_hop": legacy.received[2:],
        "forwarded": len(gateway.forwarded_request_ids), "unregistered": unregistered,
        "session_held": adapter.session is not None,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a Backend protocol of one method and one attribute, satisfied by both",
            all([result["modern_is_backend"], result["adapter_is_backend"],
                 result["members"] == ["handle", "tools"],
                 result["results"] == [(200, "legacy archive done")] * 2,
                 "legacy.archive" in result["listed"]]),
            f"the protocol's members are {result['members']}, satisfied by BackendServer and "
            f"LegacyAdapter alike, so the gateway routes to either unchanged. Two calls answer "
            f"{result['results'][0]}, joining {len(result['listed']) - 1} modern tools",
        ),
        practice.Check(
            "FINDING: the gateway's fields are unchanged, and the session is the adapter's",
            all([result["before"] == result["after"],
                 result["before"] == ["audit", "backends", "buckets",
                                      "forwarded_request_ids", "pins"],
                 result["adapter_fields"] == ["backend", "session", "tools"],
                 result["session_held"]]),
            f"Gateway carries {result['before']} before and after, none per-connection, while "
            f"the adapter carries {result['adapter_fields']} including a session that outlives "
            "a request -- the constraint holds because the adapter is a backend, not a branch",
        ),
        practice.Check(
            "FINDING: the real interface is wider than the protocol -- two registry rows too",
            all([result["members"] == ["handle", "tools"],
                 result["unregistered"] == 400, "legacy.archive" in result["listed"]]),
            f"registering the adapter and pinning its descriptor is not enough: tools/list "
            f"answers HTTP {result['unregistered']} until REGISTRY_SERVER_JSON and "
            "VERIFIED_ADMISSION_STATE carry a row under the same key. The structural "
            "interface is a method and an attribute; the working one adds two dictionaries",
        ),
        practice.Check(
            "FINDING: the handshake cost is invisible from the gateway's side",
            all([result["first_hop"] == ["initialize", "tools/call"],
                 result["later_hop"] == ["tools/call"],
                 result["forwarded"] == 2]),
            f"the first forwarded call sends {result['first_hop']} and the second "
            f"{result['later_hop']}, while the gateway records {result['forwarded']} ids either "
            "way. A latency budget at the gateway would not see the handshake, which is the "
            "argument for the adapter owning it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
