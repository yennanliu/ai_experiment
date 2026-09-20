"""Exercise 6 — the gateway keeps the session, so the server does not have to.

    Model an older server behind the gateway. Put all handshake and session
    behavior behind an explicit `2025-11-25` compatibility branch.

Reading of the exercise: "all" and "explicit" are both testable, so isolation
is defined as two measurements -- the modern path exchanges no legacy message,
and the legacy state lives on the adapter rather than on the gateway. The
branch is then made explicit by a selector that reads the backend's declared
era rather than inferring one from a failure, because inference is how a
compatibility branch becomes reachable by accident.

**ANSWER: a `2025-11-25` adapter that handshakes once and holds the session.**
The legacy backend refuses `tools/list` with `-32002` until `initialize`
succeeds; the adapter performs the handshake, keeps the session id, and
answers **3** tools through the gateway's own shape. The modern gateway sees
**0** legacy messages.

**FINDING: the session is one field, and it is on the adapter.**
`SecurityGateway` carries `approved`, `catalog`, `replay_store` and `secret`
-- **4** fields, none per-connection -- while the adapter holds a session id
that outlives a request. Re-adding the era re-adds exactly the state the
modern design removed, in the one component allowed to have it.

**FINDING: the handshake costs a message that the modern path does not
have.** Legacy needs `initialize` then `tools/list` -- **2** requests for one
answer -- where modern needs **1**. The extra round trip is the session being
established, which is the thing the modern revision deleted.

**FINDING: the branch is chosen by declaration, not by failure.** Routing on
the backend's stated era sends **0** modern requests to the legacy adapter.
Routing on "the modern call failed, try legacy" would send a legacy handshake
after any transient error -- the selector is explicit precisely so a fallback
cannot be entered by accident.

Structure: `LegacyBackend` is the older server, `LegacyAdapter` is the
compatibility branch that owns its session, and `route` is the explicit
selector.
"""

from __future__ import annotations

import secrets

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "15-mcp-security-tool-poisoning"
LEGACY = "2025-11-25"


class LegacyBackend:
    """A 2025-11-25 server: nothing works until initialize does."""

    def __init__(self, tools):
        self.tools, self.sessions, self.received = tools, set(), []

    def request(self, method, params=None, session=None):
        self.received.append(method)
        if method == "initialize":
            issued = secrets.token_hex(8)
            self.sessions.add(issued)
            return {"protocolVersion": LEGACY, "sessionId": issued}
        if session not in self.sessions:
            return {"error": {"code": -32002, "message": "Server not initialized"}}
        if method == "tools/list":
            return {"tools": [dict(tool) for tool in self.tools]}
        return {"error": {"code": -32601, "message": "Method not found"}}


class LegacyAdapter:
    """The compatibility branch: it owns the handshake and the session."""

    era = LEGACY

    def __init__(self, backend):
        self.backend, self.session = backend, None

    def connect(self):
        self.session = self.backend.request("initialize")["sessionId"]
        return self.session

    def tools_list(self):
        if self.session is None:
            self.connect()
        answer = self.backend.request("tools/list", session=self.session)
        return [f"legacy.{tool['name']}" for tool in answer["tools"]]


def route(ref, gateway, backend):
    """Explicit selection on the declared era -- never on a failure."""
    if getattr(backend, "era", None) == LEGACY:
        return "legacy", backend.tools_list()
    body, headers = ref.make_request("tools/list", 1)
    result = gateway.handle(body, headers)[1]["result"]
    return "modern", [tool["name"] for tool in result["tools"]]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    gateway = ref.SecurityGateway()
    backend = LegacyBackend([{"name": "search"}, {"name": "export"}, {"name": "archive"}])
    adapter = LegacyAdapter(backend)

    cold = backend.request("tools/list", session="nope")
    era, legacy_tools = route(ref, gateway, adapter)
    before_modern = len(backend.received)
    modern_era, modern_tools = route(ref, gateway, None)
    reached_legacy = len(backend.received) - before_modern

    fresh = LegacyBackend([{"name": "search"}])
    fresh_adapter = LegacyAdapter(fresh)
    fresh_adapter.tools_list()
    return {
        "cold": cold["error"]["code"],
        "era": era, "legacy_tools": legacy_tools,
        "modern_era": modern_era, "modern_tools": modern_tools,
        "legacy_traffic": backend.received,
        "handshake_messages": fresh.received,
        "gateway_fields": sorted(vars(gateway)),
        "adapter_fields": sorted(vars(adapter)),
        "session_held": adapter.session is not None,
        "legacy_in_modern": any(m in ("initialize",) for m in modern_tools),
        "routed_modern_to_legacy": reached_legacy,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the adapter handshakes once, holds the session, and answers three tools",
            all([result["cold"] == -32002, result["era"] == "legacy",
                 result["legacy_tools"] == ["legacy.search", "legacy.export",
                                            "legacy.archive"],
                 result["session_held"],
                 result["legacy_traffic"] == ["tools/list", "initialize", "tools/list"]]),
            f"the backend refuses a cold tools/list with {result['cold']} until initialize "
            f"succeeds; the adapter handshakes, keeps the session and returns "
            f"{result['legacy_tools']}. Its traffic is {result['legacy_traffic']}",
        ),
        practice.Check(
            "FINDING: the session is one field, and it is on the adapter",
            all([result["gateway_fields"] == ["approved", "catalog", "replay_store",
                                              "secret"],
                 result["adapter_fields"] == ["backend", "session"],
                 "session" not in result["gateway_fields"]]),
            f"SecurityGateway carries {result['gateway_fields']} -- "
            f"{len(result['gateway_fields'])} fields, none per-connection -- while the "
            f"adapter carries {result['adapter_fields']}. Re-adding the era re-adds exactly "
            "the state the modern design removed, in the one component allowed to have it",
        ),
        practice.Check(
            "FINDING: the handshake costs a message the modern path does not have",
            all([result["handshake_messages"] == ["initialize", "tools/list"],
                 len(result["handshake_messages"]) == 2]),
            f"a fresh legacy backend sees {result['handshake_messages']} -- "
            f"{len(result['handshake_messages'])} requests for one answer -- where the "
            "modern gateway answers in one. The extra round trip is the session being "
            "established, which is the thing the modern revision deleted",
        ),
        practice.Check(
            "FINDING: the branch is chosen by declaration, not by failure",
            all([result["modern_era"] == "modern",
                 result["modern_tools"] == ["issues.search", "notes.export", "notes.search"],
                 not result["legacy_in_modern"],
                 result["routed_modern_to_legacy"] == 0]),
            f"routing on the backend's stated era sends the modern call to the modern path, "
            f"{result['modern_tools']}, and {result['routed_modern_to_legacy']} modern "
            "requests to the adapter. Routing on 'the modern call failed, try legacy' would "
            "start a legacy handshake after any transient error -- the selector is explicit "
            "so a fallback cannot be entered by accident",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
