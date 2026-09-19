"""Exercise 5 — the two eras are separated by a key neither one shares.

    Build a separate legacy adapter for `2025-11-25`. Add tests proving a
    modern request never enters it.

Reading of the exercise: the adapter is built with its own connection state --
which is the whole difference between the eras -- and "never enters it" is
tested as a property of the selector rather than of the adapter. The lesson's
own text says the selection decision happens *before* either parser runs, so the
proof is that the selector is total and its two branches are disjoint, not that
the adapter happens to reject modern traffic.

**ANSWER: a selector on `params._meta`, an adapter holding connection state, and
**8** cases proving the split.** A message carrying
`io.modelcontextprotocol/protocolVersion` routes modern; one carrying
`initialize` or no `_meta` routes legacy. No message routes both, and none
routes neither.

**FINDING: the adapter needs state the modern server has nowhere to put.**
`initialize` stores negotiated capabilities on the connection, and a later
`tools/list` on that connection succeeds *without* re-sending them -- which is
exactly what Exercise 1 proved the modern path cannot do. Uninitialised, the
same `tools/list` is refused. The eras differ by one dict that outlives a
request.

**FINDING: the modern server already rejects the legacy entry point, and that
is not the same as routing.** `initialize` through the lesson's own `dispatch`
gives **-32601 Method not found** -- a modern rejection of a legacy method, not
a handover. Without a selector the two eras meet inside one dispatcher, and a
`-32601` is indistinguishable from a genuinely unknown method.

**FINDING: and the legacy adapter accepts a version the modern server refuses.**
`2025-11-25` is not in `SUPPORTED_VERSIONS`, so the modern path answers
**-32022**; the adapter answers it. Each era is total over its own inputs and
empty over the other's, which is what "a stateless modern core beside an
isolated legacy adapter" means once it is code.

Structure: `LegacyAdapter` holds the connection state, `select` is the
pre-parser decision, and `ROUTING_CASES` is the table the split is proved over.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "07-building-an-mcp-server"
LEGACY_VERSION = "2025-11-25"
MODERN, LEGACY = "modern", "legacy"


class LegacyAdapter:
    """A 2025-11-25 connection: initialize negotiates, and the state persists."""

    def __init__(self):
        self.capabilities = None

    def handle(self, message):
        method = message.get("method")
        if method == "initialize":
            self.capabilities = message.get("params", {}).get("capabilities", {})
            return {"jsonrpc": "2.0", "id": message.get("id"),
                    "result": {"protocolVersion": LEGACY_VERSION,
                               "capabilities": {"tools": {}}}}
        if self.capabilities is None:
            return {"jsonrpc": "2.0", "id": message.get("id"),
                    "error": {"code": -32002, "message": "Not initialized"}}
        return {"jsonrpc": "2.0", "id": message.get("id"),
                "result": {"tools": [], "negotiated": sorted(self.capabilities)}}


def select(ref, message):
    """The pre-parser decision: which era owns this message."""
    meta = message.get("params", {}).get("_meta")
    if isinstance(meta, dict) and ref.VERSION_KEY in meta:
        return MODERN
    return LEGACY


def legacy_message(method, request_id=1, capabilities=None):
    params = {} if capabilities is None else {"capabilities": capabilities}
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}


def routing_cases(ref):
    """Every message shape either era could receive."""
    modern = ref.make_request(1, "tools/list")
    no_meta = dict(modern)
    no_meta["params"] = {k: v for k, v in modern["params"].items() if k != "_meta"}
    stale = ref.make_request(2, "tools/list", version=LEGACY_VERSION)
    return {
        "modern tools/list": modern,
        "modern server/discover": ref.make_request(3, "server/discover"),
        "modern resources/list": ref.make_request(4, "resources/list"),
        "modern with legacy version": stale,
        "legacy initialize": legacy_message("initialize", capabilities={"tools": {}}),
        "legacy tools/list": legacy_message("tools/list"),
        "legacy no params": {"jsonrpc": "2.0", "id": 5, "method": "tools/list"},
        "modern shape without _meta": no_meta,
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.reset_notes()
    cases = routing_cases(ref)
    routed = {label: select(ref, message) for label, message in cases.items()}

    fresh = LegacyAdapter()
    uninitialised = fresh.handle(legacy_message("tools/list"))
    fresh.handle(legacy_message("initialize", capabilities={"tools": {}, "roots": {}}))
    initialised = fresh.handle(legacy_message("tools/list"))

    modern_initialize = ref.dispatch(ref.make_request(6, "initialize"))
    modern_legacy_version = ref.dispatch(
        ref.make_request(7, "tools/list", version=LEGACY_VERSION))
    return {
        "routed": routed, "cases": len(cases),
        "modern_cases": sum(era == MODERN for era in routed.values()),
        "legacy_cases": sum(era == LEGACY for era in routed.values()),
        "eras": sorted(set(routed.values())),
        "uninitialised": uninitialised["error"]["code"],
        "initialised": initialised["result"]["negotiated"],
        "adapter_holds_state": fresh.capabilities is not None,
        "modern_initialize": modern_initialize["error"]["code"],
        "modern_initialize_message": modern_initialize["error"]["message"],
        "modern_legacy_version": modern_legacy_version["error"]["code"],
        "supported": list(ref.SUPPORTED_VERSIONS),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a selector, an adapter with connection state, and 8 cases",
            all([result["cases"] == 8, result["modern_cases"] == 4,
                 result["legacy_cases"] == 4, result["eras"] == [LEGACY, MODERN],
                 result["routed"]["modern tools/list"] == MODERN,
                 result["routed"]["legacy initialize"] == LEGACY]),
            f"{result['cases']} message shapes route {result['modern_cases']} modern and "
            f"{result['legacy_cases']} legacy: {result['routed']}. A message carrying the "
            "protocolVersion key routes modern; one carrying initialize or no _meta routes "
            "legacy. None routes both and none routes neither",
        ),
        practice.Check(
            "FINDING: the adapter needs state the modern server has nowhere to put",
            all([result["uninitialised"] == -32002,
                 result["initialised"] == ["roots", "tools"],
                 result["adapter_holds_state"]]),
            f"uninitialised, the adapter refuses tools/list with {result['uninitialised']}. "
            f"After initialize it answers with {result['initialised']} -- capabilities the "
            "later request never re-sent, which is exactly what Exercise 1 proved the modern "
            "path cannot do. The eras differ by one dict that outlives a request",
        ),
        practice.Check(
            "FINDING: rejecting the legacy entry point is not the same as routing",
            all([result["modern_initialize"] == -32601,
                 result["modern_initialize_message"] == "Method not found: initialize"]),
            f"initialize through the lesson's own dispatch gives "
            f"{result['modern_initialize']} {result['modern_initialize_message']!r} -- a "
            "modern rejection of a legacy method, not a handover. Without a selector the two "
            "eras meet inside one dispatcher, and that code is indistinguishable from a "
            "genuinely unknown method",
        ),
        practice.Check(
            "FINDING: the legacy adapter accepts a version the modern server refuses",
            all([result["modern_legacy_version"] == -32022,
                 result["supported"] == ["2026-07-28"],
                 result["routed"]["modern with legacy version"] == MODERN]),
            f"{LEGACY_VERSION} is not in {result['supported']}, so the modern path answers "
            f"{result['modern_legacy_version']}. Note the selector still routes that message "
            "modern -- it carries the modern metadata key -- so a stale version is a modern "
            "error rather than a legacy handover. Each era is total over its own inputs and "
            "empty over the other's",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
