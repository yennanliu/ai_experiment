"""Exercise 2 — discovery is gated on the version it exists to report.

    Remove the protocol-version key and verify Invalid Params (`-32602`). Then
    send the well-formed but unsupported version `2025-11-25`, verify `-32022`,
    confirm `requested` echoes that revision, and choose from `supported`.

Reading of the exercise: "remove the key" has three distinct meanings -- no
`_meta` at all, `_meta` without the key, and the key holding a non-string --
so all three are sent. The unsupported-version path is then walked to its end:
read `supported`, pick from it, retry, and check that the retry is accepted.
The last step is the one that shows what the error is for.

**ANSWER: all three malformed requests return -32602, with three different
messages.** "params._meta is required", then
"io.modelcontextprotocol/protocolVersion is required", then
"io.modelcontextprotocol/protocolVersion must be a string". The server names
the key it wanted in the message, which is the only place it appears.

**ANSWER: `2025-11-25` returns -32022 and the error body is complete.**
`data` is `{"supported": ["2026-07-28"], "requested": "2025-11-25"}` --
`requested` echoes the revision that was sent, `supported` lists what to use
instead, and retrying with `supported[0]` is accepted.

**FINDING: the version-free methods are not exempt.** `server/discover` exists
to tell a client which versions the server speaks, and it is refused with the
same -32022 unless the client already knows one. So the discovery path for a
client with no prior knowledge is to send a version it knows is wrong and read
`supported` out of the error -- the -32022 body *is* the discovery mechanism,
and the well-formed `server/discover` result is the confirmation afterwards.

**FINDING: -32022 is the only self-describing error.** None of the three -32602
responses carries a `data` field, so a client that omitted the key learns the
key's name only by parsing the English message. The error that tells you the
answer is the one you get for guessing, not the one you get for asking.

**FINDING: `supported` has one entry, so "choose from supported" is not a
choice.** `SUPPORTED_VERSIONS = (PROTOCOL_VERSION,)`. A client holding a list
of revisions cannot negotiate a common one; it can only match the single value
or fail. The protocol has room for negotiation and the server has none.

**FINDING: validation is skipped entirely for notifications.** `handle` returns
`None` before `_validate_metadata` when `"id"` is absent, so a notification
carrying the version `1999-01-01` is accepted silently and does nothing. Every
guarantee this exercise checks applies only to messages with an id.

Structure: `send` posts one raw JSON-RPC message, `MALFORMED` builds the three
ways the version key can be wrong, `negotiate` walks the unsupported-version
path to a successful retry, and `gated` asks the same of `server/discover`.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "14-model-context-protocol"
OLD_VERSION = "2025-11-25"
INVALID_PARAMS, UNSUPPORTED_VERSION = -32602, -32022


def send(server, method, params, request_id=1):
    return server.handle({"jsonrpc": "2.0", "id": request_id, "method": method,
                          "params": params})


def meta(ref, version=None):
    return {"_meta": ref.request_metadata(protocol_version=version or ref.PROTOCOL_VERSION)}


def malformed(ref, server):
    """The three ways "remove the protocol-version key" can be read."""
    cases = {"no_meta": {},
             "no_key": {"_meta": {ref.CLIENT_CAPABILITIES_KEY: {}}},
             "not_a_string": {"_meta": {ref.PROTOCOL_KEY: 20260728,
                                        ref.CLIENT_CAPABILITIES_KEY: {}}}}
    errors = {name: send(server, "tools/list", params)["error"] for name, params in cases.items()}
    return {"codes": sorted({e["code"] for e in errors.values()}),
            "messages": [errors[name]["message"] for name in cases],
            "with_data": [name for name, e in errors.items() if "data" in e]}


def negotiate(ref, server):
    """Send the unsupported revision, read `supported`, pick from it, retry."""
    refused = send(server, "tools/list", meta(ref, OLD_VERSION))["error"]
    offered = refused["data"]["supported"]
    retried = send(server, "tools/list", meta(ref, offered[0]), request_id=2)
    both_bad = send(server, "tools/list", {"_meta": {ref.PROTOCOL_KEY: OLD_VERSION,
                                                     ref.CLIENT_CAPABILITIES_KEY: "nope"}})
    return {"refused_code": refused["code"], "requested": refused["data"]["requested"],
            "supported": offered, "retry_ok": "result" in retried,
            "retry_tools": len(retried["result"]["tools"]), "both_bad": both_bad["error"]["code"]}


def gated(ref, server):
    """Whether the method that reports the versions needs one."""
    refused = send(server, "server/discover", meta(ref, OLD_VERSION))["error"]
    allowed = send(server, "server/discover", meta(ref))["result"]
    notification = server.handle({"jsonrpc": "2.0", "method": "tools/list",
                                  "params": {"_meta": {ref.PROTOCOL_KEY: "1999-01-01"}}})
    return {"discover_code": refused["code"],
            "discover_supported": refused["data"]["supported"],
            "discover_result": allowed["supportedVersions"],
            "notification": notification}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    server = ref.MCPServer("practice")
    server.tools.update(ref.server.tools)
    return {"declared": list(ref.SUPPORTED_VERSIONS), "current": ref.PROTOCOL_VERSION,
            **malformed(ref, server), **negotiate(ref, server), **gated(ref, server)}


def verify(result):
    return [
        practice.Check(
            "ANSWER: all three malformed requests return -32602, with three messages",
            all([result["codes"] == [INVALID_PARAMS], len(set(result["messages"])) == 3,
                 all("protocolVersion" in m for m in result["messages"][1:])]),
            f"no _meta, _meta without the key, and the key holding an integer all return "
            f"{result['codes'][0]} with distinct messages -- {result['messages']}. The key's "
            "name appears in the message and nowhere else in the response",
        ),
        practice.Check(
            "ANSWER: 2025-11-25 returns -32022 and the error body is complete",
            all([result["refused_code"] == UNSUPPORTED_VERSION,
                 result["requested"] == OLD_VERSION,
                 result["supported"] == result["declared"], result["retry_ok"]]),
            f"the refusal is {result['refused_code']} with requested "
            f"{result['requested']!r} echoed and supported {result['supported']}; retrying "
            f"with supported[0] is accepted and lists {result['retry_tools']} tools. The "
            "error carries everything needed to fix the request",
        ),
        practice.Check(
            "FINDING: server/discover is not exempt from the version it reports",
            all([result["discover_code"] == UNSUPPORTED_VERSION,
                 result["discover_supported"] == result["discover_result"]]),
            f"the method that exists to report supported versions is refused with "
            f"{result['discover_code']} unless the client already knows one, and its error "
            f"data names {result['discover_supported']} -- the same list its result would "
            "have returned. For a client with no prior knowledge the -32022 body is the "
            "discovery mechanism and the result is the confirmation",
        ),
        practice.Check(
            "FINDING: -32022 is the only self-describing error",
            all([not result["with_data"], result["both_bad"] == UNSUPPORTED_VERSION,
                 len(result["declared"]) == 1]),
            f"{len(result['with_data'])} of the three -32602 responses carry a `data` field, "
            "so a client that omitted the key must parse English to learn its name. And "
            f"`supported` has {len(result['declared'])} entry, so 'choose from supported' is "
            "not a choice: a client holding a list of revisions can only match or fail",
        ),
        practice.Check(
            "FINDING: validation is skipped entirely for notifications",
            result["notification"] is None,
            "`handle` returns before _validate_metadata when 'id' is absent, so a "
            "notification carrying the version 1999-01-01 is accepted silently and does "
            f"nothing -- the response is {result['notification']}. Every guarantee this "
            "exercise checks applies only to messages with an id",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
