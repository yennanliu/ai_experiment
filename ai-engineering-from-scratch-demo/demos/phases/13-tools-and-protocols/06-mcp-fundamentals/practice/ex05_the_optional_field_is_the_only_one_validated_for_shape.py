"""Exercise 5 — the optional field is the only one validated for shape.

    Add an optional `clientInfo` omission test. The request should remain valid
    because client identity is recommended, not required.

Reading of the exercise: the omission test is added and passes, and then the
same field is sent malformed rather than absent -- because "recommended, not
required" describes when it may be missing and says nothing about what happens
when it is present and wrong. That turns out to be the one place in
`validate_request` where a field's *contents* are checked.

**ANSWER: omitting `clientInfo` leaves the request valid.** A `tools/list` with
only the version and capability keys returns a complete result. The lesson's own
`request_meta` takes `include_client_info=False` for exactly this, so the test
is a one-argument change.

**FINDING: the optional field is the only one whose shape is validated.**
`protocolVersion` must be a `str` and `clientCapabilities` a `dict` -- presence
checks. `clientInfo` is checked for being a dict **and** for `name` and
`version` both being strings, so `{"name": "x"}` is rejected while
`clientCapabilities: {}` sails through. The field that may be absent is the
field that is inspected hardest.

**FINDING: that asymmetry is the right way round, and it is an accident of what
is consulted.** Nothing reads a capability, so nothing can be broken by a wrong
one. `clientInfo` is the only `_meta` key whose value is destined for anything
but a presence check -- a log line, a rate-limit bucket, an audit trail -- so
its shape is the only shape that can corrupt something downstream.

**FINDING: and "recommended" is not enforced anywhere.** A client that never
sends `clientInfo` is never warned: **3 of 3** methods accept its absence, and
the server's own reply always carries `serverInfo`. The recommendation is
one-directional in the code as well as in the spec -- the server identifies
itself unconditionally and asks nothing of the client.

Structure: `send` builds a request with or without the field, `malformed`
substitutes a bad `clientInfo`, and `outcome` reduces a response to a code and a
message.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "06-mcp-fundamentals"
METHODS = ("server/discover", "tools/list", "tools/call")
MALFORMED = ({"name": "only-a-name"}, {"version": "1.0.0"},
             {"name": 1, "version": "1.0.0"}, "a string", [])


def send(ref, method, request_id=1, include_client_info=True, client_info=None):
    params = {"name": "notes_list"} if method == "tools/call" else None
    message = ref.make_request(request_id, method, params)
    meta = message["params"]["_meta"]
    if not include_client_info:
        meta.pop(ref.CLIENT_INFO_KEY, None)
    if client_info is not None:
        meta[ref.CLIENT_INFO_KEY] = client_info
    return ref.dispatch(message)


def outcome(response):
    if "error" in response:
        return response["error"]["code"]
    return response["result"]["resultType"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    omitted = {method: outcome(send(ref, method, include_client_info=False))
               for method in METHODS}
    malformed = [outcome(send(ref, "tools/list", client_info=value))
                 for value in MALFORMED]
    meta_without = ref.request_meta(include_client_info=False)
    server_info = send(ref, "tools/list")["result"]["_meta"][ref.SERVER_INFO_KEY]
    return {
        "omitted": omitted,
        "all_accept": set(omitted.values()) == {"complete"},
        "methods": len(METHODS),
        "malformed": malformed,
        "malformed_rejected": sum(value == -32602 for value in malformed),
        "malformed_tried": len(MALFORMED),
        "meta_without_keys": sorted(meta_without),
        "empty_caps_ok": outcome(
            ref.dispatch(ref.make_request(9, "tools/list", capabilities={}))) == "complete",
        "server_info": sorted(server_info),
        "server_info_always": server_info == ref.SERVER_INFO,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: omitting clientInfo leaves the request valid",
            all([result["all_accept"], result["omitted"]["tools/list"] == "complete",
                 result["meta_without_keys"] ==
                 ["io.modelcontextprotocol/clientCapabilities",
                  "io.modelcontextprotocol/protocolVersion"]]),
            f"all {result['methods']} methods accept a request carrying only "
            f"{result['meta_without_keys']} -- {result['omitted']}. The lesson's own "
            "request_meta takes include_client_info=False for exactly this, so the test is a "
            "one-argument change",
        ),
        practice.Check(
            "FINDING: the optional field is the only one whose shape is validated",
            all([result["malformed_rejected"] == result["malformed_tried"],
                 result["malformed_tried"] == 5,
                 set(result["malformed"]) == {-32602},
                 result["empty_caps_ok"]]),
            f"{result['malformed_rejected']} of {result['malformed_tried']} malformed "
            f"clientInfo values are rejected -- a name without a version, a version without "
            f"a name, a non-string name, a bare string and a list all give -32602 -- while "
            f"clientCapabilities of {{}} sails through. The field that may be absent is the "
            "field inspected hardest",
        ),
        practice.Check(
            "FINDING: that asymmetry is an accident of what is consulted",
            all([result["empty_caps_ok"], result["malformed_rejected"] == 5]),
            "nothing reads a capability, so nothing can be broken by a wrong one. clientInfo "
            "is the only _meta key whose value is destined for anything but a presence check "
            "-- a log line, a rate-limit bucket, an audit trail -- so its shape is the only "
            "one that can corrupt something downstream",
        ),
        practice.Check(
            "FINDING: and 'recommended' is not enforced anywhere",
            all([result["all_accept"], result["server_info_always"],
                 result["server_info"] == ["name", "version"]]),
            f"a client that never sends clientInfo is never warned: {result['methods']} of "
            f"{result['methods']} methods accept its absence. Meanwhile the server's reply "
            f"always carries serverInfo {result['server_info']}. The recommendation is "
            "one-directional in the code as well as the spec -- the server identifies itself "
            "unconditionally and asks nothing of the client",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
