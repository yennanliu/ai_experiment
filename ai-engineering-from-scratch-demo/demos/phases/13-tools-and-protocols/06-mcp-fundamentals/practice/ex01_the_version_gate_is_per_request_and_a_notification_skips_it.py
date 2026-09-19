"""Exercise 1 — the version gate is per-request, and a notification skips it.

    Change one request's protocol version to `2027-01-01`. Confirm the error
    code is `-32022` and the data advertises the supported version.

Reading of the exercise: the change is made and the error read back off the
lesson's own `dispatch`. "One request's" is the load-bearing phrase -- the gate
runs per request, so the natural follow-up is which messages are requests, and
one kind of message in this protocol is not.

**ANSWER: `-32022`, with `data` naming both sides.** A `tools/list` carrying
`2027-01-01` returns `{"requested": "2027-01-01", "supported":
["2026-07-28"]}`. The client learns what it asked for and what it may ask for
next, in one round trip and with no handshake to have failed.

**FINDING: the same change on a notification is accepted silently.** Strip the
`id` and `dispatch` returns **`None`** before `validate_request` is ever
called, so a notification carrying `2027-01-01` produces no error at all. The
version gate is not a property of the protocol here; it is a property of
messages that expect a reply.

**FINDING: the gate is per-request because nothing survives a request.** The
same unsupported version is rejected on every method that takes an id --
`server/discover`, `tools/list` and `tools/call` all return `-32022` -- and
accepting one supported request does not make the next one cheaper to check.
That is the cost of "no protocol session", and it is the same cost on every
message forever.

**FINDING: `-32022` is a private code inside JSON-RPC's reserved block.** It
falls in **-32768..-32000**, which JSON-RPC reserves for pre-defined errors,
and it is none of the **5** codes JSON-RPC defines. The lesson's other errors
(`-32600`, `-32601`, `-32602`) are standard; version negotiation had no
standard code to use, so MCP took one from the reserved range.

Structure: `probe` sends one method at one version through the lesson's own
`dispatch`, `as_notification` strips the id, and `RESERVED` and `STANDARD` are
JSON-RPC's own ranges.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "06-mcp-fundamentals"
FUTURE = "2027-01-01"
METHODS = ("server/discover", "tools/list", "tools/call")
RESERVED = (-32768, -32000)
STANDARD = (-32700, -32600, -32601, -32602, -32603)


def probe(ref, method, version, request_id=1):
    params = {"name": "notes_list"} if method == "tools/call" else None
    return ref.dispatch(ref.make_request(request_id, method, params, version=version))


def as_notification(message):
    without_id = dict(message)
    without_id.pop("id")
    return without_id


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rejected = probe(ref, "tools/list", FUTURE)
    error = rejected["error"]
    notification = ref.dispatch(
        as_notification(ref.make_request(1, "tools/list", version=FUTURE)))
    codes = {method: probe(ref, method, FUTURE)["error"]["code"] for method in METHODS}
    accepted = {method: "result" in probe(ref, method, ref.PROTOCOL_VERSION)
                for method in METHODS}
    code = error["code"]
    return {
        "code": code, "message": error["message"], "data": error["data"],
        "requested": error["data"]["requested"], "supported": error["data"]["supported"],
        "notification": notification,
        "codes_by_method": codes, "methods": len(METHODS),
        "accepted": accepted,
        "in_reserved": RESERVED[0] <= code <= RESERVED[1],
        "is_standard": code in STANDARD,
        "standard_count": len(STANDARD),
        "lesson_standard_codes": sorted({-32600, -32601, -32602} & set(STANDARD)),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: -32022, with data naming both sides",
            all([result["code"] == -32022,
                 result["message"] == "Unsupported protocol version",
                 result["requested"] == FUTURE,
                 result["supported"] == ["2026-07-28"]]),
            f"a tools/list carrying {FUTURE} returns {result['code']} and "
            f"data {result['data']}. The client learns what it asked for and what it may ask "
            "for next in one round trip, with no handshake to have failed",
        ),
        practice.Check(
            "FINDING: the same change on a notification is accepted silently",
            result["notification"] is None,
            f"stripping the id makes it a notification, and dispatch returns "
            f"{result['notification']} before validate_request is ever called -- so a "
            "notification carrying an unsupported version produces no error at all. The "
            "version gate is a property of messages that expect a reply, not of the protocol",
        ),
        practice.Check(
            "FINDING: the gate is per-request because nothing survives a request",
            all([set(result["codes_by_method"].values()) == {-32022},
                 len(result["codes_by_method"]) == result["methods"],
                 all(result["accepted"].values())]),
            f"all {result['methods']} methods that take an id reject {FUTURE} the same way, "
            f"{result['codes_by_method']}, and all {result['methods']} accept the supported "
            "version. Accepting one request does not make the next cheaper to check -- that "
            "is the cost of having no protocol session, paid on every message",
        ),
        practice.Check(
            "FINDING: -32022 is a private code inside JSON-RPC's reserved block",
            all([result["in_reserved"], not result["is_standard"],
                 result["standard_count"] == 5,
                 result["lesson_standard_codes"] == [-32602, -32601, -32600]]),
            f"{result['code']} falls in {RESERVED[0]}..{RESERVED[1]}, which JSON-RPC reserves "
            f"for pre-defined errors, and it is none of the {result['standard_count']} codes "
            f"JSON-RPC defines. The lesson's other errors "
            f"{result['lesson_standard_codes']} are standard; version negotiation had no "
            "standard code to use",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
