"""Exercise 2 — there is no state to reuse, so the question cannot be answered wrong.

    Remove `io.modelcontextprotocol/clientCapabilities` from the second
    request. Confirm the server does not reuse capabilities from the first
    request.

Reading of the exercise: the removal is made on a second request that follows a
complete first one, and the confirmation is taken two ways -- the error the
second request returns, and a search of the module for anywhere the first could
have been remembered. The second is the one that settles it, because an error
proves the server did not reuse capabilities *this time*, and only the absence
of storage proves it cannot.

**ANSWER: `-32602`, and the first request's success does not help.** Request one
carries capabilities and returns a complete result; request two omits them and
returns `io.modelcontextprotocol/clientCapabilities is required`. Sending them
again fixes it immediately.

**FINDING: there is nowhere for the capabilities to have been kept.**
`dispatch` takes one message and returns one message; the module holds **0**
mutable containers that outlive a call, and `validate_request` reads only from
its argument. The server does not decline to reuse state -- it has none, which
is what "no protocol session" means as an implementation property rather than a
promise.

**FINDING: the same is true of the version and the client identity.** Removing
the version gives `-32602` on the second request even though the first carried
a supported one. Every element of `params._meta` is re-established per message,
so a client cannot amortise any of it -- the metadata is the per-request cost of
statelessness, and it is **3** keys on every call.

**FINDING: the empty capability object is accepted, so the check is presence
and not content.** `{}` passes. `validate_request` asserts
`isinstance(..., dict)` and nothing reads a capability afterwards -- the field
is required, never consulted, and a client advertising nothing is
indistinguishable from one advertising everything.

Structure: `send` runs one request through the lesson's own `dispatch`,
`without` strips one `_meta` key, and `mutable_module_state` looks for anywhere
a request could be remembered.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "06-mcp-fundamentals"


def send(ref, request_id, drop=None, capabilities=None):
    message = ref.make_request(request_id, "tools/list", capabilities=capabilities)
    if drop is not None:
        message["params"]["_meta"].pop(drop, None)
    return ref.dispatch(message)


def outcome(response):
    if "error" in response:
        return response["error"]["code"], response["error"]["message"]
    return None, response["result"]["resultType"]


def mutable_module_state(ref):
    """Module-level containers that could hold anything between calls."""
    return sorted(name for name in dir(ref)
                  if not name.startswith("_")
                  and isinstance(getattr(ref, name), (dict, list, set))
                  and name.isupper() is False)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    first = send(ref, 1, capabilities={"sampling": {}})
    second = send(ref, 2, drop=ref.CAPABILITIES_KEY)
    third = send(ref, 3, capabilities={"sampling": {}})
    no_version = send(ref, 4, drop=ref.VERSION_KEY)
    empty = send(ref, 5, capabilities={})
    meta_keys = sorted(ref.request_meta())
    return {
        "first": outcome(first), "second": outcome(second), "third": outcome(third),
        "no_version": outcome(no_version), "empty_caps": outcome(empty),
        "mutable_state": mutable_module_state(ref),
        "meta_keys": meta_keys, "meta_count": len(meta_keys),
        "validate_reads_argument_only":
            "message" in ref.validate_request.__code__.co_varnames,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: -32602, and the first request's success does not help",
            all([result["first"] == (None, "complete"),
                 result["second"] ==
                 (-32602, "io.modelcontextprotocol/clientCapabilities is required"),
                 result["third"] == (None, "complete")]),
            f"request one carries capabilities and returns {result['first'][1]}; request two "
            f"omits them and returns {result['second']}. Sending them again on request three "
            f"returns {result['third'][1]} immediately",
        ),
        practice.Check(
            "FINDING: there is nowhere for the capabilities to have been kept",
            all([result["mutable_state"] == [],
                 result["validate_reads_argument_only"]]),
            f"dispatch takes one message and returns one message, and the module holds "
            f"{len(result['mutable_state'])} mutable containers that outlive a call. The "
            "server does not decline to reuse state -- it has none, which is what 'no "
            "protocol session' means as an implementation property rather than a promise",
        ),
        practice.Check(
            "FINDING: the same is true of the version and the client identity",
            all([result["no_version"][0] == -32602, result["meta_count"] == 3,
                 result["meta_keys"] == ["io.modelcontextprotocol/clientCapabilities",
                                         "io.modelcontextprotocol/clientInfo",
                                         "io.modelcontextprotocol/protocolVersion"]]),
            f"removing the version gives {result['no_version']} on a later request even "
            f"though an earlier one carried a supported version. All {result['meta_count']} "
            f"_meta keys {result['meta_keys']} are re-established per message, so a client "
            "cannot amortise any of it",
        ),
        practice.Check(
            "FINDING: the empty capability object is accepted, so the check is presence",
            result["empty_caps"] == (None, "complete"),
            f"capabilities of {{}} returns {result['empty_caps'][1]}. validate_request "
            "asserts isinstance(..., dict) and nothing reads a capability afterwards -- the "
            "field is required, never consulted, and a client advertising nothing is "
            "indistinguishable from one advertising everything",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
