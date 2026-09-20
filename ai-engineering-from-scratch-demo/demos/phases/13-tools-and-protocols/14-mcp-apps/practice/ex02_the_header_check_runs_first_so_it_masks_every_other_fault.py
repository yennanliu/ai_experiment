"""Exercise 2 — the header check runs first, so it masks every other fault.

    Send `Mcp-Name: ui://notes/other.html` with a body that reads the
    timeline. Confirm error `-32020`.

Reading of the exercise: one mismatched header gives one code, so the question
worth asking is what *else* that code is hiding. The same mismatch is
therefore sent alongside each of the other three faults `_validate` can
find -- a missing capability, an unsupported version, an unknown URI -- and
the header wins every time.

**ANSWER: `-32020 Mcp-Name header does not match body`, on HTTP `400`.** The
body asks for `ui://notes/timeline.html` and the header claims
`ui://notes/other.html`; the request never reaches the resource handler.

**FINDING: the same mismatch masks three other faults.** Sent with no apps
capability it still answers **-32020**, not the **-32021** that capability
alone produces; the same is true against an unsupported version and an unknown
URI. `_validate` checks the mirrored headers before the version and long
before the handler, so a client with two faults fixes them one round trip at a
time and never learns the second until the first is gone.

**FINDING: the expected name is `params.name or params.uri`, and `or` is not
`if present`.** A `tools/call` naming the empty string falls through to
`params.uri`, and with no uri the comparison target becomes `None` -- so
`make_request` sends `Mcp-Name: ""` and the server expects `None`. An empty
name is unreachable rather than rejected.

**FINDING: the check covers exactly three methods, and the other two are
unguarded by construction.** `tools/list` and `resources/list` take no name,
so nothing is mirrored and nothing can mismatch. The header is a routing hint
for requests that address one object, which is why the list methods do not
have one -- not an oversight, but also not stated anywhere in the validation.

Structure: `send` posts one body with the reference's own headers after
`override` has had a chance to rewrite them.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "14-mcp-apps"
OTHER = "ui://notes/other.html"
MISMATCH = "Mcp-Name header does not match body"
NAMED = ("tools/call", "resources/read", "prompts/get")


def send(ref, server, method, params=None, *, apps=True, override=None, version=None):
    body, headers = ref.make_request(method, 1, params, apps=apps)
    if version is not None:
        body["params"]["_meta"][ref.PROTOCOL_META] = version
        headers["MCP-Protocol-Version"] = version
    if override:
        headers.update(override)
    status, response = server.handle(body, headers)
    error = (response or {}).get("error", {})
    return status, error.get("code"), error.get("message")


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    server = ref.McpAppServer()
    read = {"uri": ref.RESOURCE_URI}
    wrong = {"Mcp-Name": OTHER}
    empty_body, empty_headers = ref.make_request("tools/call", 2, {"name": ""})
    return {
        "answer": send(ref, server, "resources/read", read, override=wrong),
        "clean": send(ref, server, "resources/read", read),
        "no_apps": send(ref, server, "resources/read", read, apps=False),
        "masks_capability": send(ref, server, "resources/read", read,
                                 apps=False, override=wrong),
        "masks_version": send(ref, server, "resources/read", read,
                              override=wrong, version="2027-01-01"),
        "version_alone": send(ref, server, "resources/read", read, version="2027-01-01"),
        "masks_unknown": send(ref, server, "resources/read", {"uri": OTHER},
                              override={"Mcp-Name": ref.RESOURCE_URI}),
        "unknown_alone": send(ref, server, "resources/read", {"uri": OTHER}),
        "sent_empty_name": empty_headers.get("Mcp-Name"),
        "expected_empty": (empty_body["params"].get("name")
                           or empty_body["params"].get("uri")),
        "empty_result": server.handle(empty_body, empty_headers)[1]["error"]["code"],
        "named_methods": list(NAMED),
        "list_headers": sorted(ref.make_request("tools/list", 3)[1]),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: -32020 Mcp-Name header does not match body, on HTTP 400",
            all([result["answer"] == (400, -32020, MISMATCH),
                 result["clean"] == (200, None, None)]),
            f"the body asks for {ref_uri()} and the header claims {OTHER}, giving "
            f"{result['answer']}. The same request with the matching header answers "
            f"HTTP {result['clean'][0]}, so the mismatch is the whole difference",
        ),
        practice.Check(
            "FINDING: the same mismatch masks three other faults",
            all([result["masks_capability"][1] == -32020, result["no_apps"][1] == -32021,
                 result["masks_version"][1] == -32020, result["version_alone"][1] == -32022,
                 result["masks_unknown"][1] == -32020, result["unknown_alone"][1] == -32602]),
            f"with no capability the mismatch still answers {result['masks_capability'][1]} "
            f"where the capability alone gives {result['no_apps'][1]}; against an unsupported "
            f"version {result['masks_version'][1]} where the version alone gives "
            f"{result['version_alone'][1]}; and against an unknown URI "
            f"{result['masks_unknown'][1]} where the URI alone gives "
            f"{result['unknown_alone'][1]}. Two faults cost two round trips",
        ),
        practice.Check(
            "FINDING: the expected name is `params.name or params.uri`, not `if present`",
            all([result["sent_empty_name"] == "", result["expected_empty"] is None,
                 result["empty_result"] == -32020]),
            f"a tools/call naming the empty string makes make_request send Mcp-Name "
            f"{result['sent_empty_name']!r} while the server computes "
            f"{result['expected_empty']}, because `or` falls through a falsy name to "
            f"params.uri. The request answers {result['empty_result']} -- an empty name is "
            "unreachable rather than rejected",
        ),
        practice.Check(
            "FINDING: the check covers three methods, and the list methods have no name to mirror",
            all([result["named_methods"] == list(NAMED),
                 result["list_headers"] == ["MCP-Protocol-Version", "Mcp-Method"]]),
            f"the mirrored-name rule applies to {result['named_methods']}, and tools/list "
            f"goes out with {result['list_headers']} -- no Mcp-Name to mismatch. The header "
            "is a routing hint for requests that address one object, which the validation "
            "implements and never states",
        ),
    ]


def ref_uri():
    return "ui://notes/timeline.html"


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
