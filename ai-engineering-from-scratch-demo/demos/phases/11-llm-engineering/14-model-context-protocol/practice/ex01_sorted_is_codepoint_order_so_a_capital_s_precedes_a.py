"""Exercise 1 — sorted() is codepoint order, so a capital S precedes a.

    Add a `subtract` tool and confirm `tools/list` remains alphabetically
    ordered.

Reading of the exercise: "remains ordered" is a claim about `tools/list`, not
about the registry, so the tool is added and the listing re-read -- and then
the claim is pushed until it breaks, because `sorted(key=name)` is not the same
promise as "alphabetically ordered". While `subtract` is registered its
`inputSchema` is also exercised, since a schema that nothing checks is the
other half of adding a tool.

**ANSWER: `tools/list` returns `['add', 'delete_user', 'subtract']` and it is
ordered.** The handler sorts on every call, so registration order cannot affect
it: `subtract` was added last and lands third by name, not by arrival.

**FINDING: the ordering is ASCII, not alphabetical.** Add `Subtract`, `_hidden`
and `2fa` and the listing is `['2fa', 'Subtract', '_hidden', 'add',
'delete_user', 'subtract']` -- still `sorted()`, and no longer alphabetical by
any human reading: digits before capitals, capitals before underscore,
underscore before lowercase. A client that renders the list as-is shows
`Subtract` above `add`. `key=str.casefold` is the one-word fix.

**FINDING: nothing validates `inputSchema`.** `tools/list` publishes
`{"a": {"type": "integer"}, "b": {"type": "integer"}}` for `add`, and
`tools/call` hands `arguments` straight to the handler as `**kwargs`. So
`add(a="x", b="y")` returns `{"sum": "xy"}` with `isError: False` -- the
schema is documentation, and string concatenation is a valid sum.

**FINDING: a missing required argument is reported as an internal error.**
`add(a=1)` raises `TypeError` inside the handler, which the `except Exception`
turns into **-32603 "tool handler failed"**. The JSON-RPC code for a bad
argument is -32602, and the server uses it correctly for an unknown tool name
one line earlier -- so the same class of mistake gets two different codes
depending on whether the server or Python noticed it.

**FINDING: registering a name twice replaces it silently.** The registry is a
dict keyed by name, so a second `subtract` leaves the tool count at 3 and the
listing unchanged in length. There is no duplicate-name error to catch.

Structure: `fresh` builds a server carrying the lesson's own primitives so the
module singleton is left alone, `call` sends one well-formed request, `SUBTRACT`
is the new tool and `ODD` the three names that break the alphabetical claim.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "14-model-context-protocol"
SCHEMA = {"type": "object",
          "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
          "required": ["a", "b"]}
ODD = ("Subtract", "_hidden", "2fa")


def fresh(ref):
    server = ref.MCPServer("practice")
    server.tools.update(ref.server.tools)
    server.resources.update(ref.server.resources)
    server.prompts.update(ref.server.prompts)
    return server


def call(ref, server, method, params=None, request_id=1):
    body = dict(params or {})
    body["_meta"] = ref.request_metadata()
    return server.handle({"jsonrpc": "2.0", "id": request_id, "method": method, "params": body})


def names(ref, server):
    return [tool["name"] for tool in call(ref, server, "tools/list")["result"]["tools"]]


def subtract(a, b):
    return {"difference": a - b}


def register_subtract(ref, server):
    server.tools["subtract"] = ref.Tool("subtract", "Subtract b from a.", SCHEMA, subtract)


def ordering(ref, server):
    """The listing with subtract alone, and then with three awkward names."""
    plain = names(ref, server)
    for odd in ODD:
        server.tools[odd] = ref.Tool(odd, "awkward name", {}, lambda: {})
    mixed = names(ref, server)
    for odd in ODD:
        del server.tools[odd]
    return {"listed": plain, "ascii_sorted": plain == sorted(plain), "mixed": mixed,
            "mixed_ascii_sorted": mixed == sorted(mixed),
            "mixed_alphabetical": mixed == sorted(mixed, key=str.casefold)}


def schema_checks(ref, server):
    """What tools/call does with arguments the published schema forbids."""
    listed = {tool["name"]: tool["inputSchema"]
              for tool in call(ref, server, "tools/list")["result"]["tools"]}
    strings = call(ref, server, "tools/call", {"name": "add", "arguments": {"a": "x", "b": "y"}})
    missing = call(ref, server, "tools/call", {"name": "add", "arguments": {"a": 1}})
    unknown = call(ref, server, "tools/call", {"name": "nope", "arguments": {}})
    return {"published_types": [listed["add"]["properties"][k]["type"] for k in ("a", "b")],
            "concatenated": strings["result"]["content"][0]["text"],
            "concat_is_error": strings["result"]["isError"],
            "missing_code": missing["error"]["code"],
            "unknown_code": unknown["error"]["code"]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    server = fresh(ref)
    before = len(server.tools)
    register_subtract(ref, server)
    called = call(ref, server, "tools/call",
                  {"name": "subtract", "arguments": {"a": 5, "b": 3}})
    added = len(server.tools)
    server.tools["subtract"] = ref.Tool("subtract", "a replacement", {}, lambda: {"x": 1})
    return {"before": before, "after_add": added, "after_duplicate": len(server.tools),
            "subtract_output": called["result"]["content"][0]["text"],
            **ordering(ref, server), **schema_checks(ref, server)}


def verify(result):
    return [
        practice.Check(
            "ANSWER: tools/list returns ['add', 'delete_user', 'subtract'] and is ordered",
            all([result["listed"] == ["add", "delete_user", "subtract"],
                 result["ascii_sorted"], result["after_add"] == result["before"] + 1]),
            f"registering subtract takes the registry {result['before']} -> "
            f"{result['after_add']} and the listing is {result['listed']}. The handler sorts "
            "on every call, so the tool added last lands third by name rather than by "
            f"arrival, and subtract(5, 3) returns {result['subtract_output']}",
        ),
        practice.Check(
            "FINDING: the ordering is ASCII, not alphabetical",
            all([result["mixed_ascii_sorted"], not result["mixed_alphabetical"],
                 result["mixed"][:3] == ["2fa", "Subtract", "_hidden"]]),
            f"adding {list(ODD)} gives {result['mixed']} -- still sorted(), and no longer "
            "alphabetical by any human reading: digits before capitals, capitals before "
            "underscore, underscore before lowercase. A client rendering the list as-is "
            "shows Subtract above add, and key=str.casefold is the one-word fix",
        ),
        practice.Check(
            "FINDING: nothing validates inputSchema",
            all([result["published_types"] == ["integer", "integer"],
                 result["concatenated"] == '{"sum": "xy"}',
                 not result["concat_is_error"]]),
            f"tools/list publishes {result['published_types']} for add's two arguments and "
            "tools/call hands `arguments` straight to the handler as **kwargs, so "
            f"add(a='x', b='y') returns {result['concatenated']} with isError "
            f"{result['concat_is_error']}. The schema is documentation",
        ),
        practice.Check(
            "FINDING: a missing required argument is reported as an internal error",
            all([result["missing_code"] == -32603, result["unknown_code"] == -32602]),
            f"add(a=1) raises TypeError in the handler and `except Exception` turns it into "
            f"{result['missing_code']} 'tool handler failed', while an unknown tool name one "
            f"line earlier is {result['unknown_code']}. The same class of mistake gets two "
            "codes depending on whether the server or Python noticed it",
        ),
        practice.Check(
            "FINDING: registering a name twice replaces it silently",
            result["after_duplicate"] == result["after_add"],
            f"the registry is a dict keyed by name, so a second subtract leaves the count at "
            f"{result['after_duplicate']} and the listing the same length. There is no "
            "duplicate-name error to catch, and the second registration wins",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
