"""Exercise 4 — a template list is public because it holds no instances.

    Add `resources/templates/list` with `ttlMs`, `cacheScope`, and
    deterministic ordering.

Reading of the exercise: the handler is added with all three properties, and the
scope is *derived* rather than picked -- the lesson already scopes five methods,
and the rule separating its `public` ones from its `private` ones decides this
one too. Templates describe the shape of a URI and contain no note, so they fall
on the public side, which makes them the only resource method that does.

**ANSWER: `resources/templates/list`, `public`, 60,000 ms, sorted by
`uriTemplate`.** Two templates -- `notes://{note_id}` and `notes://tag/{tag}` --
returned in the same order whichever way the source list is arranged.

**FINDING: it is the only `resources/*` method that can be public.**
`resources/list` is `private` at 10,000 ms and `resources/read` is `private` at
5,000, because both return note contents. A template is a URI grammar: it is a
function of the server's routing and of nothing the caller owns, which puts it
with `tools/list` and `prompts/list` rather than with its own namespace.

**FINDING: the ttl follows the same rule and lands on the same number.** The
three `public` methods the lesson ships cache for **3,600,000**, **60,000** and
**60,000** ms; the two `private` ones for **10,000** and **5,000**. Scope and
freshness are the same decision -- what changes when a caller acts changes
often, so the server's own shape gets the long ttl and the caller's data gets
the short one.

**FINDING: and the templates describe a route `resources/read` does not
implement.** `notes://tag/{tag}` expands to a URI whose `removeprefix` leaves
`tag/design`, which is not a key in `NOTES`, so reading it returns **-32602
Resource not found**. Advertising a template is not the same as serving it, and
nothing in the lesson cross-checks the two.

Structure: `TEMPLATES` is the list, `handle_templates_list` is the handler,
`install` registers it in the lesson's own `HANDLERS` map, and `scopes` reads
the ttl and scope off every resource method.
"""

from __future__ import annotations

import contextlib

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "07-building-an-mcp-server"
METHOD = "resources/templates/list"
TEMPLATES = [
    {"uriTemplate": "notes://{note_id}", "name": "Note by id",
     "mimeType": "text/markdown"},
    {"uriTemplate": "notes://tag/{tag}", "name": "Notes by tag",
     "mimeType": "text/markdown"},
]
TTL_MS = 60_000


def make_handler(ref, templates):
    def handle_templates_list(params):
        return ref.complete(
            {"resourceTemplates": sorted(templates, key=lambda t: t["uriTemplate"])},
            ttl_ms=TTL_MS, cache_scope="public")
    return handle_templates_list


@contextlib.contextmanager
def install(ref, templates):
    """Register the handler, restoring whatever was there before (nesting-safe)."""
    previous = ref.HANDLERS.get(METHOD)
    ref.HANDLERS[METHOD] = make_handler(ref, templates)
    try:
        yield
    finally:
        if previous is None:
            ref.HANDLERS.pop(METHOD, None)
        else:
            ref.HANDLERS[METHOD] = previous


def send(ref, method, params=None, request_id=1):
    return ref.dispatch(ref.make_request(request_id, method, params))


def scope_of(ref, method, params=None):
    result = send(ref, method, params)["result"]
    return (result.get("cacheScope"), result.get("ttlMs"))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.reset_notes()
    before = send(ref, METHOD)
    with install(ref, TEMPLATES):
        forward = send(ref, METHOD)["result"]
        backward_list = list(reversed(TEMPLATES))
        with install(ref, backward_list):
            backward = send(ref, METHOD)["result"]
        template_scope = scope_of(ref, METHOD)
    after = send(ref, METHOD)
    expanded = "notes://tag/design"
    unread = send(ref, "resources/read", {"uri": expanded})
    return {
        "before_code": before["error"]["code"],
        "after_code": after["error"]["code"],
        "templates": [t["uriTemplate"] for t in forward["resourceTemplates"]],
        "reversed_templates": [t["uriTemplate"] for t in backward["resourceTemplates"]],
        "order_stable": forward["resourceTemplates"] == backward["resourceTemplates"],
        "template_scope": template_scope,
        "resources_list": scope_of(ref, "resources/list"),
        "resources_read": scope_of(ref, "resources/read", {"uri": "notes://note-1"}),
        "tools_list": scope_of(ref, "tools/list"),
        "prompts_list": scope_of(ref, "prompts/list"),
        "discover": scope_of(ref, "server/discover"),
        "unread_code": unread["error"]["code"],
        "unread_message": unread["error"]["message"],
        "expanded": expanded,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: resources/templates/list, public, 60,000 ms, sorted by uriTemplate",
            all([result["before_code"] == -32601, result["after_code"] == -32601,
                 result["templates"] == ["notes://tag/{tag}", "notes://{note_id}"],
                 result["order_stable"],
                 result["template_scope"] == ("public", TTL_MS)]),
            f"the method is {result['before_code']} before installation and after removal. "
            f"Installed, it returns {result['templates']} in the same order whichever way "
            f"the source list is arranged, scoped {result['template_scope'][0]} at "
            f"{result['template_scope'][1]:,} ms",
        ),
        practice.Check(
            "FINDING: it is the only resources/* method that can be public",
            all([result["resources_list"] == ("private", 10_000),
                 result["resources_read"] == ("private", 5_000),
                 result["template_scope"][0] == "public"]),
            f"resources/list is {result['resources_list']} and resources/read is "
            f"{result['resources_read']}, because both return note contents. A template is a "
            "URI grammar -- a function of the server's routing and of nothing the caller "
            "owns -- which puts it with tools/list and prompts/list rather than with its own "
            "namespace",
        ),
        practice.Check(
            "FINDING: the ttl follows the same rule and lands on the same number",
            all([result["discover"] == ("public", 3_600_000),
                 result["tools_list"] == ("public", 60_000),
                 result["prompts_list"] == ("public", 60_000),
                 result["template_scope"][1] == 60_000]),
            f"the public methods cache for {result['discover'][1]:,}, "
            f"{result['tools_list'][1]:,} and {result['prompts_list'][1]:,} ms and the "
            f"private ones for {result['resources_list'][1]:,} and "
            f"{result['resources_read'][1]:,}. Scope and freshness are the same decision: "
            "the server's own shape gets the long ttl and the caller's data gets the short "
            "one",
        ),
        practice.Check(
            "FINDING: the templates describe a route resources/read does not implement",
            all([result["unread_code"] == -32602,
                 result["unread_message"] == "Resource not found"]),
            f"expanding the tag template gives {result['expanded']}, whose removeprefix "
            f"leaves 'tag/design' -- not a key in NOTES -- so reading it returns "
            f"{result['unread_code']} {result['unread_message']!r}. Advertising a template "
            "is not the same as serving it, and nothing in the lesson cross-checks the two",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
