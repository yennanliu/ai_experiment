"""Exercise 1 — the tool degrades where the resource refuses.

    Change the client capability to an empty extension map. Confirm
    `tools/list` keeps the tool but removes the UI binding.

Reading of the exercise: confirming the listing is one request, and one
request does not say what the capability governs. So the same two capability
maps are sent to every method that consults `_apps_enabled`, and the two
methods turn out to disagree about what a missing capability means -- one
quietly drops a field and the other answers an error.

**ANSWER: the tool survives and its `_meta` does not.** With
`extensions: {io.modelcontextprotocol/ui: {}}` the listing carries **1** tool
whose `_meta.ui.resourceUri` is `ui://notes/timeline.html`; with
`extensions: {}` it carries the same **1** tool and **no** `_meta` at all. The
tool's name, description and schema are byte-identical either way.

**FINDING: `resources/read` refuses where `tools/list` degrades.** The same
empty map that silently removes the binding makes the UI resource answer
**-32021** with `requiredCapabilities`. A client that lost the capability
between the two calls is handed a tool it can invoke and cannot render, with
nothing in the listing to say so.

**FINDING: the capability is read per request, and the result is cached as
`public`.** Both listings carry `ttlMs: 60000, cacheScope: "public"`, and they
differ. A cache keyed on the method alone would hand a UI-bound listing to a
client that never asked for one -- the scope says the response depends on
nothing user-specific, and it depends on the caller's capabilities.

**FINDING: the negotiation is "a dict or nothing".** `_apps_enabled` requires
`isinstance(extensions[APPS_EXTENSION], dict)`, so of `{}`,
`{"extensions": {}}`, `{"extensions": {ui: None}}` and
`{"extensions": {ui: {}}}` exactly **1** enables the binding. A client sending
`true` to mean yes is read as absent, with no error either way.

Structure: `ask` sends one method under one capability map and returns the
status with the part of the result that moved.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "14-mcp-apps"


def ask(ref, server, method, params=None, *, apps=True):
    body, headers = ref.make_request(method, 1, params, apps=apps)
    status, response = server.handle(body, headers)
    return status, response


def tool_of(response):
    return response["result"]["tools"][0]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    server = ref.McpAppServer()
    enabled = ask(ref, server, "tools/list")[1]
    disabled = ask(ref, server, "tools/list", apps=False)[1]
    read_ok = ask(ref, server, "resources/read", {"uri": ref.RESOURCE_URI})
    read_no = ask(ref, server, "resources/read", {"uri": ref.RESOURCE_URI}, apps=False)
    shapes = [{}, {"extensions": {}}, {"extensions": {ref.APPS_EXTENSION: None}},
              {"extensions": {ref.APPS_EXTENSION: {}}}]
    return {
        "bound": tool_of(enabled).get("_meta", {}).get("ui", {}).get("resourceUri"),
        "unbound_meta": "_meta" in tool_of(disabled),
        "counts": [len(enabled["result"]["tools"]), len(disabled["result"]["tools"])],
        "same_tool": ({k: v for k, v in tool_of(enabled).items() if k != "_meta"}
                      == dict(tool_of(disabled))),
        "read_ok": read_ok[0], "read_no": read_no[0],
        "read_no_code": read_no[1]["error"]["code"],
        "required": read_no[1]["error"]["data"]["requiredCapabilities"],
        "hints": [(r["result"]["ttlMs"], r["result"]["cacheScope"])
                  for r in (enabled, disabled)],
        "listings_differ": enabled["result"] != disabled["result"],
        "enabling": [ref.McpAppServer._apps_enabled({ref.CLIENT_CAPABILITIES_META: s})
                     for s in shapes],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the tool survives and its _meta does not",
            all([result["bound"] == "ui://notes/timeline.html",
                 not result["unbound_meta"], result["counts"] == [1, 1],
                 result["same_tool"]]),
            f"with the extension the listing carries {result['counts'][0]} tool bound to "
            f"{result['bound']}; with an empty map it carries {result['counts'][1]} tool and "
            f"no _meta. Everything else about the tool is identical: {result['same_tool']}",
        ),
        practice.Check(
            "FINDING: resources/read refuses where tools/list degrades",
            all([result["read_ok"] == 200, result["read_no"] == 400,
                 result["read_no_code"] == -32021,
                 result["required"] == {"extensions": {"io.modelcontextprotocol/ui": {}}}]),
            f"the same empty map that silently drops the binding makes the UI resource answer "
            f"HTTP {result['read_no']} / {result['read_no_code']} naming "
            f"{result['required']}. A client that lost the capability between the two calls "
            "gets a tool it can invoke and cannot render, with nothing in the listing to say "
            "so",
        ),
        practice.Check(
            "FINDING: the capability is read per request and the result is cached as public",
            all([result["hints"] == [(60_000, "public")] * 2, result["listings_differ"]]),
            f"both listings carry {result['hints'][0]} and they differ. A cache keyed on the "
            "method alone would hand a UI-bound listing to a client that never asked for one "
            "-- the scope says the response depends on nothing user-specific, and it depends "
            "on the caller's capabilities",
        ),
        practice.Check(
            "FINDING: the negotiation is a dict or nothing",
            result["enabling"] == [False, False, False, True],
            f"_apps_enabled requires isinstance(extensions[ui], dict), so of an absent map, "
            f"an empty one, one holding None and one holding {{}}, exactly "
            f"{sum(result['enabling'])} enables the binding. A client sending true to mean "
            "yes is read as absent, with no error either way",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
