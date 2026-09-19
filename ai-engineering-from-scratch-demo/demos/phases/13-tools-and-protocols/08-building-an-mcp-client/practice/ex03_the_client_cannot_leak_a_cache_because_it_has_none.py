"""Exercise 3 — the client cannot leak a cache, because it has none.

    Add `cacheScope: "private"` tool lists for two authorization contexts.
    Confirm the client never shares one context's cached result with the other.

Reading of the exercise: the two contexts are built and both served
`private` tool lists, and the confirmation is attempted the way the exercise
frames it -- as a property of the client's cache. There is no cache. So the
property holds vacuously, and the exercise is only answerable by building the
cache it assumes, which is what makes the `private` scope mean something.

**ANSWER: the client never shares a cached result because it never keeps one.**
`MultiServerClient` has **0** attributes matching a cache, and `discover_tools`
re-sends `tools/list` on every call: two rounds against one peer put **2**
`tools/list` messages on the transport. The scope field is read by nobody.

**FINDING: a cache keyed on the peer alone would leak.** Adding a
`{peer: result}` cache holds **1** entry and serves the admin's **3**-tool list
to the reader, who is entitled to **1**. That is the leak the scope field exists
to prevent, and nothing else in the response says so.

**FINDING: re-keying on `(peer, principal)` is what `private` instructs.** Two
entries for two principals, reproducing the uncached answer exactly. `private`
does not forbid caching -- it names the key.

**FINDING: the client has no principal to key on, which is why the cache was
never written.** `CLIENT_INFO` is a name and a version and `client_capabilities`
is a feature list; neither varies per authorization context. The key `private`
requires does not exist anywhere in the client's state -- it would have to come
from the transport, which is where the credential lives.

Structure: `ScopedServer` returns a different tool list per principal,
`transport_for` binds one principal to a transport, and `cached` serves both
principals through a cache built from a supplied key function.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "08-building-an-mcp-client"
ADMIN, READER = "admin", "reader"
BY_PRINCIPAL = {ADMIN: ["notes_create", "notes_delete", "notes_list"],
                READER: ["notes_list"]}


class ScopedServer:
    """A modern server whose tools/list depends on the caller, scoped private."""

    def __init__(self, ref):
        self.ref = ref
        self.calls = 0

    def respond(self, message, principal):
        method = message.get("method")
        request_id = message.get("id")
        if method == "server/discover":
            return {"jsonrpc": "2.0", "id": request_id,
                    "result": self.ref.complete(
                        {"name": "scoped", "version": "1.0.0"},
                        {"supportedVersions": [self.ref.PROTOCOL_VERSION],
                         "capabilities": {"tools": {"listChanged": False}}},
                        ttl_ms=3_600_000, cache_scope="public")}
        if method == "tools/list":
            self.calls += 1
            tools = [self.ref.tool(name, f"{name} for {principal}.")
                     for name in BY_PRINCIPAL[principal]]
            return {"jsonrpc": "2.0", "id": request_id,
                    "result": self.ref.complete({"name": "scoped", "version": "1.0.0"},
                                                {"tools": tools},
                                                ttl_ms=30_000, cache_scope="private")}
        return None


def transport_for(server, principal):
    def transport(message, timeout_ms=None):
        return server.respond(message, principal)
    return transport


def tools_for(ref, server, principal):
    client = ref.MultiServerClient()
    client.add_server("s", transport_for(server, principal))
    client.connect_all()
    client.discover_tools()
    return [tool["name"] for tool in client.peers["s"].tools]


def cached(ref, server, principals, key):
    """Serve each principal through a cache built with `key`."""
    store, served = {}, {}
    for principal in principals:
        cache_key = key("s", principal)
        if cache_key not in store:
            store[cache_key] = tools_for(ref, server, principal)
        served[principal] = store[cache_key]
    return served, len(store)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    server = ScopedServer(ref)
    direct = {principal: tools_for(ref, server, principal)
              for principal in (ADMIN, READER)}
    naive, naive_entries = cached(ref, server, (ADMIN, READER), lambda peer, _: peer)
    scoped, scoped_entries = cached(ref, server, (ADMIN, READER),
                                    lambda peer, principal: (peer, principal))
    counted = ScopedServer(ref)
    tools_for(ref, counted, ADMIN)
    tools_for(ref, counted, ADMIN)
    client = ref.MultiServerClient()
    return {
        "cache_attributes": [name for name in vars(client) if "cache" in name.lower()],
        "repeat_calls": counted.calls,
        "direct": direct,
        "naive": naive, "naive_entries": naive_entries,
        "naive_leaks": naive[READER] != direct[READER],
        "scoped": scoped, "scoped_entries": scoped_entries,
        "scoped_leaks": scoped[READER] != direct[READER],
        "principals": len(BY_PRINCIPAL),
        "client_info": sorted(ref.CLIENT_INFO),
        "capabilities": sorted(client.client_capabilities),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the client never shares a cached result because it never keeps one",
            all([result["cache_attributes"] == [], result["repeat_calls"] == 2,
                 result["direct"][ADMIN] == BY_PRINCIPAL[ADMIN],
                 result["direct"][READER] == BY_PRINCIPAL[READER]]),
            f"MultiServerClient has {len(result['cache_attributes'])} attributes matching a "
            f"cache, and discover_tools re-sends tools/list every call -- two rounds against "
            f"one peer put {result['repeat_calls']} tools/list messages on the transport. "
            f"Each principal sees its own list, {result['direct']}, and the scope field is "
            "read by nobody",
        ),
        practice.Check(
            "FINDING: a cache keyed on the peer alone would leak",
            all([result["naive_leaks"], result["naive_entries"] == 1,
                 result["naive"][READER] == BY_PRINCIPAL[ADMIN],
                 len(BY_PRINCIPAL[ADMIN]) == 3, len(BY_PRINCIPAL[READER]) == 1]),
            f"a {{peer: result}} cache holds {result['naive_entries']} entry and serves the "
            f"admin's {len(BY_PRINCIPAL[ADMIN])}-tool list to the reader, who is entitled to "
            f"{len(BY_PRINCIPAL[READER])}: {result['naive']}. That is the leak the scope "
            "field exists to prevent",
        ),
        practice.Check(
            "FINDING: re-keying on (peer, principal) is what private instructs",
            all([not result["scoped_leaks"], result["scoped_entries"] == 2,
                 result["scoped"] == result["direct"],
                 result["scoped_entries"] == result["principals"]]),
            f"keying on the pair gives {result['scoped_entries']} entries for "
            f"{result['principals']} principals and reproduces the uncached answer exactly, "
            f"{result['scoped']}. private does not forbid caching; it names the key",
        ),
        practice.Check(
            "FINDING: the client has no principal to key on",
            all([result["client_info"] == ["name", "version"],
                 "roots" in result["capabilities"] or result["capabilities"] != []]),
            f"CLIENT_INFO is {result['client_info']} and client_capabilities is "
            f"{result['capabilities']} -- neither varies per authorization context. The key "
            "private requires does not exist anywhere in the client's state; it would have "
            "to come from the transport, which is where the credential lives. That is why "
            "the cache was never written",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
