"""Exercise 4 — `private` is the default that nothing in the lesson uses.

    Change `cacheScope` from `public` to `private`. Explain which authorization
    contexts may reuse the response in each case.

Reading of the exercise: the change is made on both responses that carry a
scope, and the explanation is anchored to which of the three methods carries one
at all -- because "which contexts may reuse it" is a question about the payload,
and the method whose payload depends on the caller is the method the lesson
declines to cache.

**ANSWER: `public` means any authorization context; `private` means exactly
one.** A `public` response is a function of the server alone, so a shared cache
in front of many callers may serve it to all of them. A `private` response is a
function of the caller's authorization, so it may be reused only for further
requests carrying that same authorization -- a per-principal cache, never a
shared one. Getting it backwards leaks one caller's data to another.

**FINDING: both cacheable responses are `public`, and the one that is not
cacheable is the one that takes arguments.** `server/discover` caches for
**3,600,000 ms** and `tools/list` for **30,000**, both `public`; `tools/call`
carries **no `ttlMs` and no `cacheScope`** at all. The split is not arbitrary --
those two are pure functions of the server and the third is a function of the
request.

**FINDING: `private` is `complete_result`'s default and no call site reaches
it.** Passing `ttl_ms` without a scope yields `cacheScope: "private"`, so the
safe value is the fallback -- but all **2** cacheable call sites pass
`cache_scope="public"` explicitly. The default protects a case the lesson never
constructs.

**FINDING: and nothing in the response says what the caller was.** With no
session and no principal in `_meta`, a `private` response here would be
cacheable against *what*? The three `_meta` keys are version, capabilities and
client identity -- and `clientInfo` is a name and version string, not an
authorization context. Making `tools/call` cacheable would require a key the
protocol does not carry, which is why it carries no scope instead.

Structure: `scope_of` reads `ttlMs` and `cacheScope` off one dispatched
response, `rescoped` rebuilds the same payload as `private`, and `defaulted`
calls `complete_result` with a ttl and no scope.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "06-mcp-fundamentals"
METHODS = ("server/discover", "tools/list", "tools/call")


def dispatched(ref, method, request_id=1):
    params = {"name": "notes_list"} if method == "tools/call" else None
    return ref.dispatch(ref.make_request(request_id, method, params))["result"]


def scope_of(result):
    return {"ttlMs": result.get("ttlMs"), "cacheScope": result.get("cacheScope")}


def rescoped(ref, method):
    """The same payload re-emitted as private, to show the field is the only change."""
    original = dispatched(ref, method)
    payload = {key: value for key, value in original.items()
               if key not in ("_meta", "resultType", "ttlMs", "cacheScope")}
    return ref.complete_result(payload, ttl_ms=original["ttlMs"], cache_scope="private")


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    scopes = {method: scope_of(dispatched(ref, method)) for method in METHODS}
    source = inspect.getsource(ref.dispatch)
    private_list = rescoped(ref, "tools/list")
    original_list = dispatched(ref, "tools/list")
    return {
        "scopes": scopes,
        "cacheable": [m for m, s in scopes.items() if s["cacheScope"] is not None],
        "uncacheable": [m for m, s in scopes.items() if s["cacheScope"] is None],
        "discover_ttl": scopes["server/discover"]["ttlMs"],
        "list_ttl": scopes["tools/list"]["ttlMs"],
        "explicit_public": source.count('cache_scope="public"'),
        "explicit_private": source.count('cache_scope="private"'),
        "default_scope": ref.complete_result({"x": 1}, ttl_ms=5).get("cacheScope"),
        "rescoped_scope": private_list["cacheScope"],
        "rescoped_payload_same": private_list["tools"] == original_list["tools"],
        "meta_keys": sorted(ref.request_meta()),
        "client_info_fields": sorted(ref.CLIENT_INFO),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: public means any authorization context; private means exactly one",
            all([result["rescoped_scope"] == "private",
                 result["rescoped_payload_same"],
                 result["scopes"]["tools/list"]["cacheScope"] == "public"]),
            f"re-emitting tools/list as {result['rescoped_scope']} changes the scope field "
            f"and nothing else -- the payload is identical. A public response is a function "
            "of the server alone, so a shared cache may serve it to every caller; a private "
            "one is a function of the caller's authorization, so it may be reused only for "
            "requests carrying that same authorization. Getting it backwards leaks one "
            "caller's data to another",
        ),
        practice.Check(
            "FINDING: both cacheable responses are public, and the uncacheable one takes args",
            all([result["cacheable"] == ["server/discover", "tools/list"],
                 result["uncacheable"] == ["tools/call"],
                 result["discover_ttl"] == 3_600_000, result["list_ttl"] == 30_000,
                 result["scopes"]["tools/call"] == {"ttlMs": None, "cacheScope": None}]),
            f"server/discover caches for {result['discover_ttl']:,} ms and tools/list for "
            f"{result['list_ttl']:,}, both public; tools/call carries "
            f"{result['scopes']['tools/call']}. The split is not arbitrary -- those two are "
            "pure functions of the server and the third is a function of the request",
        ),
        practice.Check(
            "FINDING: private is complete_result's default and no call site reaches it",
            all([result["default_scope"] == "private", result["explicit_public"] == 2,
                 result["explicit_private"] == 0]),
            f"passing ttl_ms with no scope yields cacheScope {result['default_scope']!r}, so "
            f"the safe value is the fallback -- but all {result['explicit_public']} cacheable "
            f"call sites pass public explicitly and {result['explicit_private']} pass "
            "private. The default protects a case the lesson never constructs",
        ),
        practice.Check(
            "FINDING: and nothing in the response says what the caller was",
            all([result["meta_keys"] == ["io.modelcontextprotocol/clientCapabilities",
                                         "io.modelcontextprotocol/clientInfo",
                                         "io.modelcontextprotocol/protocolVersion"],
                 result["client_info_fields"] == ["name", "version"]]),
            f"with no session and no principal in _meta, a private response here would be "
            f"cacheable against what? The keys are {result['meta_keys']}, and clientInfo is "
            f"{result['client_info_fields']} -- a name and a version, not an authorization "
            "context. Making tools/call cacheable would need a key the protocol does not "
            "carry, which is why it carries no scope instead",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
