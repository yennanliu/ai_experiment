"""Exercise 4 — the gateway already downgrades what the backend calls public.

    Add a principal-specific server capability and explain why discovery must
    remain privately cached.

Reading of the exercise: "must remain" says the answer is already `private`,
so the explanation has to be about what would break if it were not -- and
today nothing would, because discovery is identical for every principal. The
capability is therefore added first, so the requirement has something to
protect, and the reason is then read off the difference between two callers.

**WHY DISCOVERY MUST STAY PRIVATE: the response varies by caller and the cache
key does not.** With a principal-specific capability, alice's discovery
advertises `experimental.bulkExport` and bob's does not, while both arrive at
the same method on the same gateway with nothing in the request a shared cache
would key on but the bearer it is not supposed to key on. A `public` scope
says the body depends on nothing about the caller; here it does, and the first
response cached would be served to the second caller.

**ANSWER: the capability is added and the scope is already right.** Alice sees
**2** capability groups and bob **1**; the scope is `private` at
`ttlMs: 30000` before and after, so the exercise's change makes the existing
scope necessary rather than changing it.

**FINDING: the gateway already downgrades what its backends call public.**
`BackendServer` answers `server/discover` with `cacheScope: "public"` and the
gateway answers `private` -- the same method, two scopes, because one is
per-backend and the other is per-principal. Forwarding the backend's hint
verbatim would have been wrong; the gateway synthesizes its own.

**FINDING: the shorter ttl is on the thing that moves faster.** Discovery is
private for **30,000** ms and `tools/list` private for **10,000** -- and the
listing is what a descriptor change or an RBAC edit moves, while the
capability set changes when the gateway is deployed. The two private scopes
are the same word for two different risks.

Structure: `discover` runs one principal's discovery through the lesson's own
`handle` and then applies the capability policy the exercise asks for.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "17-mcp-gateways-and-registries"
BULK = {"bulkExport": {}}
ENTITLED = {"alice"}


def discover(ref, gateway, bearer):
    """The lesson's discovery, plus the principal-specific capability."""
    body, headers = ref.make_request("server/discover", 1)
    status, response = gateway.handle(bearer, body, headers)
    result = dict(response["result"])
    principal = ref.USERS[bearer]["id"]
    if principal in ENTITLED:
        result["capabilities"] = {**result["capabilities"], "experimental": BULK}
    return status, result, principal


def listing(ref, gateway, bearer):
    body, headers = ref.make_request("tools/list", 2)
    return gateway.handle(bearer, body, headers)[1]["result"]


def backend_discover(ref, gateway):
    body, headers = ref.make_request("server/discover", 3)
    return gateway.backends["notes"].handle(body, headers)["result"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    gateway = ref.Gateway()
    _, alice, _ = discover(ref, gateway, "bearer-alice")
    _, bob, _ = discover(ref, gateway, "bearer-bob")
    body, headers = ref.make_request("server/discover", 1)
    shipped = gateway.handle("bearer-alice", body, headers)[1]["result"]
    backend = backend_discover(ref, gateway)
    tools = listing(ref, gateway, "bearer-alice")
    # the two callers send byte-identical requests; only the bearer differs
    alice_body, _ = ref.make_request("server/discover", 1)
    bob_body, _ = ref.make_request("server/discover", 1)
    return {
        "alice_caps": sorted(alice["capabilities"]), "bob_caps": sorted(bob["capabilities"]),
        "differs": alice["capabilities"] != bob["capabilities"],
        "scope": alice["cacheScope"], "ttl": alice["ttlMs"],
        "shipped_identical": shipped["capabilities"] == bob["capabilities"],
        "shipped_scope": shipped["cacheScope"],
        "backend_scope": backend["cacheScope"], "backend_ttl": backend["ttlMs"],
        "listing_scope": tools["cacheScope"], "listing_ttl": tools["ttlMs"],
        "request_differs": alice_body != bob_body,
    }


def verify(result):
    return [
        practice.Check(
            "WHY: the response varies by caller and the cache key does not",
            all([result["differs"], result["alice_caps"] == ["experimental", "tools"],
                 result["bob_caps"] == ["tools"], not result["request_differs"]]),
            f"alice's discovery advertises {result['alice_caps']} and bob's "
            f"{result['bob_caps']}, from the same method on the same gateway, with nothing in "
            "the request a shared cache would key on but the bearer it must not key on. A "
            "public scope says the body depends on nothing about the caller; here it does, "
            "and the first response cached would be served to the second",
        ),
        practice.Check(
            "ANSWER: the capability is added and the scope is already right",
            all([len(result["alice_caps"]) == 2, len(result["bob_caps"]) == 1,
                 result["scope"] == "private", result["ttl"] == 30_000,
                 result["shipped_scope"] == "private"]),
            f"alice sees {len(result['alice_caps'])} capability groups and bob "
            f"{len(result['bob_caps'])}, and the scope is {result['scope']!r} at ttlMs "
            f"{result['ttl']} before and after. The change makes the existing scope necessary "
            "rather than changing it",
        ),
        practice.Check(
            "FINDING: the gateway already downgrades what its backends call public",
            all([result["backend_scope"] == "public", result["shipped_scope"] == "private",
                 result["backend_ttl"] == 60_000, result["shipped_identical"]]),
            f"BackendServer answers server/discover with {result['backend_scope']!r} at "
            f"{result['backend_ttl']} and the gateway answers {result['shipped_scope']!r} -- "
            "the same method, two scopes, because one is per-backend and the other "
            "per-principal. Forwarding the backend's hint verbatim would have been wrong",
        ),
        practice.Check(
            "FINDING: the shorter ttl is on the thing that moves faster",
            all([result["ttl"] == 30_000, result["listing_ttl"] == 10_000,
                 result["listing_scope"] == "private"]),
            f"discovery is private for {result['ttl']} ms and tools/list private for "
            f"{result['listing_ttl']} -- and the listing is what a descriptor change or an "
            "RBAC edit moves, while the capability set changes when the gateway is deployed. "
            "The same word for two different risks",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
