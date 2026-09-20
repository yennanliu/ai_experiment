"""Exercise 5 — the listing is already private and does not yet need to be.

    Add a policy that refuses public caching when `tools/list` differs by
    principal.

Reading of the exercise: a policy that refuses something needs a case where
the refusal bites, and the shipped gateway has none -- `_visible_tools` takes
no principal, so every caller gets the same listing. The exercise is therefore
read as two jobs: write the policy, and build the per-principal visibility
that makes it fire, so that the policy is observed doing both things rather
than only one.

**ANSWER: a policy that compares two principals' listings and downgrades.**
Given identical listings it permits `public`; given listings that differ it
refuses and forces `private`. Run against the shipped gateway it permits;
run against a gateway with per-principal grants it refuses.

**FINDING: today's `private` is correct and unearned.** `_visible_tools`
takes `(self)` alone, so alice and mallory receive byte-identical listings --
yet the result is already `cacheScope: "private"`. The scope is right for the
wrong reason, and the policy is what makes it right for the right one.

**FINDING: `server/discover` is public and stays public under the same
policy.** Its result depends on the protocol version and the capability map
and on nothing about the caller, so the policy permits it while refusing the
listing. Scope is decided per response, not per server -- one gateway, two
verdicts.

**FINDING: a catalog-wide block is not principal-dependence, and the policy
must not confuse them.** A rug pull hides `notes.search` from **everyone**, so
the two listings still match and the policy still permits `public`. What
forbids public caching is variation *across callers*, not variation over
time -- which `ttlMs` already covers.

Structure: `scoped` builds a gateway whose visible tools depend on a grant
table, and `policy` is the rule: compare the principals' listings and pick the
scope from whether they agree.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "15-mcp-security-tool-poisoning"
ALICE, MALLORY = "alice@acme", "mallory@evil"
GRANTS = {ALICE: {"notes.search", "notes.export", "issues.search"},
          MALLORY: {"issues.search"}}


def scoped(ref, grants=None):
    """The gateway's listing, optionally narrowed per principal."""
    gateway = ref.SecurityGateway()

    def visible(principal):
        names = [tool["name"] for tool in gateway._visible_tools()]
        if grants is None:
            return names
        return [name for name in names if name in grants[principal]]

    return gateway, visible


def policy(visible, principals):
    """Refuse public caching when the listing differs by principal."""
    listings = [visible(principal) for principal in principals]
    differs = any(listing != listings[0] for listing in listings)
    return {"cacheScope": "private" if differs else "public", "differs": differs}


def discover_scope(ref, gateway, principals):
    """The same policy over a response that cannot vary by caller."""
    body, headers = ref.make_request("server/discover", 1)
    results = [gateway.handle(body, headers)[1]["result"] for _ in principals]
    differs = any(result != results[0] for result in results)
    return {"cacheScope": "private" if differs else "public",
            "declared": results[0]["cacheScope"]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped, shipped_visible = scoped(ref)
    narrowed, narrowed_visible = scoped(ref, GRANTS)
    principals = [ALICE, MALLORY]

    pulled, pulled_visible = scoped(ref)
    pulled.catalog["notes"][0]["inputSchema"]["properties"]["query"]["maxLength"] = 10
    signature = ref.SecurityGateway._visible_tools.__code__
    body, headers = ref.make_request("tools/list", 1)
    declared = shipped.handle(body, headers)[1]["result"]
    return {
        "shipped": policy(shipped_visible, principals),
        "narrowed": policy(narrowed_visible, principals),
        "shipped_listings": [shipped_visible(p) for p in principals],
        "narrowed_listings": [narrowed_visible(p) for p in principals],
        "visible_params": list(signature.co_varnames[:signature.co_argcount]),
        "declared_scope": declared["cacheScope"], "declared_ttl": declared["ttlMs"],
        "discover": discover_scope(ref, shipped, principals),
        "pulled": policy(pulled_visible, principals),
        "pulled_listings": [pulled_visible(p) for p in principals],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the policy permits public when the listings match and refuses when they differ",
            all([result["shipped"]["cacheScope"] == "public",
                 not result["shipped"]["differs"],
                 result["narrowed"]["cacheScope"] == "private",
                 result["narrowed"]["differs"],
                 result["narrowed_listings"][0] != result["narrowed_listings"][1]]),
            f"against the shipped gateway the two principals see "
            f"{result['shipped_listings'][0]} each and the policy permits "
            f"{result['shipped']['cacheScope']!r}; with per-principal grants they see "
            f"{result['narrowed_listings']} and it refuses, forcing "
            f"{result['narrowed']['cacheScope']!r}",
        ),
        practice.Check(
            "FINDING: today's private is correct and unearned",
            all([result["visible_params"] == ["self"],
                 result["shipped_listings"][0] == result["shipped_listings"][1],
                 result["declared_scope"] == "private"]),
            f"_visible_tools takes {result['visible_params']} alone, so both principals get "
            f"byte-identical listings -- yet the result already declares "
            f"{result['declared_scope']!r} at ttlMs {result['declared_ttl']}. The scope is "
            "right for the wrong reason, and the policy is what makes it right for the right "
            "one",
        ),
        practice.Check(
            "FINDING: server/discover is public and stays public under the same policy",
            all([result["discover"]["cacheScope"] == "public",
                 result["discover"]["declared"] == "public"]),
            f"discovery's result depends on the version and the capability map and on "
            f"nothing about the caller, so the same policy returns "
            f"{result['discover']['cacheScope']!r} for it while refusing the listing. Scope "
            "is decided per response, not per server -- one gateway, two verdicts",
        ),
        practice.Check(
            "FINDING: a catalog-wide block is not principal-dependence",
            all([result["pulled"]["cacheScope"] == "public",
                 not result["pulled"]["differs"],
                 result["pulled_listings"][0] == result["pulled_listings"][1],
                 "notes.search" not in result["pulled_listings"][0]]),
            f"a rug pull hides notes.search from everyone, leaving "
            f"{result['pulled_listings'][0]} for both principals, so the policy still "
            f"permits {result['pulled']['cacheScope']!r}. What forbids public caching is "
            "variation across callers, not variation over time -- which ttlMs already covers",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
