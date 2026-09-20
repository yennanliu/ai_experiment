"""Exercise 2 — the allowlist admits an issuer the fetch can never serve.

    Add a new IdP to the protected-resource metadata's
    `authorization_servers` list. Issue a token signed by the new IdP and
    confirm the validator accepts it. Issue a token signed by an unlisted IdP
    and confirm the validator rejects with `WWW-Authenticate: Bearer
    error="invalid_token", error_description="iss not allowed"`.

Reading of the exercise: adding the issuer to the list is one line, so the
first thing worth checking is whether that line is sufficient -- and it is
not. `refresh_jwks` fetches `self.auth_server` and nothing else, so the
second IdP's keys have no route into the cache; the token gets past the
allowlist and dies one check later. The exercise's "confirm the validator
accepts it" is therefore only reachable after the cache is taught about the
second issuer as well.

**ANSWER: the listed IdP is accepted and the unlisted one is refused with the
exact challenge.** With the second issuer in `allowed_issuers` *and* its keys
in `jwks_cache`, its token validates; an unlisted issuer answers
`Bearer error="invalid_token", error_description="iss not allowed",
resource_metadata="..."` -- the string the exercise names, plus the
`resource_metadata` every challenge in this server carries.

**FINDING: the allowlist is checked before any JWKS work, on purpose.** An
unlisted issuer costs **0** refreshes, where a listed issuer with an unknown
kid costs **1**. The comment says so and the count confirms it: rejecting on
`iss` first is what stops an attacker turning a made-up issuer into fetch
traffic.

**FINDING: `allowed_issuers` and `jwks_cache` are two lists that have to
agree, and only one has an updater.** The cache is keyed by issuer -- the
structure anticipates several IdPs -- but `refresh_jwks` hard-codes
`self.auth_server.issuer`, so adding to the allowlist alone yields
`unknown kid` rather than acceptance. Allowlisting is necessary and not
sufficient; the shipped code has no path that makes it sufficient.

**FINDING: two rejections that mean opposite things are one status.** An
unlisted issuer (never trust it) and a listed issuer whose key is unknown
(trust it, refetch later) both answer **401** with `invalid_token`, differing
only in `error_description`. A client retrying on 401 cannot tell which one
is worth retrying.

Structure: `mint` signs for any issuer, and `Multi` is the resource server
with the second issuer's keys taught to the cache -- the step the exercise's
one line leaves out.
"""

from __future__ import annotations

import inspect
import time

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "18-mcp-auth-production"
SECOND = "https://auth-b.example.com"
UNLISTED = "https://evil.example.com"


def mint(ref, issuer, key, resource):
    return ref.jwt_sign({"iss": issuer, "aud": resource, "sub": "alice",
                         "scope": "mcp:tools.invoke", "exp": time.time() + 300},
                        key.kid, key.secret)


def teach(server, issuer, auth):
    """Put a second issuer's published keys where refresh_jwks cannot."""
    server.jwks_cache[issuer] = {"keys": auth.jwks()["keys"], "fetched_at": time.time()}


def counted(server):
    """Wrap refresh_jwks so its calls can be counted."""
    calls, original = [], server.refresh_jwks
    server.refresh_jwks = lambda: (calls.append(1), original())[1]
    return calls


def description(result):
    return result["www_authenticate"].split('error_description="', 1)[1].split('"', 1)[0]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    first = ref.AuthorizationServer()
    first.rotate_key()
    second = ref.AuthorizationServer(issuer=SECOND)
    second.rotate_key()
    server = ref.ResourceServer(resource=ref.MCP_RESOURCE, auth_server=first,
                                allowed_issuers=[first.issuer, SECOND])
    server.refresh_jwks()

    listed_only = server.validate(mint(ref, SECOND, second.current_key(), ref.MCP_RESOURCE))
    teach(server, SECOND, second)
    taught = server.validate(mint(ref, SECOND, second.current_key(), ref.MCP_RESOURCE))
    unlisted = server.validate(mint(ref, UNLISTED, first.current_key(), ref.MCP_RESOURCE))

    calls = counted(server)
    server.validate(mint(ref, UNLISTED, first.current_key(), ref.MCP_RESOURCE))
    unlisted_fetches = len(calls)
    bogus = ref.jwt_sign({"iss": first.issuer, "aud": ref.MCP_RESOURCE, "sub": "a",
                          "scope": "mcp:tools.invoke", "exp": time.time() + 300},
                         "k_bogus", b"x" * 32)
    server.validate(bogus)
    return {
        "listed_only": description(listed_only), "taught": taught["valid"],
        "unlisted": unlisted["www_authenticate"],
        "unlisted_description": description(unlisted),
        "first_valid": server.validate(mint(ref, first.issuer, first.current_key(),
                                            ref.MCP_RESOURCE))["valid"],
        "unlisted_fetches": unlisted_fetches,
        "unknown_kid_fetches": len(calls) - unlisted_fetches,
        "hardcodes": "self.auth_server.issuer" in inspect.getsource(
            ref.ResourceServer.refresh_jwks),
        "cache_keys": sorted(server.jwks_cache),
        "both_401": (unlisted["status"], 401),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the listed IdP is accepted and the unlisted one gets the exact challenge",
            all([result["taught"], result["first_valid"],
                 result["unlisted_description"] == "iss not allowed",
                 result["unlisted"].startswith(
                     'Bearer error="invalid_token", error_description="iss not allowed"')]),
            f"with the second issuer allowlisted and its keys in the cache its token "
            f"validates ({result['taught']}), alongside the first issuer's "
            f"({result['first_valid']}), while an unlisted issuer answers "
            f"{result['unlisted']} -- the string the exercise names, plus the "
            "resource_metadata every challenge here carries",
        ),
        practice.Check(
            "FINDING: the allowlist is checked before any JWKS work, on purpose",
            all([result["unlisted_fetches"] == 0, result["unknown_kid_fetches"] == 1]),
            f"an unlisted issuer costs {result['unlisted_fetches']} refreshes where a listed "
            f"issuer with an unknown kid costs {result['unknown_kid_fetches']}. Rejecting on "
            "iss first is what stops an attacker turning a made-up issuer into fetch traffic",
        ),
        practice.Check(
            "FINDING: allowed_issuers and jwks_cache must agree, and only one has an updater",
            all([result["listed_only"] == "unknown kid", result["hardcodes"],
                 result["cache_keys"] == sorted(["https://auth.example.com", SECOND])]),
            f"allowlisting alone yields {result['listed_only']!r}, because refresh_jwks "
            f"hard-codes self.auth_server.issuer while the cache is keyed by issuer and now "
            f"holds {result['cache_keys']}. The structure anticipates several IdPs and the "
            "fetch cannot serve them: allowlisting is necessary and not sufficient",
        ),
        practice.Check(
            "FINDING: two rejections that mean opposite things are one status",
            all([result["both_401"] == (401, 401),
                 result["unlisted_description"] != result["listed_only"]]),
            f"an unlisted issuer and a listed issuer with an unknown key both answer "
            f"{result['both_401'][0]} invalid_token, differing only in error_description "
            f"({result['unlisted_description']!r} against {result['listed_only']!r}). One is "
            "never worth retrying and the other is, and a client branching on status cannot "
            "tell them apart",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
