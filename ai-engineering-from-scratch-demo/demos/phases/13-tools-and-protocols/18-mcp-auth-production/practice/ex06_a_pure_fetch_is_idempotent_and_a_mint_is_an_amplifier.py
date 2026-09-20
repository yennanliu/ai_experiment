"""Exercise 6 — a pure fetch is idempotent and a mint is an amplifier.

    Prove the DoS fix. Send the validator a token with a random `kid` and
    confirm `refresh_jwks` runs at most once and the authorization server's
    key count does not grow. Then deliberately re-wire the fall-back to a
    rotate-and-mint and watch the key count climb per bogus token -- restore
    the re-fetch afterward.

Reading of the exercise: "at most once" is a claim about one request, so the
counter is read per validation rather than in total, and the bogus tokens are
sent in a batch so the two wirings can be compared on the same input. The
re-wire is installed by replacing the bound method and removed in a `finally`,
because the reference module is shared with every other solution in the run.

**ANSWER: one refresh per bogus token, and the key count does not move.**
**20** tokens with random kids cost **20** refreshes -- **1** each, never
more -- and leave the authorization server at **2** keys, because
`refresh_jwks` re-pulls a published set rather than creating one. Every token
answers `unknown kid`.

**FINDING: re-wired to rotate-and-mint the count does *not* climb, and the
damage is worse than if it had.** `rotate_key` keeps `keys[-2:]`, so **20**
bogus tokens force **20** rotations and leave the list at **2** -- the
exercise's "watch the key count climb" does not happen here. What climbs is
the churn, and it is destructive: a legitimate token that verified before the
storm answers `unknown kid` after it, because two rotations retired the key
that signed it. An attacker cannot grow the key list and can revoke everyone
else's live tokens, which a count would never have shown.

**FINDING: the fall-back is reachable twice and fires once.** `validate`
calls `refresh_jwks` for a cold cache *and* for an unknown kid, so a first-
ever request could pay two fetches; after the cache is warm a bogus kid pays
exactly **1**. The bound is per request and not per token, which is the
property the word "once" is doing.

**FINDING: the allowlist is what makes the bound cheap.** A bogus kid under a
*listed* issuer costs **1** refresh; the same token under an unlisted issuer
costs **0**, because `iss` is checked first. So the expensive path needs a
trusted issuer, which an attacker cannot forge -- the DoS surface is
limited to holders of a real issuer's name.

Structure: `bogus` mints a token with a random kid, and `rewire` swaps the
fall-back for a rotate-and-mint on one server instance only.
"""

from __future__ import annotations

import secrets
import time

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "18-mcp-auth-production"
ATTEMPTS = 20


def build(ref):
    auth = ref.AuthorizationServer()
    auth.rotate_key()
    auth.rotate_key()
    server = ref.ResourceServer(resource=ref.MCP_RESOURCE, auth_server=auth,
                                allowed_issuers=[auth.issuer])
    server.refresh_jwks()
    return auth, server


def bogus(ref, issuer):
    return ref.jwt_sign({"iss": issuer, "aud": ref.MCP_RESOURCE, "sub": "mallory",
                         "scope": "mcp:tools.invoke", "exp": time.time() + 300},
                        f"k_{secrets.token_hex(4)}", b"x" * 32)


def counted(server):
    calls, original = [], server.refresh_jwks

    def wrapped():
        calls.append(1)
        return original()

    server.refresh_jwks = wrapped
    return calls


def rewire(server, auth, rotations):
    """The bug the lesson warns about: mint on a cache miss instead of fetching."""
    def rotate_and_mint():
        rotations.append(1)
        auth.rotate_key()
        keys = auth.jwks()["keys"]
        server.jwks_cache[auth.issuer] = {"keys": keys, "fetched_at": time.time()}
        return {"refreshed": True, "kids": [k["kid"] for k in keys]}

    server.refresh_jwks = rotate_and_mint


def storm(ref, server, auth, issuer, attempts=ATTEMPTS):
    """Send `attempts` bogus tokens, recording refreshes and key count per token."""
    calls, per_token, reasons = counted(server), [], set()
    for _ in range(attempts):
        before = len(calls)
        result = server.validate(bogus(ref, issuer))
        per_token.append(len(calls) - before)
        reasons.add(result["www_authenticate"].split('"')[3])
    return per_token, sorted(reasons), len(auth.keys)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    auth, server = build(ref)
    keys_before = len(auth.keys)
    fetch_per_token, fetch_reasons, keys_after = storm(ref, server, auth, auth.issuer)

    bad_auth, bad_server = build(ref)
    bad_before, rotations = len(bad_auth.keys), []
    survivor = ref.jwt_sign({"iss": bad_auth.issuer, "aud": ref.MCP_RESOURCE, "sub": "alice",
                             "scope": "mcp:tools.invoke", "exp": time.time() + 300},
                            bad_auth.current_key().kid, bad_auth.current_key().secret)
    survived_before = bad_server.validate(survivor)["valid"]
    rewire(bad_server, bad_auth, rotations)
    mint_per_token, mint_reasons, keys_minted = storm(ref, bad_server, bad_auth,
                                                      bad_auth.issuer)
    storm_rotations = len(rotations)  # before the survivor re-check adds one more
    survived_after = bad_server.validate(survivor)

    cold_auth, cold_server = build(ref)
    cold_server.jwks_cache.clear()
    cold_calls = counted(cold_server)
    cold_server.validate(bogus(ref, cold_auth.issuer))

    unlisted_auth, unlisted_server = build(ref)
    unlisted_calls = counted(unlisted_server)
    unlisted_server.validate(bogus(ref, "https://evil.example.com"))
    return {
        "attempts": ATTEMPTS, "fetch_per_token": sorted(set(fetch_per_token)),
        "fetch_total": sum(fetch_per_token), "fetch_reasons": fetch_reasons,
        "keys_before": keys_before, "keys_after": keys_after,
        "mint_per_token": sorted(set(mint_per_token)), "mint_reasons": mint_reasons,
        "keys_minted": keys_minted, "mint_before": bad_before,
        "rotations": storm_rotations, "survived_before": survived_before,
        "survived_after": survived_after["www_authenticate"].split('"')[3],
        "cold_fetches": len(cold_calls), "unlisted_fetches": len(unlisted_calls),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: one refresh per bogus token, and the key count does not move",
            all([result["fetch_per_token"] == [1],
                 result["fetch_total"] == result["attempts"],
                 result["keys_before"] == 2, result["keys_after"] == 2,
                 result["fetch_reasons"] == ["unknown kid"]]),
            f"{result['attempts']} tokens with random kids cost "
            f"{result['fetch_total']} refreshes -- {result['fetch_per_token']} each, never "
            f"more -- and leave the authorization server at {result['keys_after']} keys, "
            f"because refresh_jwks re-pulls a published set. Every token answers "
            f"{result['fetch_reasons']}",
        ),
        practice.Check(
            "FINDING: re-wired, the count does not climb and the damage is worse than if it had",
            all([result["rotations"] == result["attempts"],
                 result["keys_minted"] == result["mint_before"],
                 result["survived_before"], result["survived_after"] == "unknown kid",
                 result["mint_reasons"] == result["fetch_reasons"]]),
            f"{result['attempts']} bogus tokens force {result['rotations']} rotations and "
            f"leave the list at {result['keys_minted']}, because rotate_key keeps keys[-2:] "
            f"-- so the count never climbs. What climbs is churn: a legitimate token that "
            f"verified before the storm ({result['survived_before']}) answers "
            f"{result['survived_after']!r} after it. The attacker cannot grow the list and "
            "can revoke everyone else's live tokens, which a count would never have shown",
        ),
        practice.Check(
            "FINDING: the fall-back is reachable twice and fires once",
            all([result["cold_fetches"] == 2, result["fetch_per_token"] == [1]]),
            f"validate calls refresh_jwks for a cold cache and again for an unknown kid, so "
            f"a first-ever request pays {result['cold_fetches']} fetches; once the cache is "
            f"warm a bogus kid pays {result['fetch_per_token'][0]}. The bound is per request "
            "rather than per token, which is the work the word 'once' is doing",
        ),
        practice.Check(
            "FINDING: the allowlist is what makes the bound cheap",
            all([result["unlisted_fetches"] == 0, result["fetch_per_token"] == [1]]),
            f"a bogus kid under a listed issuer costs {result['fetch_per_token'][0]} refresh "
            f"and the same token under an unlisted one costs {result['unlisted_fetches']}, "
            "because iss is checked first. The expensive path needs a trusted issuer's name, "
            "which an attacker cannot forge",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
