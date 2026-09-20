"""Exercise 1 — the overlap is one rotation, and only for the current key.

    Run `code/main.py`. Trace the flow. Note how the IdP rotates a key in step
    6, the scheduled `refresh_jwks` re-pulls the published set, and both the
    old token (overlap window) and a fresh token validate without restart.

Reading of the exercise: "the overlap window" is stated as a property of the
design, so the trace is run twice -- once with a token minted from the key
that was current, and once from the *older* of the two live keys. The two do
not survive the same number of rotations, which means the window is a
property of when in the cycle a token was minted and not of the design alone.

**ANSWER: both tokens validate after one rotation and a refresh, with no
restart.** The cache holds **2** kids; a token signed by the pre-rotation
current key still verifies and a freshly minted one does too -- **2** of **2**
valid, because `rotate_key` keeps `keys[-2:]` and the scheduled refresh
re-pulls both.

**FINDING: the window is one rotation for the newest key and zero for the
other.** `rotate_key` retires by position, so the older of the two live keys
is dropped by the very next rotation. A token minted from `keys[0]` is
already unverifiable after **1** rotation -- `unknown kid` -- while one minted
from `current_key()` survives that rotation and dies at the second. The grace
period a token actually gets is between zero and one rotations, decided by
where in the cycle it was issued.

**FINDING: the resource server cannot rotate, only re-pull.** `refresh_jwks`
reads `auth_server.jwks()` and writes the cache, so calling it twice leaves
the authorization server's key count at **2**. Rotation is the IdP's; the MCP
server's only move is to fetch, which is what makes the cache-miss fall-back
safe.

**FINDING: retirement is by count, and nothing compares it to token
lifetime.** `keys[-2:]` is the whole policy -- no expiry, no reference to the
**300**-second token TTL. Whether a live token outlives its key is a
scheduling property of how often `rotate_key` is called, enforced nowhere in
the code.

Structure: `mint` signs one token with one key, and `cycle` rotates and
refreshes together, the way step 6 does.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import time

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "18-mcp-auth-production"
TTL = 300


def compared(func):
    """What every comparison in `func` tests, by its left-hand source."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return [ast.unparse(node.left) for node in ast.walk(tree)
            if isinstance(node, ast.Compare)]


def build(ref):
    auth = ref.AuthorizationServer()
    auth.rotate_key()
    auth.rotate_key()
    server = ref.ResourceServer(resource=ref.MCP_RESOURCE, auth_server=auth,
                                allowed_issuers=[auth.issuer])
    server.refresh_jwks()
    return auth, server


def mint(ref, auth, key):
    return ref.jwt_sign({"iss": auth.issuer, "aud": ref.MCP_RESOURCE, "sub": "alice",
                         "scope": "mcp:tools.invoke", "exp": time.time() + TTL},
                        key.kid, key.secret)


def cycle(auth, server):
    """Step 6: the IdP rotates, the scheduled job re-pulls."""
    auth.rotate_key()
    server.refresh_jwks()


def outcome(server, token):
    result = server.validate(token)
    return True if result["valid"] else result["www_authenticate"].split('"')[3]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    auth, server = build(ref)
    current_token = mint(ref, auth, auth.current_key())
    older_token = mint(ref, auth, auth.keys[0])
    before = server.cached_kids()

    cycle(auth, server)
    after_one = (outcome(server, current_token), outcome(server, older_token))
    fresh = outcome(server, mint(ref, auth, auth.current_key()))
    keys_after_one = len(auth.keys)

    cycle(auth, server)
    after_two = outcome(server, current_token)

    twice = build(ref)[1]
    twice.refresh_jwks()
    twice.refresh_jwks()
    return {
        "before": len(before), "cached": len(server.cached_kids()),
        "after_one": after_one, "fresh": fresh, "after_two": after_two,
        "keys": keys_after_one, "keys_after_refreshes": len(twice.auth_server.keys),
        "retire": "keys[-2:]" in inspect.getsource(ref.AuthorizationServer.rotate_key),
        "compares": compared(ref.AuthorizationServer.rotate_key),
        "ttl": TTL,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: both tokens validate after one rotation and a refresh, no restart",
            all([result["before"] == 2, result["cached"] == 2,
                 result["after_one"][0] is True, result["fresh"] is True,
                 result["keys"] == 2]),
            f"the cache holds {result['cached']} kids after the rotation and the scheduled "
            f"refresh; the token minted from the pre-rotation current key still verifies "
            f"({result['after_one'][0]}) and a fresh one does too ({result['fresh']}), with "
            f"the authorization server steady at {result['keys']} keys",
        ),
        practice.Check(
            "FINDING: the window is one rotation for the newest key and zero for the other",
            all([result["after_one"][1] == "unknown kid",
                 result["after_one"][0] is True, result["after_two"] == "unknown kid"]),
            f"a token minted from keys[0] answers {result['after_one'][1]!r} after one "
            f"rotation, while one minted from current_key() survives it and dies at the "
            f"second ({result['after_two']!r}). The grace period a token gets is between "
            "zero and one rotations, decided by where in the cycle it was issued",
        ),
        practice.Check(
            "FINDING: the resource server cannot rotate, only re-pull",
            result["keys_after_refreshes"] == 2,
            f"calling refresh_jwks twice leaves the authorization server at "
            f"{result['keys_after_refreshes']} keys, because it reads auth_server.jwks() and "
            "writes the cache. Rotation is the IdP's; the MCP server's only move is to "
            "fetch, which is what makes the cache-miss fall-back safe",
        ),
        practice.Check(
            "FINDING: retirement is by count, and nothing compares it to token lifetime",
            all([result["retire"], result["compares"] == ["len(self.keys)"]]),
            f"keys[-2:] is the whole policy, and the only comparison rotate_key makes is on "
            f"{result['compares']} -- never on a clock and never on the {result['ttl']}-second "
            "token TTL. Whether a live token outlives its key is a scheduling property of "
            "how often rotate_key is called, enforced nowhere in the code",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
