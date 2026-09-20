"""Exercise 1 — refusing the old token is not the point; revoking the family is.

    Add refresh-token rotation and reject reuse of the previous refresh token.

Reading of the exercise: rejecting the reused token is the easy half and
protects nobody -- by the time it is replayed the thief already has whatever
the last rotation produced. So the solution implements the half OAuth 2.1
actually asks for, family revocation, and then ablates it, because the
difference between the two is the whole security value.

**ANSWER: rotation, and a reused refresh token is refused.** Each refresh
mints a new access token and a new refresh token and retires the one it
consumed, so a chain of **3** refreshes leaves **1** live token; replaying
retired token **1** answers `refresh token already used`.

**FINDING: refusing the replay leaves the thief holding the live token.**
With rotation but no family revocation, a stolen token replayed after the
legitimate client has refreshed is refused -- and the thief's *earlier*
successful refresh is still valid, so **1** live token remains and it is
theirs. With family revocation the replay invalidates the whole chain and
**0** tokens remain: the reuse is the signal that two parties hold one
credential, and the only safe response is to end the session.

**FINDING: the lesson has no refresh token to rotate.** `Token` has **7**
fields -- none of them a refresh token -- and `exchange` returns a single
access token at `expires_at + 3600` with no renewal path. Today the only way
past an hour is a fresh authorization, PKCE round trip and all, so rotation is
new machinery rather than a change to existing machinery.

**FINDING: the family has to be named at issue time, not derived at reuse.**
Each rotation carries the family id of the token it replaced, so a reuse
anywhere in the chain reaches every descendant -- **3** of **3** retired
tokens map to the same family. Deriving the relationship from the access
tokens instead would not work: they carry `client_id` and `subject`, which a
legitimate second session shares.

Structure: `Rotator` mints and refreshes; `revoke_family` is the switch the
third check turns off.
"""

from __future__ import annotations

import dataclasses
import secrets
import time

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "16-mcp-security-oauth-2-1"
REUSED = "refresh token already used"


class Rotator:
    """Refresh tokens with rotation, and family revocation as a switch."""

    def __init__(self, ref, auth, *, revoke_family=True):
        self.ref, self.auth, self.revoke_family = ref, auth, revoke_family
        self.live, self.retired, self.family_of = {}, {}, {}

    def issue(self, token):
        family = secrets.token_hex(4)
        return self._mint(token, family)

    def _mint(self, token, family):
        refresh = f"rt_{secrets.token_hex(6)}"
        self.live[refresh] = token
        self.family_of[refresh] = family
        return refresh

    def refresh(self, refresh):
        if refresh in self.retired:
            if self.revoke_family:
                family = self.family_of[refresh]
                for held in [k for k, f in self.family_of.items() if f == family]:
                    self.live.pop(held, None)
            raise ValueError(REUSED)
        token = self.live.pop(refresh, None)
        if token is None:
            raise ValueError("unknown refresh token")
        self.retired[refresh] = token
        renewed = dataclasses.replace(token, value=f"tok_{secrets.token_hex(6)}",
                                      expires_at=time.time() + 3600)
        return renewed, self._mint(renewed, self.family_of[refresh])


def chain(rotator, token, rounds):
    """Refresh `rounds` times, keeping every refresh token handed out."""
    handed = [rotator.issue(token)]
    for _ in range(rounds):
        _, nxt = rotator.refresh(handed[-1])
        handed.append(nxt)
    return handed


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    auth, client, server = ref.AuthorizationServer(), ref.Client(), ref.ResourceServer()
    token = client.authorize(auth, server.resource, {"notes:read"})

    strict = Rotator(ref, auth)
    handed = chain(strict, token, 3)
    replayed = None
    try:
        strict.refresh(handed[0])
    except ValueError as exc:
        replayed = str(exc)

    lenient = Rotator(ref, auth, revoke_family=False)
    stolen = chain(lenient, token, 1)          # the thief refreshed once
    lenient_live_before = len(lenient.live)
    lenient_replay = None
    try:
        lenient.refresh(stolen[0])             # and replays the token it consumed
    except ValueError as exc:
        lenient_replay = str(exc)
    return {
        "handed": len(handed), "live": len(strict.live),
        "retired": len(strict.retired), "replayed": replayed,
        "families": len({strict.family_of[t] for t in handed}),
        "lenient_replay": lenient_replay,
        "lenient_live": len(lenient.live), "lenient_before": lenient_live_before,
        "token_fields": [f.name for f in dataclasses.fields(ref.Token)],
        "lifetime": round(token.expires_at - time.time()),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: rotation, and a reused refresh token is refused",
            all([result["handed"] == 4, result["retired"] == 3,
                 result["replayed"] == REUSED]),
            f"three refreshes hand out {result['handed']} tokens and retire "
            f"{result['retired']}, and replaying the first answers "
            f"{result['replayed']!r}. Each refresh mints a new access token and a new "
            "refresh token and retires the one it consumed",
        ),
        practice.Check(
            "FINDING: refusing the replay leaves the thief holding the live token",
            all([result["lenient_replay"] == REUSED, result["lenient_before"] == 1,
                 result["lenient_live"] == 1, result["live"] == 0]),
            f"without family revocation the replay is refused and "
            f"{result['lenient_live']} token stays live -- the one the thief's own refresh "
            f"minted. With revocation the same replay leaves {result['live']}. The reuse is "
            "the signal that two parties hold one credential, and ending the session is the "
            "only response that helps",
        ),
        practice.Check(
            "FINDING: the lesson has no refresh token to rotate",
            all([len(result["token_fields"]) == 7,
                 not any("refresh" in name for name in result["token_fields"]),
                 3500 < result["lifetime"] <= 3600]),
            f"Token has {len(result['token_fields'])} fields, {result['token_fields']}, none "
            f"a refresh token, and exchange returns one access token good for "
            f"{result['lifetime']} seconds with no renewal path. Past an hour the only route "
            "is a fresh authorization, PKCE round trip and all",
        ),
        practice.Check(
            "FINDING: the family has to be named at issue time, not derived at reuse",
            all([result["families"] == 1, result["retired"] == 3]),
            f"all {result['handed']} tokens in the chain carry {result['families']} family "
            "id, carried forward at each rotation, so a reuse anywhere reaches every "
            "descendant. Deriving the relationship from the access tokens would not work: "
            "they carry client_id and subject, which a legitimate second session shares",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
