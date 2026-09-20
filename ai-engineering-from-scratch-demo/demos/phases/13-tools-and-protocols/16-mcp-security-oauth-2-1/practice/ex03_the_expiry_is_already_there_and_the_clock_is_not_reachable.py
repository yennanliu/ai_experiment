"""Exercise 3 — the expiry is already there, and the clock is not reachable.

    Add an expiry to authorization codes and confirm a late exchange fails.

Reading of the exercise: the first thing to check is whether the expiry is
missing, and it is not -- `authorize` already writes `expires_at: now + 300`
and `exchange` already refuses a code past it. So the exercise reduces to
confirming it, and the confirmation turns out to be the interesting part:
there is no injectable clock anywhere in the module, so "late" can only be
reached by editing the pending record or waiting five minutes.

**ANSWER: a late exchange fails with `invalid authorization code`.** The
window is **300** seconds; a code presented at `expires_at - 1` succeeds and
the same code at `expires_at + 1` does not. The record is also popped on the
way out, so the failure is terminal.

**FINDING: the expiry already existed, and the test cannot reach it
honestly.** `exchange` takes `code, client_id, verifier, redirect_uri,
resource` -- **5** parameters, no clock -- and compares against
`time.time()` directly. Confirming the expiry means reaching into
`pending_codes` and rewriting `expires_at`, which is testing the branch rather
than the behaviour. An injectable `now`, as the sampling lesson has, is the
difference between a testable deadline and a documented one.

**FINDING: expired and never-issued are the same answer, and that is
deliberate.** A code past its deadline and a code that was never minted both
raise `invalid authorization code`. Distinguishing them would tell an attacker
which guesses were once real, and the code is the one value in this flow that
travels through a browser redirect.

**FINDING: single use is enforced separately from expiry, and it is the
stronger of the two.** A successful exchange pops the record, so the same code
replayed one second later -- well inside the window -- fails identically.
Expiry bounds how long a *leaked, unused* code is worth something; the pop is
what stops the replay of one that worked.

Structure: `issue` mints a code through the lesson's own `authorize`, and
`exchange_at` moves the record's deadline rather than the clock, because the
clock is not a parameter.
"""

from __future__ import annotations

import inspect
import time

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "16-mcp-security-oauth-2-1"
WINDOW = 300


def issue(ref, auth, client, resource):
    verifier, challenge = ref.pkce_pair()
    response = auth.authorize(client_id=client.client_ids_by_issuer[auth.issuer],
                              redirect_uri=client.redirect_uri, subject=client.subject,
                              scopes={"notes:read"}, challenge=challenge, resource=resource)
    return response["code"], verifier


def exchange_at(ref, auth, client, resource, code, verifier, *, offset=None):
    """Exchange, optionally after moving the record's deadline by `offset`."""
    if offset is not None and code in auth.pending_codes:
        auth.pending_codes[code]["expires_at"] = time.time() + offset
    try:
        token = auth.exchange(code=code,
                              client_id=client.client_ids_by_issuer[auth.issuer],
                              verifier=verifier, redirect_uri=client.redirect_uri,
                              resource=resource)
        return token.audience
    except ValueError as exc:
        return str(exc)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    auth, client, server = ref.AuthorizationServer(), ref.Client(), ref.ResourceServer()
    client.enroll(auth)

    fresh_code, fresh_verifier = issue(ref, auth, client, server.resource)
    window = auth.pending_codes[fresh_code]["expires_at"] - time.time()
    early = exchange_at(ref, auth, client, server.resource, fresh_code, fresh_verifier,
                        offset=1)

    late_code, late_verifier = issue(ref, auth, client, server.resource)
    late = exchange_at(ref, auth, client, server.resource, late_code, late_verifier,
                       offset=-1)
    after_late = late_code in auth.pending_codes

    bogus = exchange_at(ref, auth, client, server.resource, "code_deadbeef", "v")

    replay_code, replay_verifier = issue(ref, auth, client, server.resource)
    deadline = auth.pending_codes[replay_code]["expires_at"]
    first = exchange_at(ref, auth, client, server.resource, replay_code, replay_verifier)
    second = exchange_at(ref, auth, client, server.resource, replay_code, replay_verifier)
    signature = inspect.signature(ref.AuthorizationServer.exchange).parameters
    return {
        "window": round(window), "early": early, "late": late, "bogus": bogus,
        "popped": not after_late,
        "params": [name for name in signature if name != "self"],
        "has_clock": any("now" in name or "clock" in name for name in signature),
        "first": first, "second": second,
        "replay_inside_window": time.time() < deadline,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a late exchange fails, and the window is 300 seconds",
            all([result["window"] == WINDOW,
                 result["early"] == "https://notes.example.com/mcp",
                 result["late"] == "invalid authorization code", result["popped"]]),
            f"the window is {result['window']} seconds; a code presented one second early "
            f"exchanges for a token on {result['early']} and the same code one second late "
            f"answers {result['late']!r}. The record is popped on the way out, so the "
            "failure is terminal",
        ),
        practice.Check(
            "FINDING: the expiry already existed, and the test cannot reach it honestly",
            all([result["params"] == ["code", "client_id", "verifier", "redirect_uri",
                                      "resource"],
                 not result["has_clock"]]),
            f"exchange takes {result['params']} -- {len(result['params'])} parameters, no "
            "clock -- and compares against time.time() directly. Confirming the expiry means "
            "rewriting expires_at inside pending_codes, which tests the branch rather than "
            "the behaviour. An injectable now is the difference between a testable deadline "
            "and a documented one",
        ),
        practice.Check(
            "FINDING: expired and never-issued are the same answer, deliberately",
            result["bogus"] == result["late"] == "invalid authorization code",
            f"a code past its deadline and one that was never minted both answer "
            f"{result['bogus']!r}. Distinguishing them would tell an attacker which guesses "
            "were once real -- and the code is the one value in this flow that travels "
            "through a browser redirect",
        ),
        practice.Check(
            "FINDING: single use is enforced separately, and it is the stronger of the two",
            all([result["first"] == "https://notes.example.com/mcp",
                 result["second"] == "invalid authorization code",
                 result["replay_inside_window"]]),
            f"a successful exchange gives {result['first']} and the same code replayed "
            f"immediately -- well inside the window -- gives {result['second']!r}. Expiry "
            "bounds how long a leaked, unused code is worth something; the pop is what stops "
            "the replay of one that worked",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
