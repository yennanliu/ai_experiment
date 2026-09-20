"""Exercise 1 — the token is bearer, because the gateway has no caller.

    Bind the authenticated principal and current authorization decision to the
    sealed MRTR state, then reject a retry under a different principal.

Reading of the exercise: before a principal can be bound there has to be one,
and `SecurityGateway.handle` takes no principal at all -- so the first thing
the exercise needs is a caller, and the finding is what its absence already
costs. Binding the *decision* is then treated as a separate question from
binding the principal, because one is a fact about who asked and the other is
a fact about what the server allowed, and only one of them can go stale.

**ANSWER: `principal` and `decision` in the sealed state, and a retry under a
different principal is refused.** Alice's confirmation completes the export;
the identical token replayed by mallory answers
`requestState principal mismatch` and exports nothing -- **1** export from
**2** retries of one token.

**FINDING: the shipped state binds the request and not the requester.** Its
fields are `arguments`, `expiresAt`, `issuedAt`, `nonce`, `purpose` and
`tool` -- **6**, and none names a caller. `handle` has no principal parameter
either, so the token is a bearer credential: whoever holds it completes the
export, and the gateway cannot tell that anyone else did.

**FINDING: binding the decision records it and does not re-check it.** A token
sealed while alice was authorized still carries `decision: "allow"` after the
grant is revoked, so the seal is a snapshot. Re-deriving the decision at the
retry refuses the export that the unchanged token would have completed --
**2** different outcomes from one token, decided by the server's state rather
than the client's.

**FINDING: the three bindings fail independently, which is why all three are
needed.** Changing the arguments is caught by the digest the lesson already
has; changing the caller is caught only by the principal; revoking the grant
is caught only by the re-check. **3** attacks, **3** distinct refusals, and no
one of the checks subsumes another.

Structure: `Gateway` adds a principal to `handle` and to the seal; `run` drives
one export to its confirmation and returns what the retry made of it.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "15-mcp-security-tool-poisoning"
ALICE, MALLORY = "alice@acme", "mallory@evil"
ARGUMENTS = {"query": "q4", "destination": "s3://reports"}
ACCEPT = {"confirm": {"action": "accept", "content": {"confirm": True}}}


class Gateway:
    """The lesson's gateway with a caller, and the caller sealed into the state."""

    def __init__(self, ref, grants):
        self.ref, self.grants = ref, grants
        self.inner = ref.SecurityGateway()

    def decide(self, principal):
        return "allow" if principal in self.grants else "deny"

    def begin(self, principal):
        pending = self.call(principal, {"name": "notes.export", "arguments": ARGUMENTS})
        state = self.ref.open_state(pending["requestState"], self.inner.secret)
        state.update(principal=principal, decision=self.decide(principal))
        return self.ref.seal_state(state, self.inner.secret)

    def retry(self, principal, token, *, recheck=True):
        state = self.ref.open_state(token, self.inner.secret)
        if state.get("principal") != principal:
            return "requestState principal mismatch"
        if recheck and self.decide(principal) != "allow":
            return "authorization revoked"
        if state.get("decision") != "allow":
            return "sealed decision denies"
        result = self.call(principal, {"name": "notes.export", "arguments": ARGUMENTS,
                                       "requestState": token, "inputResponses": ACCEPT})
        return result["structuredContent"]["exported"]

    def call(self, principal, params):
        body, headers = self.ref.make_request("tools/call", 1, params)
        headers["Mcp-Name"] = params["name"]
        response = self.inner.handle(body, headers)[1]
        return response.get("result", response.get("error"))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    grants = {ALICE}
    gateway = Gateway(ref, grants)
    token = gateway.begin(ALICE)
    shipped = ref.open_state(
        gateway.call(ALICE, {"name": "notes.export",
                             "arguments": ARGUMENTS})["requestState"], gateway.inner.secret)

    stranger = gateway.retry(MALLORY, token)
    owner = gateway.retry(ALICE, token)

    revoked_gateway = Gateway(ref, grants)
    revoked_token = revoked_gateway.begin(ALICE)
    revoked_gateway.grants = set()
    revoked = revoked_gateway.retry(ALICE, revoked_token)
    unchecked = revoked_gateway.retry(ALICE, revoked_token, recheck=False)

    tampered = Gateway(ref, grants)
    other_token = tampered.begin(ALICE)
    wrong_args = tampered.call(ALICE, {"name": "notes.export",
                                       "arguments": {**ARGUMENTS, "destination": "s3://evil"},
                                       "requestState": other_token, "inputResponses": ACCEPT})
    signature = ref.SecurityGateway.handle.__code__
    return {
        "shipped_fields": sorted(shipped),
        "handle_params": list(signature.co_varnames[:signature.co_argcount]),
        "stranger": stranger, "owner": owner,
        "revoked": revoked, "unchecked": unchecked,
        "wrong_args": wrong_args.get("message"),
        "exports": [stranger, owner].count(True),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the principal is sealed, and a retry under another one is refused",
            all([result["stranger"] == "requestState principal mismatch",
                 result["owner"] is True, result["exports"] == 1]),
            f"mallory replaying alice's token answers {result['stranger']!r} and exports "
            f"nothing, while alice's own retry exports: {result['owner']}. One token, two "
            f"retries, {result['exports']} export",
        ),
        practice.Check(
            "FINDING: the shipped state binds the request and not the requester",
            all([result["shipped_fields"] == ["arguments", "expiresAt", "issuedAt",
                                              "nonce", "purpose", "tool"],
                 result["handle_params"] == ["self", "body", "headers"]]),
            f"the sealed state's fields are {result['shipped_fields']} -- "
            f"{len(result['shipped_fields'])}, none naming a caller -- and handle takes "
            f"{result['handle_params']} with no principal at all. The token is a bearer "
            "credential: whoever holds it completes the export",
        ),
        practice.Check(
            "FINDING: binding the decision records it and does not re-check it",
            all([result["revoked"] == "authorization revoked",
                 result["unchecked"] is True]),
            f"a token sealed while alice was authorized still carries decision='allow' after "
            f"the grant is revoked, so re-deriving refuses ({result['revoked']!r}) where "
            f"trusting the seal completes ({result['unchecked']}). The seal is a snapshot of "
            "what was decided, not a substitute for deciding",
        ),
        practice.Check(
            "FINDING: the three bindings fail independently",
            all([result["wrong_args"] == "requestState does not match retried request",
                 result["stranger"] == "requestState principal mismatch",
                 result["revoked"] == "authorization revoked"]),
            f"changed arguments give {result['wrong_args']!r} from the lesson's own check, a "
            f"changed caller {result['stranger']!r} from the principal, and a revoked grant "
            f"{result['revoked']!r} from the re-check. Three attacks, three distinct "
            "refusals, and none of the checks subsumes another",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
