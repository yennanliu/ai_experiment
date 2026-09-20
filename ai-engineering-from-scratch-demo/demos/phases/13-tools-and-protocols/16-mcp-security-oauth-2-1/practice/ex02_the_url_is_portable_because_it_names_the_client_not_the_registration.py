"""Exercise 2 — the URL is portable because it names the client, not the registration.

    Add an issuer allowlist. On issuer change, reuse only a portable CIMD URL;
    refuse all prior issuer-minted credentials and tokens.

Reading of the exercise: "portable" is the claim to test, so the move to a new
issuer is performed twice -- once by a CIMD client and once by a DCR client --
and the difference between what survives is the answer. "Refuse all prior
issuer-minted credentials" is then checked from both ends, because the client
dropping a token and the resource server refusing it are separate mechanisms
and only one of them is the client's to get right.

**ANSWER: the CIMD id crosses unchanged and the DCR id does not.** Enrolling
at a second issuer returns the identical
`https://client.example.com/oauth/metadata.json`, because the client id *is*
the document's URL; the DCR client gets a fresh `dcr_...` opaque string that
means nothing to anyone else. **1** identifier reused, **1** reissued.

**FINDING: the allowlist is what makes the move deliberate.** With
`{issuer_a}` allowed, enrolling at issuer B is refused before any credential
is minted; adding B admits it. Without the list the client would follow
whichever issuer the resource named -- and the resource's metadata is the
thing an attacker who controls the resource gets to write.

**FINDING: the client's stores are already keyed by issuer, so dropping is a
deletion and not a redesign.** `client_ids_by_issuer` and
`tokens_by_issuer_resource` are both keyed on the issuer, so refusing the
prior issuer's material is **1** key removed from each. What the keying does
not do is *expire* anything: the old token object is still valid until
something drops it.

**FINDING: the resource server refuses a foreign-issuer token anyway, and
cannot say why.** A token minted by issuer B presented at a resource trusting
issuer A answers **401 invalid_token** -- the same code, message and data as an
expired token or a wrong audience, because `issuer != self.issuer` and
`audience != self.resource` share one branch. The client's own hygiene is what
distinguishes them; the wire does not.

Structure: `Wallet` is the client plus the allowlist and the drop; `move` runs
one issuer change and reports what each enrollment style kept.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "16-mcp-security-oauth-2-1"
ISSUER_B = "https://auth-b.example.com"


class Wallet:
    """A client that will only enroll at an allowlisted issuer, and drops on change."""

    def __init__(self, ref, allowed, **kwargs):
        self.ref, self.allowed = ref, set(allowed)
        self.client = ref.Client(**kwargs)

    def enroll(self, auth):
        if auth.issuer not in self.allowed:
            raise ValueError(f"issuer not allowlisted: {auth.issuer}")
        return self.client.enroll(auth)

    def drop(self, issuer):
        """Refuse everything the prior issuer minted."""
        removed = [self.client.client_ids_by_issuer.pop(issuer, None)]
        for key in [k for k in self.client.tokens_by_issuer_resource if k[0] == issuer]:
            removed.append(self.client.tokens_by_issuer_resource.pop(key))
        return len([item for item in removed if item is not None])


def move(ref, allowed, **kwargs):
    """Enroll at A, take a token, then move to B and report what carried over."""
    wallet = Wallet(ref, allowed, **kwargs)
    first = ref.AuthorizationServer()
    second = ref.AuthorizationServer(issuer=ISSUER_B)
    server = ref.ResourceServer()
    wallet.enroll(first)
    token = wallet.client.authorize(first, server.resource, {"notes:read"})
    held = len(wallet.client.tokens_by_issuer_resource)
    dropped = wallet.drop(first.issuer)
    return wallet, second, token, {"held": held, "dropped": dropped}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    allowed = {ref.ISSUER, ISSUER_B}
    cimd_wallet, second, token, counts = move(ref, allowed)
    cimd_id = cimd_wallet.enroll(second)

    dcr_wallet, dcr_second, _, _ = move(ref, allowed)
    dcr_second.supports_cimd = False
    dcr_id = dcr_wallet.enroll(dcr_second)

    strict = Wallet(ref, {ref.ISSUER})
    refused = None
    try:
        strict.enroll(ref.AuthorizationServer(issuer=ISSUER_B))
    except ValueError as exc:
        refused = str(exc)

    server = ref.ResourceServer()
    foreign = ref.Client()
    foreign_token = foreign.authorize(ref.AuthorizationServer(issuer=ISSUER_B),
                                      server.resource, {"notes:read"})
    body, headers = ref.make_mcp_request(1, "notes.list")
    status, response, _ = server.call(body, headers, foreign_token)
    expired = ref.Token(value="t", issuer=ref.ISSUER, audience=server.resource,
                        subject="alice", client_id="c", scopes=frozenset({"notes:read"}),
                        expires_at=0)
    expired_status, expired_response, _ = server.call(body, headers, expired)
    return {
        "cimd_id": cimd_id, "metadata_url": ref.CLIENT_METADATA_URL,
        "cimd_portable": cimd_id == ref.CLIENT_METADATA_URL,
        "dcr_id": dcr_id, "dcr_portable": dcr_id == ref.CLIENT_METADATA_URL,
        "refused": refused, "counts": counts,
        "keys_left": len(cimd_wallet.client.tokens_by_issuer_resource),
        "ids_left": sorted(cimd_wallet.client.client_ids_by_issuer),
        "foreign": (status, response["error"]),
        "expired": (expired_status, expired_response["error"]),
        "token_still_valid": token.expires_at > 0,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the CIMD id crosses unchanged and the DCR id does not",
            all([result["cimd_portable"], result["cimd_id"] == result["metadata_url"],
                 not result["dcr_portable"], result["dcr_id"].startswith("dcr_")]),
            f"enrolling at the second issuer returns {result['cimd_id']} -- the identical URL, "
            f"because the client id is the document's address -- while the DCR client gets "
            f"{result['dcr_id']}, an opaque string that means nothing to anyone else. One "
            "identifier reused, one reissued",
        ),
        practice.Check(
            "FINDING: the allowlist is what makes the move deliberate",
            result["refused"] == f"issuer not allowlisted: {ISSUER_B}",
            f"with only the first issuer allowed, enrolling at the second answers "
            f"{result['refused']!r} before any credential is minted. Without the list the "
            "client follows whichever issuer the resource named -- and that metadata is what "
            "an attacker who controls the resource gets to write",
        ),
        practice.Check(
            "FINDING: the client's stores are keyed by issuer, so dropping is a deletion",
            all([result["counts"] == {"held": 1, "dropped": 2},
                 result["keys_left"] == 0, result["ids_left"] == [ISSUER_B],
                 result["token_still_valid"]]),
            f"client_ids_by_issuer and tokens_by_issuer_resource are both keyed on the "
            f"issuer, so refusing the prior issuer's material removes {result['counts']} and "
            f"leaves {result['keys_left']} tokens and only the new issuer's id, "
            f"{result['ids_left']}. What the keying does not do is expire anything -- the "
            "dropped token object is still valid until something drops it",
        ),
        practice.Check(
            "FINDING: the resource server refuses a foreign issuer and cannot say why",
            all([result["foreign"][0] == 401,
                 result["foreign"][1] == result["expired"][1],
                 result["expired"][0] == 401]),
            f"a token minted by the second issuer answers {result['foreign']} at a resource "
            f"trusting the first -- the same code, message and data as an expired token, "
            "because the issuer and audience checks share one branch. The client's own "
            "hygiene is what distinguishes them; the wire does not",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
