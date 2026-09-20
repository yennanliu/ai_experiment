"""Exercise 5 — the client is keyed by issuer, and the server it points at is not.

    Add a second authorization server. Confirm the client stores a separate
    issuer-keyed enrollment and refuses to reuse the first issuer's token or
    `client_id`.

Reading of the exercise: the three stores the confirmation needs --
`client_ids_by_issuer`, `access_tokens_by_issuer_resource` and
`expected_issuer` -- are keyed differently from each other, so the check is
run against all three rather than the one the exercise names. Two are keyed
and one is a single slot, and the single slot is the one that moves when the
client is pointed at a second server.

**ANSWER: two enrollments, two tokens, and neither is reused.** After
enrolling at both issuers the client holds **2** entries in
`client_ids_by_issuer` and **2** in `access_tokens_by_issuer_resource` keyed
on `(issuer, resource)`, with distinct tokens. What is issuer-keyed is the
*entry* and not the identifier: both enrollments carry the same
`client_id`, because CIMD makes it the document URL and that URL is portable
across issuers by design. The first issuer's token presented to a resource
server trusting the second answers `iss not allowed`.

**FINDING: `expected_issuer` is a single slot, so pointing the client at the
second server overwrites it.** `discover` assigns `self.expected_issuer =
meta["issuer"]`, and `use_authorization_server` resets it to `None`. The
per-issuer dicts remember both enrollments while the validation state
remembers only the last one -- so an authorization response from the first
issuer, arriving after the switch, is checked against the second issuer's
name.

**FINDING: `auth_server` is one attribute, so the client talks to one server
at a time.** The dicts imply concurrency the object does not have: enrolling
at the second issuer means re-pointing `self.auth_server`, and nothing keeps
a handle on the first. The keying is a memory of past enrollments, not a
router.

**FINDING: refusing to reuse is the resource server's doing, not the
client's.** The client would happily send the first issuer's token if asked
for the wrong key -- the refusal comes from `allowed_issuers` on the other
side, answering **401**. The client's contribution is keying its cache so it
never asks the wrong question.

Structure: `enrol` points the client at one authorization server and takes a
token for one resource; the three stores are read after both rounds.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "18-mcp-auth-production"
SECOND = "https://auth-b.example.com"


def build(ref, issuer=None):
    auth = (ref.AuthorizationServer() if issuer is None
            else ref.AuthorizationServer(issuer=issuer))
    auth.rotate_key()
    return auth


def enrol(ref, client, auth, resource):
    """Point the client at one authorization server and take one token."""
    client.use_authorization_server(auth)
    client_id = client.enroll()
    token = client.authorize({"mcp:tools.invoke"}, resource, "alice")
    return client_id, token


def server_for(ref, auth, resource):
    server = ref.ResourceServer(resource=resource, auth_server=auth,
                                allowed_issuers=[auth.issuer])
    server.refresh_jwks()
    return server


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    first, second = build(ref), build(ref, SECOND)
    client = ref.Client(name="notes-client", auth_server=first,
                        client_metadata_url="https://app.example/metadata.json",
                        client_metadata={"client_id": "https://app.example/metadata.json",
                                         "client_name": "notes-client",
                                         "application_type": "web",
                                         "redirect_uris": ["https://app.example/cb"]})
    first_id, first_token = enrol(ref, client, first, ref.MCP_RESOURCE)
    expected_after_first = client.expected_issuer
    second_id, second_token = enrol(ref, client, second, ref.MCP_RESOURCE)

    second_server = server_for(ref, second, ref.MCP_RESOURCE)
    crossed = second_server.validate(first_token)
    native = second_server.validate(second_token)
    return {
        "ids": sorted(client.client_ids_by_issuer),
        "id_values": sorted({first_id, second_id}),
        "token_keys": sorted(client.access_tokens_by_issuer_resource),
        "tokens_differ": first_token != second_token,
        "expected_after_first": expected_after_first,
        "expected_now": client.expected_issuer,
        "points_at": client.auth_server.issuer,
        "crossed": crossed["www_authenticate"].split('"')[3],
        "crossed_status": crossed["status"], "native": native["valid"],
        "resets": "expected_issuer = None" in inspect.getsource(
            ref.Client.use_authorization_server),
        "single_slot": not isinstance(client.expected_issuer, dict),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: two enrollments, two tokens, and neither is reused",
            all([result["ids"] == sorted(["https://auth.example.com", SECOND]),
                 len(result["id_values"]) == 1, len(result["token_keys"]) == 2,
                 result["tokens_differ"], result["crossed"] == "iss not allowed",
                 result["native"]]),
            f"the client holds {len(result['ids'])} entries in client_ids_by_issuer and "
            f"{len(result['token_keys'])} in access_tokens_by_issuer_resource keyed on "
            f"(issuer, resource), with distinct tokens. What is issuer-keyed is the entry, "
            f"not the identifier: both enrollments carry {result['id_values']}, because CIMD "
            f"makes the client_id the document URL. The first issuer's token at the second "
            f"resource server answers {result['crossed']!r} while its own validates",
        ),
        practice.Check(
            "FINDING: expected_issuer is a single slot, so the second server overwrites it",
            all([result["expected_after_first"] == "https://auth.example.com",
                 result["expected_now"] == SECOND, result["resets"],
                 result["single_slot"]]),
            f"expected_issuer was {result['expected_after_first']} and is now "
            f"{result['expected_now']}, because discover assigns it and "
            "use_authorization_server resets it to None. The per-issuer dicts remember both "
            "enrollments while the validation state remembers only the last, so a response "
            "from the first issuer arriving after the switch is checked against the second's "
            "name",
        ),
        practice.Check(
            "FINDING: auth_server is one attribute, so the client talks to one server at a time",
            all([result["points_at"] == SECOND, len(result["ids"]) == 2]),
            f"the client remembers {len(result['ids'])} enrollments and points at "
            f"{result['points_at']}. Enrolling at the second issuer means re-pointing "
            "self.auth_server, and nothing keeps a handle on the first -- the keying is a "
            "memory of past enrollments rather than a router",
        ),
        practice.Check(
            "FINDING: refusing to reuse is the resource server's doing, not the client's",
            all([result["crossed_status"] == 401, result["crossed"] == "iss not allowed"]),
            f"the refusal is a {result['crossed_status']} from allowed_issuers on the other "
            "side; the client would send the first issuer's token if it were asked for the "
            "wrong key. Its contribution is keying the cache so it never asks the wrong "
            "question",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
