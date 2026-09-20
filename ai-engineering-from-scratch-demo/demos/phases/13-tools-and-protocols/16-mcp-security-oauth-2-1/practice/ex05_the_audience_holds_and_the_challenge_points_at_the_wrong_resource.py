"""Exercise 5 — the audience holds, and the challenge points at the wrong resource.

    Add a second resource under the same issuer. Confirm its access token
    cannot be used at the first resource.

Reading of the exercise: confirming the refusal is one comparison, so the
response is read in full rather than by status code -- and the part that is
not the status turns out to be wrong. A `401` is an instruction to go and get
the right token, and the instruction it carries names the wrong place.

**ANSWER: the second resource's token is refused at the first, and the reverse
too.** Both directions answer **401** with `invalid_token`, because
`token.audience != self.resource`. The audience is written at exchange time
from the `resource` parameter, so the two tokens are distinguishable at issue
and not merely at use.

**FINDING: the `WWW-Authenticate` challenge names a module constant, not the
server.** Both resource servers answer with
`resource_metadata="https://notes.example.com/.well-known/..."`, because the
handler builds the challenge from `RESOURCE_METADATA_URI` while checking
against `self.resource`. A client that follows the challenge from the second
resource is sent to read the first resource's metadata -- and would come back
with a token for the wrong audience, in a loop.

**FINDING: the client will not make that mistake, because its cache is keyed
on the pair.** `tokens_by_issuer_resource` is keyed `(issuer, resource)`, so
asking for the second resource mints a second token rather than reusing the
first: **2** entries, **2** audiences, **1** issuer. The isolation is enforced
on both sides, and only the server's side is the protocol's.

**FINDING: wrong-resource and wrong-issuer are one branch, so the client
cannot tell which it got wrong.** A same-issuer token for the other resource
and a foreign-issuer token for this one produce identical errors. One needs a
new token from the same authorization server and the other needs a different
authorization server entirely, and the response distinguishes them by nothing.

Structure: `token_for` authorizes one client against one resource, and `call`
presents any token at any server and returns the status with the challenge.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "16-mcp-security-oauth-2-1"
SECOND = "https://issues.example.com/mcp"
OTHER_ISSUER = "https://auth-b.example.com"


def call(ref, server, token, request_id=1):
    body, headers = ref.make_mcp_request(request_id, "notes.list")
    status, response, response_headers = server.call(body, headers, token)
    error = (response or {}).get("error", {})
    return {"status": status, "code": error.get("code"), "data": error.get("data"),
            "challenge": response_headers.get("WWW-Authenticate")}


def ref_resource():
    return "https://notes.example.com/mcp"


def metadata_uri(challenge):
    return challenge.split('resource_metadata="', 1)[1].split('"', 1)[0]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    auth = ref.AuthorizationServer()
    first, second = ref.ResourceServer(), ref.ResourceServer(resource=SECOND)
    client = ref.Client()
    first_token = client.authorize(auth, first.resource, {"notes:read"})
    second_token = client.authorize(auth, second.resource, {"notes:read"})

    foreign = ref.Client()
    foreign_token = foreign.authorize(ref.AuthorizationServer(issuer=OTHER_ISSUER),
                                      first.resource, {"notes:read"})
    return {
        "audiences": sorted({first_token.audience, second_token.audience}),
        "issuers": sorted({first_token.issuer, second_token.issuer}),
        "cached": len(client.tokens_by_issuer_resource),
        "cache_keys": sorted(client.tokens_by_issuer_resource),
        "right": call(ref, first, first_token),
        "crossed": call(ref, first, second_token, 2),
        "reverse": call(ref, second, first_token, 3),
        "foreign": call(ref, first, foreign_token, 4),
        "first_metadata": ref.RESOURCE_METADATA_URI,
        "second_resource": second.resource,
    }


def verify(result):
    crossed, reverse = result["crossed"], result["reverse"]
    return [
        practice.Check(
            "ANSWER: the second resource's token is refused at the first, and the reverse too",
            all([result["right"]["status"] == 200,
                 crossed["status"] == 401, crossed["data"] == {"reason": "invalid_token"},
                 reverse["status"] == 401, reverse["data"] == crossed["data"],
                 result["audiences"] == sorted([ref_resource(), SECOND]),
                 len(result["issuers"]) == 1]),
            f"the matching token answers HTTP {result['right']['status']} and each token at "
            f"the other resource answers {crossed['status']} {crossed['data']}. The two "
            f"audiences are {result['audiences']} under {len(result['issuers'])} issuer, so "
            "they are distinguishable at issue and not merely at use",
        ),
        practice.Check(
            "FINDING: the WWW-Authenticate challenge names a module constant, not the server",
            all([metadata_uri(reverse["challenge"]) == result["first_metadata"],
                 result["second_resource"] not in reverse["challenge"]]),
            f"the second resource answers with resource_metadata="
            f"{metadata_uri(reverse['challenge'])} -- the first resource's -- because the "
            f"handler builds the challenge from RESOURCE_METADATA_URI while checking against "
            f"self.resource. A client following it reads the wrong metadata and comes back "
            "with a token for the wrong audience, in a loop",
        ),
        practice.Check(
            "FINDING: the client will not make that mistake, because its cache is keyed on the pair",
            all([result["cached"] == 2,
                 [key[1] for key in result["cache_keys"]] == [SECOND,
                                                              "https://notes.example.com/mcp"],
                 len({key[0] for key in result["cache_keys"]}) == 1]),
            f"tokens_by_issuer_resource holds {result['cached']} entries keyed on "
            f"(issuer, resource), so asking for the second resource mints a second token "
            "rather than reusing the first. The isolation is enforced on both sides, and "
            "only the server's side is the protocol's",
        ),
        practice.Check(
            "FINDING: wrong-resource and wrong-issuer are one branch",
            all([result["foreign"]["status"] == crossed["status"],
                 result["foreign"]["code"] == crossed["code"],
                 result["foreign"]["data"] == crossed["data"]]),
            f"a same-issuer token for the other resource and a foreign-issuer token for this "
            f"one both answer {crossed['status']} {crossed['code']} {crossed['data']}. One "
            "needs a new token from the same authorization server and the other needs a "
            "different authorization server, and the response separates them by nothing",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
