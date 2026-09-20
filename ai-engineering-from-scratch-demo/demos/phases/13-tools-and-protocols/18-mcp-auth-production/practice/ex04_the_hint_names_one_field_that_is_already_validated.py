"""Exercise 4 — the hint names one field that is already validated.

    Read RFC 7591 and identify two fields the lesson's `/register` handler
    does not validate. Add the validation. (Hint: `software_statement` and
    `redirect_uris` URI scheme.)

Reading of the exercise: the hint is checked before it is followed.
`software_statement` is indeed ignored, but the redirect-URI scheme is
already validated in three places -- `parsed_absolute_redirect_uri` rejects a
schemeless value, `valid_web_redirect_uri` demands https, and
`valid_native_redirect_uri` demands loopback or a reverse-DNS private-use
scheme. So the second unvalidated field is found by reading what the handler
*stores* rather than what it checks, and it is `grant_types`.

**ANSWER: `software_statement` and `grant_types`, both accepted and one of
them stored.** A registration carrying `software_statement: "not.a.jwt"` --
three dot-separated parts, so it passes for a JWS on shape alone and fails
the lesson's own `jwt_decode` -- answers **201** -- RFC 7591 §2.3 makes it a signed JWT whose claims take
precedence, so ignoring it is worse than rejecting it. And
`grant_types: ["implicit", "nonsense"]` is echoed back and persisted verbatim
with **0** checks, on a server that only implements `authorization_code`.

**FINDING: the hint's second field is already validated, in three layers.**
`https://app.example/cb` passes and `http://app.example/cb` is refused as a
web client; `http://127.0.0.1:1234/cb` passes and `http://evil.example/cb` is
refused as a native one; `javascript:alert(1)` is refused as either. The
scheme is the most-checked thing in the handler.

**FINDING: validating them is a rejection each, and they fail differently.**
A `software_statement` that is not a well-formed JWS is a
`invalid_software_statement` per RFC 7591 §3.2.2, while an unsupported grant
is `invalid_client_metadata`. Collapsing both into one code would lose which
half of the document the caller has to fix.

**FINDING: the unvalidated grant is not inert -- it is quoted back.** The
**201** body returns `grant_types` as sent, so a caller reading the response
is told the server supports `implicit`. A registration response is a
statement about the server, and this one repeats the client's claim as if it
were one.

Structure: `register` runs one body through the lesson's handler; `validated`
adds the two checks the exercise asks for and runs the same bodies again.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "18-mcp-auth-production"
SUPPORTED_GRANTS = {"authorization_code", "refresh_token"}
WEB = {"application_type": "web", "redirect_uris": ["https://app.example/cb"]}


def register(auth, **extra):
    response = auth.register_client({**WEB, **extra})
    return response["status"], response["body"]


def validated(ref, auth, body):
    """The two checks RFC 7591 asks for and the handler omits."""
    statement = body.get("software_statement")
    if statement is not None:
        try:  # the lesson's own decoder, which a three-dot string does not survive
            ref.jwt_decode(str(statement))
        except Exception:
            return 400, {"error": "invalid_software_statement"}
    grants = body.get("grant_types", ["authorization_code"])
    if not isinstance(grants, list) or not set(grants) <= SUPPORTED_GRANTS:
        return 400, {"error": "invalid_client_metadata"}
    response = auth.register_client(body)
    return response["status"], response["body"]


def scheme_case(auth, application_type, uri):
    response = auth.register_client({"application_type": application_type,
                                     "redirect_uris": [uri]})
    return response["status"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    auth = ref.AuthorizationServer()
    statement_status, _ = register(auth, software_statement="not.a.jwt")
    grants_status, grants_body = register(auth, grant_types=["implicit", "nonsense"])
    stored = next(record["grant_types"] for record in auth.clients.values()
                  if record["grant_types"] == ["implicit", "nonsense"])
    return {
        "statement_status": statement_status,
        "grants_status": grants_status, "echoed": grants_body.get("grant_types"),
        "stored": stored,
        "fixed_statement": validated(ref, auth, {**WEB, "software_statement": "not.a.jwt"}),
        "fixed_grants": validated(ref, auth, {**WEB, "grant_types": ["implicit"]}),
        "fixed_ok": validated(ref, auth, dict(WEB))[0],
        "web_https": scheme_case(auth, "web", "https://app.example/cb"),
        "web_http": scheme_case(auth, "web", "http://app.example/cb"),
        "native_loopback": scheme_case(auth, "native", "http://127.0.0.1:1234/cb"),
        "native_remote": scheme_case(auth, "native", "http://evil.example/cb"),
        "javascript": [scheme_case(auth, kind, "javascript:alert(1)")
                       for kind in ("web", "native")],
        "supported": sorted(SUPPORTED_GRANTS),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: software_statement and grant_types, both accepted and one stored",
            all([result["statement_status"] == 201, result["grants_status"] == 201,
                 result["echoed"] == ["implicit", "nonsense"],
                 result["stored"] == ["implicit", "nonsense"]]),
            f"a registration carrying software_statement 'not.a.jwt' answers "
            f"{result['statement_status']}, and grant_types {result['echoed']} is echoed and "
            f"persisted as {result['stored']} with no checks -- on a server that implements "
            f"only {result['supported'][0]}",
        ),
        practice.Check(
            "FINDING: the hint's second field is already validated, in three layers",
            all([result["web_https"] == 201, result["web_http"] == 400,
                 result["native_loopback"] == 201, result["native_remote"] == 400,
                 result["javascript"] == [400, 400]]),
            f"https passes and http is refused for a web client "
            f"({result['web_https']}/{result['web_http']}); loopback passes and a remote "
            f"http is refused for a native one "
            f"({result['native_loopback']}/{result['native_remote']}); and "
            f"javascript: is refused as either, {result['javascript']}. The scheme is the "
            "most-checked thing in the handler",
        ),
        practice.Check(
            "FINDING: validating them is a rejection each, and they fail differently",
            all([result["fixed_statement"] == (400, {"error": "invalid_software_statement"}),
                 result["fixed_grants"] == (400, {"error": "invalid_client_metadata"}),
                 result["fixed_ok"] == 201]),
            f"a malformed software_statement becomes {result['fixed_statement'][1]} per RFC "
            f"7591 3.2.2 and an unsupported grant {result['fixed_grants'][1]}, while a clean "
            f"document still answers {result['fixed_ok']}. Collapsing both into one code "
            "would lose which half of the document the caller has to fix",
        ),
        practice.Check(
            "FINDING: the unvalidated grant is not inert -- it is quoted back",
            all([result["echoed"] == result["stored"], result["grants_status"] == 201]),
            f"the 201 body returns grant_types as sent, {result['echoed']}, so a caller "
            "reading the response is told the server supports implicit. A registration "
            "response is a statement about the server, and this one repeats the client's "
            "claim as if it were one",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
