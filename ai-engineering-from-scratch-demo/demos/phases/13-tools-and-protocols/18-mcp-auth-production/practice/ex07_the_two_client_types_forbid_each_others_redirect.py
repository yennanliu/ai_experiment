"""Exercise 7 — the two client types forbid each other's redirect.

    Exercise deprecated DCR with both `native` and `web` clients. Confirm a
    web client with an HTTP redirect URI and a native client without an exact
    loopback redirect are rejected.

Reading of the exercise: the two rejections are one line each, so the sweep is
widened to the URIs each type *accepts*, which is where the asymmetry lives.
A web client and a native client are not two settings of one rule; they are
two rules, and every URI in the sweep is legal for at most one of them.

**ANSWER: both rejections land, with `invalid_redirect_uri`.** A web client
offering `http://app.example/cb` is refused and a native client offering
`http://evil.example/cb` is refused, while `https://app.example/cb` and
`http://127.0.0.1:1234/cb` are accepted by their own types. **4** cases,
**2** rejections, both **400**.

**FINDING: the two rules are disjoint on the interesting cases.** Of the
sweep, `https://app.example/cb` is legal for both, loopback only for native,
plain remote HTTP for neither, and a reverse-DNS private-use scheme
(`com.example.app:/cb`) only for native. No URI is web-only, which is what
makes "web" the narrower rule rather than a different one.

**FINDING: the loopback allowance is by hostname, not by port.**
`http://127.0.0.1:1234/cb` and `http://127.0.0.1:9999/cb` are both accepted,
because `valid_native_redirect_uri` checks `hostname in {localhost, 127.0.0.1,
::1}` and never reads the port -- which is the correct reading of RFC 8252,
since a native app cannot reserve a port in advance.

**FINDING: `application_type` is required, so there is no third behaviour.**
Omitting it answers `invalid_client_metadata` rather than falling back to
either rule -- and a value of `"mobile"` does too. The handler will not guess
which of the two rules to apply, which is what keeps the disjointness above
from becoming a default.

Structure: `attempt` registers one redirect URI under one application type and
reports the status with the error code.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "18-mcp-auth-production"
SWEEP = ["https://app.example/cb", "http://app.example/cb",
         "http://127.0.0.1:1234/cb", "com.example.app:/cb"]


def attempt(auth, application_type, uri):
    body = {"redirect_uris": [uri]}
    if application_type is not None:
        body["application_type"] = application_type
    response = auth.register_client(body)
    return response["status"], response["body"].get("error")


def legal(auth, uri):
    """Which application types accept this URI."""
    return [kind for kind in ("web", "native") if attempt(auth, kind, uri)[0] == 201]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    auth = ref.AuthorizationServer()
    return {
        "web_https": attempt(auth, "web", "https://app.example/cb"),
        "web_http": attempt(auth, "web", "http://app.example/cb"),
        "native_loopback": attempt(auth, "native", "http://127.0.0.1:1234/cb"),
        "native_remote": attempt(auth, "native", "http://evil.example/cb"),
        "matrix": {uri: legal(auth, uri) for uri in SWEEP},
        "other_port": attempt(auth, "native", "http://127.0.0.1:9999/cb")[0],
        "localhost": attempt(auth, "native", "http://localhost:1/cb")[0],
        "missing_type": attempt(auth, None, "https://app.example/cb"),
        "unknown_type": attempt(auth, "mobile", "https://app.example/cb"),
    }


def verify(result):
    matrix = result["matrix"]
    return [
        practice.Check(
            "ANSWER: both rejections land, with invalid_redirect_uri",
            all([result["web_http"] == (400, "invalid_redirect_uri"),
                 result["native_remote"] == (400, "invalid_redirect_uri"),
                 result["web_https"][0] == 201, result["native_loopback"][0] == 201]),
            f"a web client offering http answers {result['web_http']} and a native client "
            f"offering a remote http URI answers {result['native_remote']}, while "
            f"https ({result['web_https'][0]}) and loopback "
            f"({result['native_loopback'][0]}) are accepted by their own types",
        ),
        practice.Check(
            "FINDING: the two rules are disjoint on the interesting cases",
            all([matrix["https://app.example/cb"] == ["web", "native"],
                 matrix["http://app.example/cb"] == [],
                 matrix["http://127.0.0.1:1234/cb"] == ["native"],
                 matrix["com.example.app:/cb"] == ["native"]]),
            f"across the sweep: https is legal for {matrix['https://app.example/cb']}, remote "
            f"http for {matrix['http://app.example/cb']}, loopback for "
            f"{matrix['http://127.0.0.1:1234/cb']} and a reverse-DNS private-use scheme for "
            f"{matrix['com.example.app:/cb']}. No URI is web-only, which makes web the "
            "narrower rule rather than a different one",
        ),
        practice.Check(
            "FINDING: the loopback allowance is by hostname, not by port",
            all([result["native_loopback"][0] == 201, result["other_port"] == 201,
                 result["localhost"] == 201]),
            "127.0.0.1 on two different ports and localhost on a third are all accepted, "
            "because valid_native_redirect_uri checks the hostname against "
            "{localhost, 127.0.0.1, ::1} and never reads the port -- the correct reading of "
            "RFC 8252, since a native app cannot reserve a port in advance",
        ),
        practice.Check(
            "FINDING: application_type is required, so there is no third behaviour",
            all([result["missing_type"] == (400, "invalid_client_metadata"),
                 result["unknown_type"] == result["missing_type"]]),
            f"omitting application_type answers {result['missing_type']} and 'mobile' answers "
            f"{result['unknown_type']} -- neither falls back to a rule. The handler will not "
            "guess which of the two to apply, which is what stops the disjointness above "
            "from becoming a default",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
