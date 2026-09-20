"""Exercise 2 — "matching" is the word that selects the only path to -32022.

    Send matching header and body version `2027-01-01`. Confirm HTTP `400`,
    error `-32022`, and exact data
    `{"supported":["2026-07-28"],"requested":"2027-01-01"}`.

Reading of the exercise: "exact data" is taken literally, so the check is on
the serialized bytes with the doc's separators rather than on a dict that
happens to compare equal -- key order is the only part of "exact" a dict
equality would not catch. And "matching" is tested by breaking it, in both
directions, because a word in the exercise that changes the answer is worth
one measurement.

**ANSWER: HTTP `400`, `-32022`, and
`{"supported":["2026-07-28"],"requested":"2027-01-01"}` byte for byte** —
`supported` before `requested`, exactly as the lesson prints it.

**FINDING: "matching" is load-bearing, and disagreeing either way is a
different error.** Header `2026-07-28` with body `2027-01-01`, and header
`2027-01-01` with body `2026-07-28`, both answer **-32020 Header mismatch:
MCP-Protocol-Version**. `validate_http_headers` compares the two strings before
`validate_supported_version` ever asks whether the version exists, so an
unsupported version is only reported as unsupported when both copies say so.

**FINDING: `-32022` does not mean "too new".** The legacy `2025-11-25` answers
the same code with the same shape, `{"supported":["2026-07-28"],"requested":
"2025-11-25"}`. The code says the intersection is empty; only `data.supported`
says what to do about it, so a dual-era client must read the payload rather than
branch on the number.

**FINDING: the error carries everything discovery would have, and the HTTP
status is a separate channel.** `data.supported` equals
`server/discover`'s `supportedVersions`, so a client can renegotiate from the
rejection without a second round trip. Meanwhile `-32022` rides on `400` and
`-32601` on `404` — the status is picked from the code by one branch in
`do_POST`, so the two carry the same news in two vocabularies.

Structure: `answer` posts one message with the reference's own headers after
`mutate` has had a chance to disagree with the body, and returns the error
alongside its compact serialization.
"""

from __future__ import annotations

import contextlib
import io
import json

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "09-mcp-transports"
CURRENT, FUTURE, LEGACY = "2026-07-28", "2027-01-01", "2025-11-25"
EXACT = '{"supported":["2026-07-28"],"requested":"2027-01-01"}'
MISMATCH = "Header mismatch: MCP-Protocol-Version"


def answer(ref, url, message, mutate=None):
    """Post `message` with its own headers, optionally made to disagree."""
    headers = ref.http_headers_for(message)
    if mutate is not None:
        mutate(headers)
    status, _, body = ref.post(url, message, headers)
    error = body.get("error", {}) if isinstance(body, dict) else {}
    data = error.get("data")
    return {"status": status, "code": error.get("code"), "message": error.get("message"),
            "data": data, "exact": json.dumps(data, separators=(",", ":")) if data else None,
            "result": body.get("result") if isinstance(body, dict) else None}


def pin(version):
    return lambda headers: headers.update({"MCP-Protocol-Version": version})


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    server = ref.serve()
    url = f"http://127.0.0.1:{server.server_port}/mcp"
    with contextlib.redirect_stderr(io.StringIO()):  # the handler logs every request
        future = ref.make_request(1, "tools/list", version=FUTURE)
        current = ref.make_request(2, "tools/list")
        answers = {
            "matched": answer(ref, url, future),
            "header_current": answer(ref, url, future, pin(CURRENT)),
            "header_future": answer(ref, url, current, pin(FUTURE)),
            "legacy": answer(ref, url, ref.make_request(3, "tools/list", version=LEGACY)),
            "discover": answer(ref, url, ref.make_request(4, "server/discover")),
            "unknown": answer(ref, url, ref.make_request(5, "tools/nope")),
        }
    server.shutdown()
    server.server_close()
    return answers


def verify(result):
    matched, legacy = result["matched"], result["legacy"]
    return [
        practice.Check(
            "ANSWER: HTTP 400, -32022, and the data serializes exactly as the lesson prints it",
            all([matched["status"] == 400, matched["code"] == -32022,
                 matched["message"] == "Unsupported protocol version",
                 matched["exact"] == EXACT]),
            f"the POST answers HTTP {matched['status']} with code {matched['code']} and data "
            f"{matched['exact']} -- supported before requested, byte for byte the lesson's "
            f"own {EXACT}",
        ),
        practice.Check(
            "FINDING: disagreeing header and body is -32020 in both directions, never -32022",
            all([result["header_current"]["code"] == -32020,
                 result["header_future"]["code"] == -32020,
                 result["header_current"]["message"] == MISMATCH,
                 result["header_current"]["data"] is None]),
            f"header {CURRENT} over body {FUTURE}, and header {FUTURE} over body {CURRENT}, "
            f"both answer {result['header_current']['code']} {MISMATCH!r} with no data. The "
            "two strings are compared before validate_supported_version asks whether the "
            "version exists, so 'matching' is what selects the path the exercise wants",
        ),
        practice.Check(
            "FINDING: -32022 does not mean too new -- the legacy revision gets the same code",
            all([legacy["code"] == matched["code"], legacy["status"] == matched["status"],
                 legacy["data"] == {"supported": [CURRENT], "requested": LEGACY}]),
            f"{LEGACY} answers {legacy['code']} on HTTP {legacy['status']} with "
            f"{legacy['exact']}. The code reports an empty intersection and not a direction, "
            "so a dual-era client has to read data.supported rather than branch on the number",
        ),
        practice.Check(
            "FINDING: the rejection carries discovery's answer, and the status is a separate channel",
            all([matched["data"]["supported"] == result["discover"]["result"]["supportedVersions"],
                 result["unknown"]["code"] == -32601, result["unknown"]["status"] == 404,
                 matched["status"] == 400]),
            f"data.supported is {matched['data']['supported']}, which is exactly "
            f"server/discover's supportedVersions, so renegotiation needs no second round "
            f"trip. And {matched['code']} rides on HTTP {matched['status']} while "
            f"{result['unknown']['code']} rides on HTTP {result['unknown']['status']} -- one "
            "branch in do_POST picks the status from the code, so both carry the same news",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
