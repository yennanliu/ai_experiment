"""Exercise 3 — the comparison is proved by which failure you get.

    Send a Base64 sentinel `Mcp-Name` for a non-ASCII resource URI. Confirm
    the decoded value is compared with `params.uri`.

Reading of the exercise: `resources/read` has no handler in this server, so a
correct sentinel cannot be confirmed by a successful read -- there is no
success to reach. The comparison is instead shown by moving one input: the same
request with a sentinel encoding a *different* URI must fail differently. A
decode that was never compared would give the two the same answer.

**ANSWER: the decoded value is compared, and the proof is 404 against 400.**
The correct sentinel `=?base64?bm90ZXM6Ly9ub3RlL+WQjeWJjQ==?=` gets past
`validate_http_headers` and dies at the dispatcher with **-32601 Method not
found: resources/read**; a sentinel encoding `notes://other/名` dies earlier
with **-32020 Header mismatch: Mcp-Name**. Only a comparison against
`params.uri` can separate those two.

**FINDING: the contract is enforced for a method the server does not
implement.** An ASCII URI answers the same `-32601`, so the header pipeline and
the dispatcher never consult each other: every `resources/read` is header-
validated in full and then rejected as unknown. `body_name` knows the method
reads `params.uri`; `dispatch` has never heard of it.

**FINDING: the sentinel is not a convention layered on a working channel — it
is what makes the channel usable.** Putting the raw UTF-8 URI in the header
never leaves the client: `http.client` encodes headers as latin-1 and raises
**UnicodeEncodeError** before a request exists. There is no server behaviour to
observe, because there is no request.

**FINDING: a malformed sentinel is a different message under the same code,
and the encoder is round-trip safe against its own syntax.**
`=?base64?!!!notb64?=` answers `-32020 Malformed Base64 MCP header value`
rather than `Header mismatch`. And `encode_header_value("=?base64?zzz?=")`
encodes that literal instead of passing it through, so a value that looks like
an encoding decodes back to itself and can never be read as one.

Structure: `answer` posts one message with the reference's own headers after
`mutate` has had a chance to rewrite `Mcp-Name`, returning a client-side
failure as a status of its own.
"""

from __future__ import annotations

import contextlib
import io

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "09-mcp-transports"
URI = "notes://note/名前"
OTHER = "notes://other/名"
ASCII_URI = "notes://note-1"
LOOKALIKE = "=?base64?zzz?="


def answer(ref, url, message, mutate=None):
    """Post `message`, reporting a client-side refusal instead of a status."""
    headers = ref.http_headers_for(message)
    if mutate is not None:
        mutate(headers)
    try:
        status, _, body = ref.post(url, message, headers)
    except Exception as exc:  # the header never reaches the wire
        return {"status": None, "client_error": type(exc).__name__, "sent": headers.get("Mcp-Name")}
    error = body.get("error", {}) if isinstance(body, dict) else {}
    return {"status": status, "client_error": None, "sent": headers.get("Mcp-Name"),
            "code": error.get("code"), "message": error.get("message")}


def named(value):
    return lambda headers: headers.update({"Mcp-Name": value})


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    server = ref.serve()
    url = f"http://127.0.0.1:{server.server_port}/mcp"
    with contextlib.redirect_stderr(io.StringIO()):  # the handler logs every request
        read = ref.make_request(1, "resources/read", {"uri": URI})
        answers = {
            "sentinel": answer(ref, url, read),
            "wrong": answer(ref, url, read, named(ref.encode_header_value(OTHER))),
            "malformed": answer(ref, url, read, named("=?base64?!!!notb64?=")),
            "raw": answer(ref, url, read, named(URI)),
            "ascii": answer(ref, url, ref.make_request(2, "resources/read", {"uri": ASCII_URI})),
        }
    server.shutdown()
    server.server_close()
    answers["roundtrip"] = ref.decode_header_value(ref.encode_header_value(URI)) == URI
    answers["lookalike"] = ref.encode_header_value(LOOKALIKE)
    answers["lookalike_back"] = ref.decode_header_value(ref.encode_header_value(LOOKALIKE))
    return answers


def verify(result):
    sentinel, wrong = result["sentinel"], result["wrong"]
    return [
        practice.Check(
            "ANSWER: the decoded sentinel is compared with params.uri, and 404 against 400 proves it",
            all([sentinel["status"] == 404, sentinel["code"] == -32601,
                 wrong["status"] == 400, wrong["code"] == -32020,
                 wrong["message"] == "Header mismatch: Mcp-Name", result["roundtrip"]]),
            f"the correct sentinel {sentinel['sent']} passes validate_http_headers and dies at "
            f"the dispatcher, HTTP {sentinel['status']} {sentinel['code']}, while a sentinel "
            f"encoding {OTHER!r} dies earlier, HTTP {wrong['status']} {wrong['code']} "
            f"{wrong['message']!r}. Only a comparison against params.uri separates the two",
        ),
        practice.Check(
            "FINDING: the header contract is enforced for a method the server does not implement",
            all([result["ascii"]["code"] == -32601, result["ascii"]["status"] == 404,
                 result["ascii"]["sent"] == ASCII_URI]),
            f"an ASCII uri sends {result['ascii']['sent']!r} unencoded and answers the same "
            f"{result['ascii']['code']}. body_name knows resources/read reads params.uri and "
            "dispatch has never heard of the method, so every such request is validated in "
            "full and then rejected as unknown",
        ),
        practice.Check(
            "FINDING: the raw value cannot be sent at all, which is why the sentinel exists",
            all([result["raw"]["status"] is None,
                 result["raw"]["client_error"] == "UnicodeEncodeError"]),
            f"putting the raw UTF-8 uri in Mcp-Name raises {result['raw']['client_error']} in "
            "http.client, which encodes headers as latin-1. There is no server behaviour to "
            "observe because there is no request -- the encoding is what makes the channel "
            "usable, not a convention layered on a working one",
        ),
        practice.Check(
            "FINDING: a malformed sentinel is a different message, and the encoder is round-trip safe",
            all([result["malformed"]["code"] == -32020,
                 result["malformed"]["message"] == "Malformed Base64 MCP header value",
                 result["lookalike"] != LOOKALIKE, result["lookalike_back"] == LOOKALIKE]),
            f"'=?base64?!!!notb64?=' answers {result['malformed']['code']} "
            f"{result['malformed']['message']!r} rather than a mismatch, so one code carries "
            f"two faults. And {LOOKALIKE!r} encodes to {result['lookalike']} instead of "
            "passing through, so a value shaped like an encoding decodes back to itself",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
