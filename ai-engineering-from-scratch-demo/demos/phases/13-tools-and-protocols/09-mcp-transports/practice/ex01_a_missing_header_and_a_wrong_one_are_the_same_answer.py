"""Exercise 1 — a missing header and a wrong one are the same answer.

    Remove `Mcp-Method` from a POST. Confirm HTTP `400` and error `-32020`.

Reading of the exercise: removing the header is one input, and one input tells
you the code but not what the code distinguishes. So the removal is run against
a live server and then compared with the three neighbouring inputs that ought
to differ from it -- a wrong value, a differently-cased value, and a
differently-cased header *name* -- because `-32020` is only informative if
something else returns something else.

**ANSWER: HTTP `400`, `-32020`, `Header mismatch: Mcp-Method`, with the
JSON-RPC id echoed.** The same request with the header present answers `200`.

**FINDING: absent and wrong are indistinguishable.** Dropping `Mcp-Method` and
setting it to `tools/call` produce byte-identical responses — same status, code
and message. `validate_http_headers` tests `header_method is None or
header_method != message["method"]` and raises one fault for both, so a client
cannot tell "you forgot it" from "you sent the wrong one" without re-reading its
own request.

**FINDING: the name is case-insensitive and the value is not.** `mcp-method`
lowercased answers **200** — `http.client` folds header names — while
`TOOLS/LIST` as a *value* is a mismatch. The docs say this; the measurement is
that the two casings sit on opposite sides of the same request.

**FINDING: the header contract is checked before the version is.** A request
carrying both an unsupported `2027-01-01` and no `Mcp-Method` answers
**-32020**, not `-32022`; sent with the header it answers `-32022`. Mirrored
headers are validated first, so a client fixing two faults at once sees only the
second one after fixing the first.

**FINDING: and a notification is not exempt.** A message with no `id` and no
`Mcp-Method` answers `400` with `"id": null` rather than the `202` a
well-formed notification gets. A fire-and-forget message still gets a reply when
the transport envelope is wrong, so "no id" does not mean "no response".

Structure: `answer` posts one message with the headers the reference would have
built, after `mutate` has had a chance to break them. The handler logs every
request to stderr, so the whole exchange runs under a redirect.
"""

from __future__ import annotations

import contextlib
import io

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "09-mcp-transports"
FAULT = "Header mismatch: Mcp-Method"


def answer(ref, url, message, mutate=None):
    """Post `message` with its own correct headers, optionally broken first."""
    headers = ref.http_headers_for(message)
    if mutate is not None:
        mutate(headers)
    status, _, body = ref.post(url, message, headers)
    error = body.get("error", {}) if isinstance(body, dict) else {}
    return {"status": status, "code": error.get("code"), "message": error.get("message"),
            "id": body.get("id") if isinstance(body, dict) else None}


def lowercased(headers):
    headers["mcp-method"] = headers.pop("Mcp-Method")


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    server = ref.serve()
    url = f"http://127.0.0.1:{server.server_port}/mcp"
    with contextlib.redirect_stderr(io.StringIO()):  # the handler logs every request
        listing = ref.make_request(1, "tools/list")
        future = ref.make_request(2, "tools/list", version="2027-01-01")
        notification = ref.make_request(3, "tools/list")
        del notification["id"]
        answers = {
            "removed": answer(ref, url, listing, lambda h: h.pop("Mcp-Method")),
            "present": answer(ref, url, listing),
            "wrong": answer(ref, url, listing, lambda h: h.update({"Mcp-Method": "tools/call"})),
            "upper": answer(ref, url, listing, lambda h: h.update({"Mcp-Method": "TOOLS/LIST"})),
            "lower_name": answer(ref, url, listing, lowercased),
            "future": answer(ref, url, future),
            "future_removed": answer(ref, url, future, lambda h: h.pop("Mcp-Method")),
            "notification": answer(ref, url, notification),
            "notification_removed": answer(ref, url, notification, lambda h: h.pop("Mcp-Method")),
        }
    server.shutdown()
    server.server_close()
    return answers


def verify(result):
    removed, present = result["removed"], result["present"]
    return [
        practice.Check(
            "ANSWER: removing Mcp-Method gives HTTP 400 and -32020, with the id echoed",
            all([removed["status"] == 400, removed["code"] == -32020,
                 removed["message"] == FAULT, removed["id"] == 1,
                 present["status"] == 200, present["code"] is None]),
            f"the POST answers HTTP {removed['status']} with code {removed['code']}, "
            f"{removed['message']!r}, and id {removed['id']} echoed from the body. The same "
            f"request with the header present answers HTTP {present['status']}",
        ),
        practice.Check(
            "FINDING: an absent header and a wrong one are the same response",
            removed == result["wrong"] == result["upper"],
            f"dropping Mcp-Method, setting it to 'tools/call' and setting it to 'TOOLS/LIST' "
            f"all answer {removed['status']}/{removed['code']}/{removed['message']!r}. "
            "validate_http_headers raises one fault for `is None or != method`, so a client "
            "cannot tell a forgotten header from a wrong one",
        ),
        practice.Check(
            "FINDING: the header name is case-insensitive and its value is not",
            all([result["lower_name"]["status"] == 200, result["upper"]["status"] == 400]),
            f"'mcp-method' lowercased answers HTTP {result['lower_name']['status']} because "
            f"http.client folds header names, while 'TOOLS/LIST' as a value answers "
            f"HTTP {result['upper']['status']}. The two casings sit on opposite sides of one "
            "request",
        ),
        practice.Check(
            "FINDING: the header contract is checked before the version, and notifications are not exempt",
            all([result["future"]["code"] == -32022,
                 result["future_removed"]["code"] == -32020,
                 result["notification"]["status"] == 202,
                 result["notification_removed"]["status"] == 400,
                 result["notification_removed"]["id"] is None]),
            f"an unsupported 2027-01-01 answers {result['future']['code']} with the header "
            f"present and {result['future_removed']['code']} without it, so mirrored headers "
            f"are validated first. And a message with no id answers "
            f"HTTP {result['notification']['status']} when well formed but "
            f"HTTP {result['notification_removed']['status']} with id "
            f"{result['notification_removed']['id']} when not -- no id does not mean no "
            "response",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
