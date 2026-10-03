"""Exercise 3 -- the 401's message is never shown, and a key with a newline leaks whole.

    Intentionally use a wrong API key and read the error message

Reading of the exercise: there is no network here, so the wrong key is sent
through the lesson's own `call_raw_http` twice. First to a fake `urlopen` that
answers the way the API answers a bad key -- HTTP 401 with the documented
error body (AUTH_ERROR) -- and the "error message" is the traceback a learner
would read. Second, a wrong key of the commonest kind, a right key with a
trailing newline, goes through the real `urllib` stack with
`socket.create_connection` replaced by a tripwire, so nothing can leave the
machine. Both keys are dummies; the host environment is cleared.

**ANSWER: the learner reads `HTTP Error 401: Unauthorized`.** That is the
whole message: the API's own explanation, "invalid x-api-key" with type
`authentication_error`, is in the response body, which `call_raw_http` never
reads -- it appears in the traceback 0 times and only `e.read()` recovers it.
The TypeScript port reads it (`resp.text()`, first 200 characters).

**FINDING: a key with a trailing newline never reaches the server -- and the
error prints the whole key.** `http.client` rejects the header before
connecting (the tripwire fires 0 times), raising `ValueError: Invalid header
value b'sk-ant-...\\n'` with all 28 characters of the key in it. A wrong key
is a 401; a malformed one is a secret in your logs.

**CONTROL: the wrong key was really sent.** The fake server saw exactly the
dummy key in `x-api-key`, so the 401 above is the bad-key path, not a missing
one (a missing key stops earlier, at "Set ANTHROPIC_API_KEY").

Structure: `with_401` and `with_newline` run the lesson's raw call once each.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import traceback
import urllib.error
import urllib.request
from unittest import mock

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "04-apis-and-keys"
WRONG_KEY = "sk-ant-wrong-dummy-key-00000"
AUTH_ERROR = {"type": "error",
              "error": {"type": "authentication_error", "message": "invalid x-api-key"}}


def call(env, urlopen):
    """Run call_raw_http in a cleared env; return the exception it raised."""
    with mock.patch.dict(os.environ, {}, clear=True):
        ref = parity.load_reference(PHASE, LESSON, "first_api_call")
    with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch("urllib.request.urlopen", urlopen), \
            contextlib.redirect_stdout(io.StringIO()):
        try:
            ref.call_raw_http()
        except Exception as exc:  # the learner's traceback is the measurement
            return exc
    return None


def with_401():
    seen = {}

    def server(req, *args, **kwargs):
        seen["key"] = req.get_header("X-api-key")
        body = io.BytesIO(json.dumps(AUTH_ERROR).encode())
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, body)

    exc = call({"ANTHROPIC_API_KEY": WRONG_KEY}, server)
    shown = "".join(traceback.format_exception(exc))
    return {"last_line": shown.strip().splitlines()[-1], "sent_key": seen["key"],
            "in_traceback": shown.count("invalid x-api-key"),
            "body": json.loads(exc.read())["error"]}


def with_newline():
    trips = []

    def tripwire(*args, **kwargs):
        trips.append(args)
        raise ConnectionRefusedError("tripwire: no network in this exercise")

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with mock.patch("socket.create_connection", tripwire):
        exc = call({"ANTHROPIC_API_KEY": WRONG_KEY + "\n"}, opener.open)
    return {"type": type(exc).__name__, "message": str(exc), "connects": len(trips)}


def solve():
    ts = (parity.lesson_dir(PHASE, LESSON) / "code" / "first_api_call.ts").read_text()
    return {"auth": with_401(), "newline": with_newline(),
            "ts_reads_body": "await resp.text()" in ts and "body.slice(0, 200)" in ts}


def verify(r):
    a, n = r["auth"], r["newline"]
    return [
        practice.Check(
            "ANSWER: the learner reads 'HTTP Error 401: Unauthorized', not the API's reason",
            a["last_line"].endswith("HTTP Error 401: Unauthorized") and a["in_traceback"] == 0
            and a["body"]["message"] == "invalid x-api-key" and r["ts_reads_body"],
            f"traceback ends {a['last_line']!r}; 'invalid x-api-key' appears in it "
            f"{a['in_traceback']} times, but e.read() gives {a['body']}; the TS port prints "
            f"the body: {r['ts_reads_body']}",
        ),
        practice.Check(
            "FINDING: a key with a trailing newline fails locally and prints the whole key",
            n["type"] == "ValueError" and WRONG_KEY in n["message"] and n["connects"] == 0,
            f"{n['type']}: {n['message']} -- all {len(WRONG_KEY)} characters of the key, "
            f"with {n['connects']} connection attempts",
        ),
        practice.Check(
            "CONTROL: the 401 came from the wrong key actually being sent",
            a["sent_key"] == WRONG_KEY,
            f"x-api-key at the fake server: {a['sent_key']!r}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
