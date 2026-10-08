"""Exercise 2 -- both paths print a truncated reply exactly as if it were complete.

    Try the raw HTTP version and compare the response format to the SDK version

Reading of the exercise: with no network and no key, both of the lesson's own
functions -- `call_raw_http` and `call_with_sdk` -- are fed the same
/v1/messages response: a fixture in the documented response shape (RESPONSE),
returned as JSON bytes by a fake `urlopen` and as attribute objects by a fake
`anthropic` SDK. "Compare the format" is read as: what each path sends, what it
gets back, and how much of that the lesson's code actually looks at.

**ANSWER: same body, different envelope, and the code reads 2 of 8 fields.**
The raw path's JSON body equals the SDK's keyword arguments key for key; it
adds three headers by hand (`x-api-key`, `anthropic-version: 2023-06-01`,
`Content-Type`). The reply is a dict on one side and attribute objects on the
other, and both print the same text and "12 in, 28 out" -- reading only
`content` and `usage` of the response's 8 top-level fields.

**FINDING: truncation is invisible.** Replay the fixture with
`stop_reason: "max_tokens"` instead of `"end_turn"` and both paths print
byte-identical output: with `max_tokens=256` hard-coded, a cut-off answer looks
finished, because `stop_reason` is one of the 6 fields never read.

**FINDING: the raw call has no timeout.** `urlopen(req)` is called with 1
argument and no `timeout`, so it inherits the socket default -- None, which
blocks forever on a stalled connection.

Structure: `raw` and `sdk` run the lesson's two functions on one fixture.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import sys
import types
from unittest import mock

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "04-apis-and-keys"
ENV = {"ANTHROPIC_API_KEY": "sk-ant-dummy-not-a-real-key"}
RESPONSE = {  # the documented /v1/messages response shape, with fixture values
    "id": "msg_fixture", "type": "message", "role": "assistant", "model": "claude-sonnet-5",
    "content": [{"type": "text", "text": "A neural network is a learned function."}],
    "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 28},
}



# the harness finds the reference through AIEFS_REFERENCE (CI sets it), so a cleared
# environment keeps that one variable and nothing else from the host
KEEP = {k: os.environ[k] for k in ("AIEFS_REFERENCE",) if k in os.environ}


def clean_env(env):
    return mock.patch.dict(os.environ, {**KEEP, **env}, clear=True)

def load():
    with clean_env(ENV):
        return parity.load_reference(PHASE, LESSON, "first_api_call")


def raw(response):
    """call_raw_http against a fake urlopen: (stdout, request, urlopen args, kwargs)."""
    seen, out = {}, io.StringIO()

    def urlopen(*args, **kwargs):
        seen.update(request=args[0], args=args, kwargs=kwargs)
        body = types.SimpleNamespace(read=lambda: json.dumps(response).encode())
        return contextlib.nullcontext(body)

    with clean_env(ENV), \
            mock.patch("urllib.request.urlopen", urlopen), contextlib.redirect_stdout(out):
        load().call_raw_http()
    return out.getvalue(), seen


def sdk(response):
    """call_with_sdk against a fake SDK returning the same fixture as objects."""
    sent, out = [], io.StringIO()
    reply = json.loads(json.dumps(response), object_hook=lambda d: types.SimpleNamespace(**d))
    create = lambda **kwargs: sent.append(kwargs) or reply  # noqa: E731
    module = types.SimpleNamespace(Anthropic=lambda **k: types.SimpleNamespace(
        messages=types.SimpleNamespace(create=create)))
    with clean_env(ENV), \
            mock.patch.dict(sys.modules, {"anthropic": module}), contextlib.redirect_stdout(out):
        load().call_with_sdk()
    return out.getvalue(), sent[0]


def body_lines(text):
    """Printed lines with the path's own prefix ("Raw HTTP response:" / "SDK response:") cut."""
    return [line.split(":", 1)[1] if "response:" in line else line for line in text.splitlines()]


def solve():
    raw_out, seen = raw(RESPONSE)
    sdk_out, kwargs = sdk(RESPONSE)
    cut = {**RESPONSE, "stop_reason": "max_tokens"}
    request = seen["request"]
    return {
        "same_body": json.loads(request.data) == kwargs,
        "headers": dict(request.header_items()),
        "same_print": body_lines(raw_out) == body_lines(sdk_out),
        "fields": len(RESPONSE),
        "two_fields_same": raw({k: RESPONSE[k] for k in ("content", "usage")})[0] == raw_out,
        "raw_cut_same": raw(cut)[0] == raw_out, "sdk_cut_same": sdk(cut)[0] == sdk_out,
        "urlopen_args": len(seen["args"]), "urlopen_kwargs": sorted(seen["kwargs"]),
        "socket_default": socket.getdefaulttimeout(),
        "printed": raw_out.strip().splitlines(),
    }


def verify(r):
    h = r["headers"]
    return [
        practice.Check(
            "ANSWER: identical body and printout; the code reads 2 of 8 response fields",
            all((r["same_body"], r["same_print"], h.get("Anthropic-version") == "2023-06-01",
                 "X-api-key" in h, r["fields"] == 8, r["two_fields_same"])),
            f"raw JSON body == SDK kwargs: {r['same_body']}; hand-set headers "
            f"{sorted(h)}; both print {r['printed']}, and a response cut to content and usage "
            f"alone prints the same: {r['two_fields_same']}",
        ),
        practice.Check(
            "FINDING: a max_tokens cut-off prints byte-identically to a finished reply",
            r["raw_cut_same"] and r["sdk_cut_same"],
            f"stop_reason 'max_tokens' vs 'end_turn': raw output identical {r['raw_cut_same']},"
            f" SDK output identical {r['sdk_cut_same']}",
        ),
        practice.Check(
            "FINDING: the raw urlopen call has no timeout",
            all((r["urlopen_args"] == 1, "timeout" not in r["urlopen_kwargs"],
                 r["socket_default"] is None)),
            f"urlopen got {r['urlopen_args']} positional argument and kwargs "
            f"{r['urlopen_kwargs']}: the socket default timeout, "
            f"{r['socket_default']}, applies",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
