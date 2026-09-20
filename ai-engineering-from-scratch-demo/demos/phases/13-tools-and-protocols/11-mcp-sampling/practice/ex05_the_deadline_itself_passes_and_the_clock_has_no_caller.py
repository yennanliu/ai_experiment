"""Exercise 5 — the deadline itself passes, and the clock has no caller.

    Add an expiry test using a state value that is one second past its
    deadline.

Reading of the exercise: "one second past" names a boundary, and a boundary is
only worth testing from both sides, so `expiresAt` itself is tried alongside
`expiresAt + 1`. Writing the test then turns up the reason one was missing:
`verify_request_state` takes an injectable `now`, and the only caller in the
lesson never passes it.

**ANSWER: `expiresAt + 1` raises `-32602 requestState expired`, and
`expiresAt` itself does not.** The comparison is `expiresAt < now`, so the
deadline second is inside the window and the failure begins exactly one second
later. A test written at the deadline would have passed and proved nothing.

**FINDING: the injectable clock has no caller.** `verify_request_state`
accepts `now=None`, but `tools_call` calls it with `principal` and `arguments`
only, so the live path always reads `time.time()`. The parameter exists for a
test that the lesson does not ship -- which is why this exercise has to reach
past `dispatch` to `verify_request_state` directly.

**FINDING: expiry is checked last, so an expired token reports something
else.** The order is integrity, principal, method, arguments, expiry. A token
that is both expired and carrying changed arguments answers `requestState
arguments mismatch`; fix the arguments and the same token then answers
`requestState expired`. Two round trips to learn two faults, and the one the
client can actually fix is reported second.

**FINDING: the window is 300 seconds and is re-granted, not consumed.** The
`pick` state expires 300s out and the `summarize` state re-seals at `now +
300`, so the deadline measures idle time between rounds rather than the age of
the flow. An expired token is only ever evidence that *this* round was slow.

Structure: `at` runs `verify_request_state` at an injected instant and reports
the message, so every row is the same call with one number changed.
"""

from __future__ import annotations

import base64
import inspect
import json

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "11-mcp-sampling"
ARGUMENTS = {"audience": "developer"}
CHANGED = {"audience": "executive"}
PRINCIPAL = "user-42"
PICKS = '["README.md","server.py"]'


def unseal(token):
    body = token.split(".", 1)[0]
    return json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))


def at(ref, token, now, arguments=None):
    """Verify one token at an injected instant, reporting only the outcome."""
    try:
        ref.verify_request_state(token, principal=PRINCIPAL,
                                 arguments=arguments or ARGUMENTS, now=now)
        return None
    except ref.McpError as exc:
        return exc.message


def open_flow(ref):
    params = {"name": "summarize_repo", "arguments": dict(ARGUMENTS), "_meta": ref.request_meta()}
    return ref.dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": params})["result"]["requestState"]


def second_token(ref, token):
    params = {"name": "summarize_repo", "arguments": dict(ARGUMENTS),
              "_meta": ref.request_meta(), "requestState": token,
              "inputResponses": {"pick_files": {"role": "assistant",
                                                "content": {"type": "text", "text": PICKS}}}}
    return ref.dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                         "params": params})["result"]["requestState"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    token = open_flow(ref)
    deadline = unseal(token)["expiresAt"]
    later = unseal(second_token(ref, token))["expiresAt"]
    call = inspect.signature(ref.verify_request_state).parameters
    body = inspect.getsource(ref.tools_call)
    return {
        "deadline": deadline,
        "before": at(ref, token, deadline - 1),
        "exact": at(ref, token, deadline),
        "past": at(ref, token, deadline + 1),
        "far_past": at(ref, token, deadline + 86_400),
        "expired_and_changed": at(ref, token, deadline + 1, arguments=CHANGED),
        "signature": list(call), "now_default": call["now"].default,
        "caller_passes_now": "now=" in body,
        "window": deadline - (unseal(token)["expiresAt"] - 300),
        "renewed": later >= deadline,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: expiresAt + 1 is expired and expiresAt itself is not",
            all([result["past"] == "requestState expired",
                 result["far_past"] == "requestState expired",
                 result["exact"] is None, result["before"] is None]),
            f"at the deadline the token verifies ({result['exact']}) and one second later it "
            f"raises {result['past']!r}, as it still does a day later. The comparison is "
            "`expiresAt < now`, so the deadline second is inside the window -- a test written "
            "at the deadline would have passed and proved nothing",
        ),
        practice.Check(
            "FINDING: the injectable clock has no caller",
            all([result["signature"] == ["token", "principal", "arguments", "now"],
                 result["now_default"] is None, not result["caller_passes_now"]]),
            f"verify_request_state's parameters are {result['signature']} with now defaulting "
            f"to {result['now_default']}, and tools_call passes it: "
            f"{result['caller_passes_now']}. The live path always reads time.time(), so the "
            "parameter exists for a test the lesson does not ship -- which is why this "
            "exercise reaches past dispatch",
        ),
        practice.Check(
            "FINDING: expiry is checked last, so an expired token reports something else",
            all([result["expired_and_changed"] == "requestState arguments mismatch",
                 result["past"] == "requestState expired"]),
            f"a token that is both expired and carrying changed arguments answers "
            f"{result['expired_and_changed']!r}; fix the arguments and the same token answers "
            f"{result['past']!r}. Two round trips for two faults, and the one the client can "
            "actually fix is reported second",
        ),
        practice.Check(
            "FINDING: the window is 300 seconds and is re-granted, not consumed",
            all([result["window"] == 300, result["renewed"]]),
            f"the pick state expires {result['window']} seconds out and the summarize state "
            "re-seals at now + 300, so the deadline measures idle time between rounds rather "
            "than the age of the flow. An expired token is only ever evidence that this round "
            "was slow",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
