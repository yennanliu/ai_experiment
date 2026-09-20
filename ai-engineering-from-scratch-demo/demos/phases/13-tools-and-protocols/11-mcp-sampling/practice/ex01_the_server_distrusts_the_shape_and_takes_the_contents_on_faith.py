"""Exercise 1 — the server distrusts the shape and takes the contents on faith.

    Change the file-selection response to invalid JSON. Confirm the server
    returns `-32602` instead of trusting model output.

Reading of the exercise: one bad response confirms the code but not the
boundary, so the whole path from the host's text to `picked` is walked with
six responses -- three malformed, one wholly invented, one partly invented,
and one over-long. "Instead of trusting model output" is then answerable as a
degree rather than a yes: the interesting responses are the ones the server
accepts.

**ANSWER: `-32602`, `pick_files must return JSON`.** The model's text is
parsed before it is used, so unparseable output never reaches `FAKE_REPO`.

**FINDING: there are three distinct rejections, and they test different
things.** `not json` fails parsing; `{"a":1}` and `[1,2]` fail the shape check
`isinstance(picks, list) and all(isinstance(item, str))`; `["nope.md"]` parses
and is well-shaped and fails on membership. Structure and contents are
separate gates with separate messages.

**FINDING: the membership gate is "at least one", not "all".**
`["README.md","ghost.md","server.py"]` succeeds and yields
`['README.md', 'server.py']` — the invented name is filtered out by
`[name for name in picks if name in FAKE_REPO]` with nothing said. A model
that hallucinates a third of its answer is indistinguishable, downstream, from
one that returned two files.

**FINDING: over-delivery is truncated just as quietly.** Five valid files
yield **3**, because the comprehension ends in `[:3]`. The prompt asked for
three; the server neither requires three nor reports that it discarded two.
And every one of these outcomes, accepted or rejected, is reported as
`-32602` — *invalid params* — so a model failure is attributed to the client
that relayed it. There is no code for "the text you were told to generate came
back wrong".

Structure: `first_round` opens the flow and `retry` replays one host answer
into it, returning either the error or the files the server kept.
"""

from __future__ import annotations

import base64
import json

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "11-mcp-sampling"
ARGUMENTS = {"audience": "developer"}
CASES = {
    "unparseable": "not json",
    "object": '{"a":1}',
    "integers": "[1,2]",
    "invented": '["nope.md"]',
    "partly_invented": '["README.md","ghost.md","server.py"]',
    "over_long": '["README.md","server.py","LICENSE","client.py","docs/intro.md"]',
}


def unseal(token):
    """The sealed body is signed, not encrypted, so it reads back directly."""
    body = token.split(".", 1)[0]
    return json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))


def first_round(ref):
    params = {"name": "summarize_repo", "arguments": dict(ARGUMENTS), "_meta": ref.request_meta()}
    return ref.dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params})


def retry(ref, token, text):
    """Replay one host answer for pick_files and report what the server made of it."""
    params = {"name": "summarize_repo", "arguments": dict(ARGUMENTS),
              "_meta": ref.request_meta(), "requestState": token,
              "inputResponses": {"pick_files": {"role": "assistant",
                                                "content": {"type": "text", "text": text}}}}
    response = ref.dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": params})
    if "error" in response:
        return {"code": response["error"]["code"], "message": response["error"]["message"]}
    return {"code": None, "picked": unseal(response["result"]["requestState"])["picked"]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    opening = first_round(ref)["result"]
    token = opening["requestState"]
    outcomes = {name: retry(ref, token, text) for name, text in CASES.items()}
    return {
        "opening": opening["resultType"], "requests": sorted(opening["inputRequests"]),
        "outcomes": outcomes,
        "codes": sorted({o["code"] for o in outcomes.values()}, key=lambda c: (c is None, c)),
        "messages": [outcomes[name]["message"] for name in
                     ("unparseable", "object", "integers", "invented")],
        "repo": len(ref.FAKE_REPO),
    }


def verify(result):
    outcomes = result["outcomes"]
    return [
        practice.Check(
            "ANSWER: invalid JSON answers -32602 before the text reaches the repo",
            all([result["opening"] == "input_required", result["requests"] == ["pick_files"],
                 outcomes["unparseable"] == {"code": -32602,
                                             "message": "pick_files must return JSON"}]),
            f"the first round returns {result['opening']!r} asking for "
            f"{result['requests']}, and replaying {CASES['unparseable']!r} answers "
            f"{outcomes['unparseable']['code']} {outcomes['unparseable']['message']!r}. The "
            "text is parsed before it is used, so unparseable output never reaches FAKE_REPO",
        ),
        practice.Check(
            "FINDING: three distinct rejections, testing parsing, shape and membership",
            all([len(set(result["messages"])) == 3,
                 outcomes["object"]["message"] == outcomes["integers"]["message"],
                 outcomes["invented"]["message"] == "pick_files returned no known files"]),
            f"the four malformed answers give {len(set(result['messages']))} distinct "
            f"messages: {sorted(set(result['messages']))}. An object and a list of integers "
            "share the shape check, while an invented filename parses and is well-shaped and "
            "fails only on membership -- structure and contents are separate gates",
        ),
        practice.Check(
            "FINDING: the membership gate is 'at least one', not 'all'",
            all([outcomes["partly_invented"]["code"] is None,
                 outcomes["partly_invented"]["picked"] == ["README.md", "server.py"]]),
            f"{CASES['partly_invented']} succeeds and yields "
            f"{outcomes['partly_invented']['picked']} -- the invented name filtered out by "
            "`if name in FAKE_REPO` with nothing said. A model that hallucinated a third of "
            "its answer is indistinguishable downstream from one that returned two files",
        ),
        practice.Check(
            "FINDING: over-delivery is truncated as quietly, and every outcome is -32602",
            all([outcomes["over_long"]["code"] is None,
                 len(outcomes["over_long"]["picked"]) == 3,
                 result["codes"] == [-32602, None], result["repo"] == 6]),
            f"five valid files out of {result['repo']} yield "
            f"{outcomes['over_long']['picked']}, three, because the comprehension ends in "
            f"[:3] -- neither required nor reported. And every rejection here is "
            f"{result['codes'][0]}, invalid params, so a model failure is attributed to the "
            "client that relayed it; there is no code for text that came back wrong",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
