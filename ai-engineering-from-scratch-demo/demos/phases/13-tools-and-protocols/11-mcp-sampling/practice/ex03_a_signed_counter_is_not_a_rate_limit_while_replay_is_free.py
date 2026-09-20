"""Exercise 3 — a signed counter is not a rate limit while replay is free.

    Add a third round that asks the host to critique the summary. Carry the
    earlier summary inside signed state and cap the entire flow at three
    rounds.

Reading of the exercise: the cap is the interesting half. The server has
nowhere to keep a counter, so "cap the entire flow" can only be implemented by
putting the count in the same sealed token the client carries -- which raises
the question the exercise does not ask and this solution does: what is that
cap worth when the client chooses which token to send back?

**ANSWER: three rounds, and the summary reaches the end without being
resent.** `pick` → `summarize` → `critique`, and the final
`structuredContent` carries `picked`, `summary` and `critique` although round
three's params contain **0** of them: the summary is read out of the verified
state, so the server trusts it because it signed it.

**FINDING: the cap lives in the token because there is nowhere else, and that
makes it resettable.** Replaying the round-1 token **5** times yields **5**
fresh round-2 tokens, each at `round: 2`. The counter is monotonic within a
chain and the client holds every link, so nothing stops it starting again from
an old one. A signed counter bounds a *chain*, not a flow — bounding the flow
needs the tokens to be single-use, which needs the server state the lesson
removed.

**FINDING: each round refreshes the deadline, so the flow outlives its own
expiry.** `expiresAt` is re-set to `now + 300` in every re-seal, so the final
token's deadline is later than the first's. A three-round flow has no overall
time limit; it has three consecutive ones.

**FINDING: the cap is checked after the whole verification runs.** Lowering
the limit to 2 refuses the third round with `round cap exceeded` — but only
after the HMAC, the principal, the method, the digest and the expiry have all
been checked. Refusing late costs the server the full verification for every
replayed token, which is the work an attacker wanted done.

Structure: `Flow` is the lesson's `tools_call` with a `critique` phase and a
`round` field; `host` answers whichever key it is asked for.
"""

from __future__ import annotations

import base64
import json
import time

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "11-mcp-sampling"
ARGUMENTS = {"audience": "developer"}
PRINCIPAL = "user-42"
MAX_ROUNDS = 3
ANSWERS = {"pick_files": '["README.md","server.py"]',
           "summary": "A stateless MCP server with a retry loop.",
           "critique": "Accurate, but it omits the sealed state."}


def unseal(token):
    body = token.split(".", 1)[0]
    return json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))


def host(pending):
    """The host model: answer whichever input request arrived."""
    key = next(iter(pending["inputRequests"]))
    return {key: {"role": "assistant", "content": {"type": "text", "text": ANSWERS[key]}}}


class Flow:
    """The lesson's tool with a third round, a carried summary and a cap."""

    def __init__(self, ref, max_rounds=MAX_ROUNDS):
        self.ref, self.max_rounds = ref, max_rounds

    def open(self, arguments=None):
        state = {"phase": "pick", "principal": PRINCIPAL, "method": "tools/call",
                 "argumentsDigest": self.ref._arguments_digest(arguments or ARGUMENTS),
                 "round": 1, "expiresAt": int(time.time()) + 300}
        return self.ref._input_required("pick_files", "Choose files.", state, intelligence=0.2)

    def resume(self, token, responses, arguments=None):
        state = self.ref.verify_request_state(token, principal=PRINCIPAL,
                                              arguments=arguments or ARGUMENTS)
        rounds = state["round"] + 1
        # the cap bounds rounds of model input, and the critique round issues no token
        if state["phase"] != "critique" and rounds > self.max_rounds:
            raise self.ref.McpError(-32602, "round cap exceeded", {"round": rounds})
        if state["phase"] == "pick":
            picks = json.loads(self.ref._sampling_text(responses, "pick_files"))
            picked = [n for n in picks if n in self.ref.FAKE_REPO][:3]
            following = {**state, "phase": "summarize", "picked": picked, "round": rounds,
                         "expiresAt": int(time.time()) + 300}
            return self.ref._input_required("summary", "Summarize " + ", ".join(picked),
                                            following, intelligence=0.8)
        if state["phase"] == "summarize":
            summary = self.ref._sampling_text(responses, "summary")
            following = {**state, "phase": "critique", "summary": summary, "round": rounds,
                         "expiresAt": int(time.time()) + 300}
            return self.ref._input_required("critique", "Critique:\n\n" + summary,
                                            following, intelligence=0.9)
        critique = self.ref._sampling_text(responses, "critique")
        whole = {"picked": state["picked"], "summary": state["summary"], "critique": critique}
        return self.ref.complete(content=[{"type": "text", "text": critique}],
                                 isError=False, structuredContent=whole)


def drive(flow):
    """Run one flow to completion, keeping every token it handed out."""
    result, tokens, sent = flow.open(), [], []
    while result["resultType"] == "input_required":
        tokens.append(result["requestState"])
        responses = host(result)
        sent.append(sorted(responses))
        result = flow.resume(result["requestState"], responses)
    return result, tokens, sent


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    final, tokens, sent = drive(Flow(ref))
    picked = host({"inputRequests": {"pick_files": {}}})
    replays = [unseal(Flow(ref).resume(tokens[0], picked)["requestState"])["round"]
               for _ in range(5)]
    capped, refusal = Flow(ref, max_rounds=2), None
    result = capped.open()
    try:
        while result["resultType"] == "input_required":
            result = capped.resume(result["requestState"], host(result))
    except ref.McpError as exc:
        refusal = (exc.code, exc.message, exc.data)
    return {
        "rounds": len(tokens), "sent": sent,
        "phases": [unseal(t)["phase"] for t in tokens],
        "counts": [unseal(t)["round"] for t in tokens],
        "structured": sorted(final["structuredContent"]),
        "summary": final["structuredContent"]["summary"],
        "summary_in_state": unseal(tokens[2])["summary"],
        "replays": replays,
        "deadlines_grow": unseal(tokens[-1])["expiresAt"] >= unseal(tokens[0])["expiresAt"],
        "refusal": refusal,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: three rounds, and the summary reaches the end without being resent",
            all([result["rounds"] == 3, result["counts"] == [1, 2, 3],
                 result["phases"] == ["pick", "summarize", "critique"],
                 result["structured"] == ["critique", "picked", "summary"],
                 result["sent"][2] == ["critique"],
                 result["summary_in_state"] == result["summary"]]),
            f"the flow runs {result['phases']} at rounds {result['counts']} and the final "
            f"structuredContent carries {result['structured']}, although round three sends "
            f"only {result['sent'][2]}. The summary is read out of the verified state, so the "
            "server trusts it because it signed it",
        ),
        practice.Check(
            "FINDING: the cap lives in the token because there is nowhere else, so replay resets it",
            all([result["replays"] == [2] * 5, len(result["replays"]) == 5]),
            f"replaying the round-1 token {len(result['replays'])} times yields as many "
            f"fresh tokens, at rounds {result['replays']}. The client holds every link, so a "
            "signed counter bounds a chain and not a flow -- bounding the flow needs the "
            "single-use tokens that need the server state the lesson removed",
        ),
        practice.Check(
            "FINDING: each round refreshes the deadline, so the flow outlives its own expiry",
            result["deadlines_grow"],
            "expiresAt is re-set to now + 300 in every re-seal, so the last token's deadline "
            "is no earlier than the first's -- a three-round flow has no overall time limit, "
            "it has three consecutive ones",
        ),
        practice.Check(
            "FINDING: the cap is checked after the whole verification runs",
            all([result["refusal"][0] == -32602,
                 result["refusal"][1] == "round cap exceeded",
                 result["refusal"][2] == {"round": 3}]),
            f"lowering the limit to 2 refuses the third round with {result['refusal'][1]!r} "
            f"at {result['refusal'][2]}, but only after the HMAC, the principal, the method, "
            "the digest and the expiry are all checked -- so a replayed token costs the "
            "server the full verification, the work an attacker wanted",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
