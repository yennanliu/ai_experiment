"""Exercise 1 — the round trip has no return leg.

    Run `code/main.py`. Observe the round-trip encoding. Identify which FIPA
    performative corresponds to `tools/call`, `resources/read`, and A2A task
    creation.

Reading of the exercise: answer the mapping question, then check the word the
question rests on. "Round-trip" asserts that the encoding can be undone, and
the module that prints the banner "Round-trip: 2026 JSON-RPC / REST <-> FIPA-ACL
envelope" contains no function that goes the other way.

**ANSWER: `request`, `query-ref`, and `request` again.** `tools/call` becomes
`request` because invoking a tool asks the receiver to act. `resources/read`
becomes `query-ref` because it asks for the referent of an expression rather
than for an action. A2A task creation becomes `request`. So **3** message
types map onto **2** performatives, and the mapping is not injective: what
separates an MCP tool call from an A2A task is the `:protocol` and `:ontology`
slots, not the performative.

**FINDING: the round trip is one-way.** The module defines **4** functions
named `*_to_acl` and **0** named `acl_to_*`. Nothing decodes. The banner says
`<->` and the arrow only runs left to right, so the claim a reader is told to
"observe" is not implemented.

**FINDING: the envelope declares a language it does not emit.** The MCP
tool-call message sets `:language JSON` and `render()` writes `:content` with
`{self.content!r}`, which is Python's repr. The rendered line is
`{'symbol': 'IBM'}` -- single quotes -- and `json.loads` raises
**JSONDecodeError** on it. A decoder could not be written against this output
even if someone wanted to, which is the mechanical reason the return leg is
missing.

**FINDING: nine of the sixteen performatives are declared and never built.**
`PERFORMATIVES` holds **16** names and `__post_init__` rejects anything else.
The module constructs **7**. The remaining **9** -- agree, cancel, confirm,
disconfirm, failure, inform, not-understood, query-if, refuse -- exist only as
a whitelist, which is why the next exercise can ask the reader to add one that
is already there.

Structure: `encoders()` counts the direction functions; `rendered()` pulls the
content line back out of a rendered envelope.
"""

from __future__ import annotations

import inspect
import json
import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "02-fipa-acl-heritage"
MCP_CALL = {"jsonrpc": "2.0", "method": "tools/call",
            "params": {"name": "lookup_stock", "arguments": {"symbol": "IBM"}}, "id": 42}
MCP_READ = {"jsonrpc": "2.0", "method": "resources/read",
            "params": {"uri": "file:///etc/hosts"}, "id": 43}
A2A_TASK = {"client": "research-host", "agent": "code-review-agent",
            "skill": "review-python", "input": "def f(x): return x", "task_id": "t-12"}


def encoders(src):
    """Functions that go JSON-RPC to ACL, and functions that go back."""
    return (len(re.findall(r"(?m)^def \w+_to_acl\(", src)),
            len(re.findall(r"(?m)^def acl_to_\w+\(", src)))


def rendered(message):
    """The `:content` line of a rendered envelope, and whether it parses as JSON."""
    line = next(l for l in message.render().splitlines() if ":content" in l)
    payload = line.split(":content", 1)[1].strip()
    try:
        json.loads(payload)
    except json.JSONDecodeError as exc:
        return payload, type(exc).__name__
    return payload, None


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    src = inspect.getsource(ref)
    call = ref.mcp_tools_call_to_acl(MCP_CALL)
    mapping = {"tools/call": call.performative,
               "resources/read": ref.mcp_resources_read_to_acl(MCP_READ).performative,
               "a2a-task": ref.a2a_task_create_to_acl(A2A_TASK).performative}
    payload, failure = rendered(call)
    to_acl, from_acl = encoders(src)
    built = set(re.findall(r'performative="([a-z-]+)"', src))
    return {
        "mapping": mapping,
        "distinct": sorted(set(mapping.values())),
        "to_acl": to_acl, "from_acl": from_acl,
        "banner": "<->" in src,
        "language": call.language,
        "payload": payload, "json_failure": failure,
        "declared": len(ref.PERFORMATIVES),
        "built": len(built),
        "unused": sorted(ref.PERFORMATIVES - built),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: request, query-ref, and request again",
            all([result["mapping"]["tools/call"] == "request",
                 result["mapping"]["resources/read"] == "query-ref",
                 result["mapping"]["a2a-task"] == "request",
                 result["distinct"] == ["query-ref", "request"]]),
            f"tools/call -> {result['mapping']['tools/call']}, resources/read -> "
            f"{result['mapping']['resources/read']}, A2A task creation -> "
            f"{result['mapping']['a2a-task']}: {len(result['mapping'])} message types "
            f"onto {len(result['distinct'])} performatives, so what separates a tool "
            "call from a task is :protocol and :ontology, not the performative",
        ),
        practice.Check(
            "FINDING: the round trip is one-way",
            all([result["to_acl"] == 4, result["from_acl"] == 0, result["banner"]]),
            f"the module defines {result['to_acl']} functions named *_to_acl and "
            f"{result['from_acl']} named acl_to_*, under a banner that writes the arrow "
            "as <-> -- nothing decodes, so the encoding a reader is told to observe as a "
            "round trip has no return leg",
        ),
        practice.Check(
            "FINDING: the envelope declares a language it does not emit",
            all([result["language"] == "JSON", result["json_failure"] == "JSONDecodeError",
                 result["payload"] == "{'symbol': 'IBM'}"]),
            f"the message sets :language {result['language']} and render() writes "
            f"{{self.content!r}}, so the line is {result['payload']} and json.loads "
            f"raises {result['json_failure']} -- a decoder could not be written against "
            "this output, which is the mechanical reason there is none",
        ),
        practice.Check(
            "FINDING: nine of the sixteen performatives are declared and never built",
            all([result["declared"] == 16, result["built"] == 7,
                 len(result["unused"]) == 9, "cancel" in result["unused"]]),
            f"PERFORMATIVES holds {result['declared']} names that __post_init__ enforces "
            f"and the module constructs {result['built']}; the other "
            f"{len(result['unused'])} ({', '.join(result['unused'])}) exist only as a "
            "whitelist, which is why the next exercise asks for one already there",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
