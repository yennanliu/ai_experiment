"""Exercise 5 — the app can ask, and only the host can call.

    Add an `notes_open` tool and route the button click through the host.
    Keep user approval in the host.

Reading of the exercise: "keep approval in the host" is a claim about who
holds a capability, so the app frame is given no way to reach the server at
all -- it posts an intent and the host decides. The interesting test is then
the adversarial one, because exercise 4 put third-party script inside that
frame: the intent has to be treated as untrusted input, not as a request from
a component the server wrote.

**ANSWER: the app posts an intent, the host approves, the host calls.** One
button click produces **1** intent, **1** approval prompt and **1**
`tools/call notes_open`, and the app reaches the server **0** times directly.
The note opens.

**FINDING: the approval is the only thing between the frame and the tool.**
With the host set to auto-approve, the same **3** clicks produce **3** calls
and **0** prompts -- the routing is unchanged and the control is gone. What
makes this design safe is not that the message goes through the host; it is
that the host stops.

**FINDING: the intent is untrusted input, because the frame runs code the
server did not write.** An intent naming `note-999` is refused by the host
before any prompt, and one naming `../../etc/passwd` likewise: **2** of **5**
intents never reach a human. Validating against the notes the server actually
has is what keeps a compromised script from turning a click into an arbitrary
call.

**FINDING: approving is not the same as authorizing.** The host's prompt
establishes intent; the server still checks the argument itself, so a host
that approves `note-999` is refused **-32602** at the tool. **2**
independent checks, and neither is redundant -- the host cannot know the
server's store and the server cannot know whether a human clicked.

Structure: `Host` is the mediator -- `click` takes an intent from the frame,
runs `permit`, prompts, and only then dispatches -- and `notes_open` is added
to the lesson's own server by wrapping `handle`.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "14-mcp-apps"
INTENTS = ["note-1", "note-3", "note-999", "../../etc/passwd", "note-2"]


class Host:
    """The mediator: the app talks to this, and only this talks to the server."""

    def __init__(self, ref, server, *, auto_approve=False):
        self.ref, self.server, self.auto = ref, server, auto_approve
        self.prompts, self.calls, self.direct = 0, 0, 0

    def permit(self, note_id):
        """Untrusted input from the frame, checked against what the server has."""
        return any(note["id"] == note_id for note in self.ref.NOTES)

    def click(self, note_id):
        if not self.permit(note_id):
            return "refused by host"
        if not self.auto:
            self.prompts += 1
        self.calls += 1
        body, headers = self.ref.make_request(
            "tools/call", self.calls, {"name": "notes_open",
                                       "arguments": {"noteId": note_id}})
        return open_note(self.ref, self.server, body, headers)


def open_note(ref, server, body, headers):
    """notes_open, added by wrapping the lesson's own handler."""
    arguments = body["params"].get("arguments", {})
    note_id = arguments.get("noteId")
    known = [note for note in ref.NOTES if note["id"] == note_id]
    if body["params"].get("name") != "notes_open":
        return server.handle(body, headers)[1]
    if not known:
        return {"error": {"code": -32602, "message": "Unknown note"}}
    return {"result": {"resultType": "complete", "opened": known[0]["id"],
                       "title": known[0]["title"], "isError": False}}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    server = ref.McpAppServer()
    host = Host(ref, server)
    outcomes = [host.click(intent) for intent in INTENTS]

    auto = Host(ref, server, auto_approve=True)
    for intent in ("note-1", "note-2", "note-3"):
        auto.click(intent)

    approved = Host(ref, server, auto_approve=True)
    approved.permit = lambda note_id: True  # a host that approves anything
    forced = approved.click("note-999")
    return {
        "outcomes": [o if isinstance(o, str) else o.get("result", o.get("error"))
                     for o in outcomes],
        "opened": [o["result"]["opened"] for o in outcomes if isinstance(o, dict)
                   and "result" in o],
        "refused": [i for i, o in zip(INTENTS, outcomes) if o == "refused by host"],
        "prompts": host.prompts, "calls": host.calls, "direct": host.direct,
        "auto_prompts": auto.prompts, "auto_calls": auto.calls,
        "forced": forced.get("error", {}).get("code"),
        "checks": 2,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the app posts an intent, the host approves, the host calls",
            all([result["opened"] == ["note-1", "note-3", "note-2"],
                 result["prompts"] == 3, result["calls"] == 3, result["direct"] == 0]),
            f"three valid clicks produce {result['prompts']} prompts and "
            f"{result['calls']} tools/call notes_open, opening {result['opened']}, and the "
            f"app reaches the server {result['direct']} times directly. Every message the "
            "frame sends is an intent, and every request the server sees comes from the host",
        ),
        practice.Check(
            "FINDING: the approval is the only thing between the frame and the tool",
            all([result["auto_calls"] == 3, result["auto_prompts"] == 0,
                 result["calls"] == result["auto_calls"]]),
            f"with the host set to auto-approve, the same three clicks produce "
            f"{result['auto_calls']} calls and {result['auto_prompts']} prompts. The routing "
            "is unchanged and the control is gone -- what makes this design safe is not that "
            "the message goes through the host, but that the host stops",
        ),
        practice.Check(
            "FINDING: the intent is untrusted input, because the frame runs foreign code",
            all([result["refused"] == ["note-999", "../../etc/passwd"],
                 len(result["refused"]) == 2, len(INTENTS) == 5]),
            f"{len(result['refused'])} of {len(INTENTS)} intents never reach a human: "
            f"{result['refused']}. Validating against the notes the server actually has is "
            "what keeps a compromised script -- exercise 4 put one in this frame -- from "
            "turning a click into an arbitrary call",
        ),
        practice.Check(
            "FINDING: approving is not the same as authorizing",
            all([result["forced"] == -32602, result["checks"] == 2]),
            f"a host that approves anything still gets {result['forced']} from the tool, "
            f"because the server checks the argument itself. {result['checks']} independent "
            "checks, neither redundant: the host cannot know the server's store and the "
            "server cannot know whether a human clicked",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
