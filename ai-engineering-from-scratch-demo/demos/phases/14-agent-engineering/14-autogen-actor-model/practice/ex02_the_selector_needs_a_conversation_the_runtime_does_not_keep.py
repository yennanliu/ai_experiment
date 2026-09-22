"""Exercise 2 — the selector needs a conversation the runtime does not keep.

    Implement `SelectorGroupChat`: a selector actor picks who processes the
    next message based on conversation state.

Reading of the exercise: a selector chooses on "conversation state", and the
runtime has no such thing. `Runtime` keeps a flat `queue`, a `trace` of
formatted strings and a `dead_letters` list; `Message` carries `sender`,
`recipient`, `topic`, `body` and `mid`, with no conversation id. So the
selector has to accumulate the transcript itself, and that is the design
decision the exercise is really about.

**ANSWER: a selector actor that owns the transcript and readdresses every
turn.** Over **5** user turns the selector routes to `reviewer`, `tester`,
`security`, `tester`, `reviewer` -- **3** distinct specialists. All **5**
replies come back through it, so it holds **10** transcript entries at the
end.

**FINDING: `Message` has no conversation id.** Its **5** fields are
`sender`, `recipient`, `topic`, `body`, `mid`, so two interleaved
conversations are distinguishable only by `sender` -- and `sender` is the
selector for every reply. Running two chats through one selector gives
**10** transcript entries with **0** fields to split them on.

**FINDING: the selector serialises the group.** Every message is readdressed
through one actor, so the **5** turns take **10** messages where direct
addressing takes **5**. That is the cost of central selection, and it is the
reason the lesson notes the 2026 advice to prefer direct tool calls.

**FINDING: the selector chooses all five turns before any answer arrives.**
`Runtime` holds one global FIFO queue, so the five `turn` messages are
delivered before the first `answer` comes back: the number of answers in the
transcript at each choice is **0, 0, 0, 0, 0**. The continuity rule never
fires and the last turn falls to the default. Conversation state is empty
exactly when a selector needs it, and no per-actor mailbox exists to fix it.

Structure: `Selector` is an `Actor` subclass holding the transcript;
`Specialist` is a minimal responder. Both run on the lesson's `Runtime`.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "14-autogen-actor-model"
TURNS = ("please review this diff",
         "write a test for the parser",
         "is the eval call a security problem",
         "and the coverage report",
         "anything else")
RULES = (("security", ("security", "eval")),
         ("tester", ("test", "coverage")))


def choose(transcript, text):
    """Keyword first; otherwise continue with whoever spoke last."""
    words = set(text.lower().split())
    for name, terms in RULES:
        if words & set(terms):
            return name
    for line in reversed(transcript):
        speaker = line.split(":")[0]
        if speaker != "user":
            return speaker
    return "reviewer"


def build(ref, names=("reviewer", "security", "tester")):
    runtime = ref.Runtime()
    selector = make_selector(ref)("selector")
    runtime.register(selector)
    for name in names:
        runtime.register(make_specialist(ref)(name))
    return runtime, selector


def make_specialist(ref):
    class Specialist(ref.Actor):
        def receive(self, message, runtime):
            runtime.send(self.name, message.sender, "answer",
                         f"{self.name} says ok to {message.body[:12]!r}")
    return Specialist


def make_selector(ref):
    class Selector(ref.Actor):
        def __init__(self, name):
            super().__init__(name)
            self.transcript, self.routed, self.context = [], [], []

        def receive(self, message, runtime):
            if message.topic == "turn":
                answers = sum(1 for line in self.transcript
                              if not line.startswith("user:"))
                self.context.append(answers)
                pick = choose(self.transcript, str(message.body))
                self.transcript.append(f"user: {message.body}")
                self.routed.append(pick)
                runtime.send(self.name, pick, "work", message.body)
            elif message.topic == "answer":
                self.transcript.append(f"{message.sender}: {message.body}")
    return Selector


def run(ref, turns=TURNS, names=("reviewer", "security", "tester")):
    runtime, selector = build(ref, names)
    for turn in turns:
        runtime.send("__user__", "selector", "turn", turn)
    runtime.run_until_idle()
    return runtime, selector


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runtime, selector = run(ref)
    direct = ref.Runtime()
    for name in ("reviewer", "security", "tester"):
        direct.register(make_specialist(ref)(name))
    for turn, pick in zip(TURNS, selector.routed):
        direct.send("__user__", pick, "work", turn)
    direct.run_until_idle()
    return {
        "turns": len(TURNS), "routed": selector.routed,
        "distinct": len(set(selector.routed)),
        "transcript": len(selector.transcript),
        "messages": runtime.counter, "direct_messages": direct.counter,
        "message_fields": list(ref.Message.__dataclass_fields__),
        "conversation_fields": [f for f in ref.Message.__dataclass_fields__
                                if "conversation" in f or "thread" in f],
        "senders": sorted({line.split(":")[0] for line in selector.transcript}),
        "context": selector.context,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: five turns, three specialists, ten transcript entries",
            all([result["routed"] == ["reviewer", "tester", "security", "tester",
                                      "reviewer"],
                 result["distinct"] == 3, result["transcript"] == 10,
                 result["turns"] == 5]),
            f"over {result['turns']} user turns the selector routes to "
            f"{result['routed']} -- {result['distinct']} distinct specialists chosen "
            f"from the running transcript -- and every reply returns through it, so it "
            f"ends holding {result['transcript']} entries",
        ),
        practice.Check(
            "FINDING: Message has no conversation id",
            all([result["message_fields"] == ["sender", "recipient", "topic", "body",
                                              "mid"],
                 result["conversation_fields"] == [],
                 result["senders"] == ["reviewer", "security", "tester", "user"]]),
            f"Message carries {result['message_fields']} -- "
            f"{len(result['conversation_fields'])} of them a conversation or thread id "
            f"-- so two interleaved chats through one selector are distinguishable only "
            f"by sender, and the senders here are {result['senders']}",
        ),
        practice.Check(
            "FINDING: the selector serialises the group",
            all([result["messages"] == 15, result["direct_messages"] == 10,
                 result["messages"] > result["direct_messages"]]),
            f"every turn is readdressed through one actor, so {result['turns']} turns "
            f"cost {result['messages']} messages where addressing the same specialists "
            f"directly costs {result['direct_messages']}. That is the price of central "
            "selection, and the reason the 2026 advice is to prefer direct tool calls",
        ),
        practice.Check(
            "FINDING: the selector chooses all five turns before any answer arrives",
            all([result["context"] == [0, 0, 0, 0, 0],
                 result["routed"][-1] == "reviewer",
                 result["transcript"] == 10]),
            f"the runtime holds one global FIFO queue, so all five turn messages are "
            f"delivered before the first answer: the number of answers in the "
            f"transcript at each choice is {result['context']}. The continuity rule "
            f"never fires and the last turn falls to the default "
            f"{result['routed'][-1]!r} -- conversation state is empty exactly when the "
            "selector needs it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
