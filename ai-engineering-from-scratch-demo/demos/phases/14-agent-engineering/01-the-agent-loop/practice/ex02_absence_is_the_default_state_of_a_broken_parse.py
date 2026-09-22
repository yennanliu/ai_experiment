"""Exercise 2 — absence is the default state of a broken parse.

    Implement a `no_tool_calls → done` stop path. Contrast with `finish` as an
    explicit tool. Which is safer against early-termination bugs?

Reading of the exercise: the two designs only differ on one input -- the reply
the adapter could not read a call out of. Both are built here and fed the same
six replies, one of which is empty, so the contrast is a measurement rather
than an argument. "Safer" is then decided by which design treats the empty
reply as a decision and which treats it as a mistake.

**ANSWER: both stop paths, run over the same six replies.** Under
`no_tool_calls -> done` the run stops at the empty reply having dispatched
**3** of **5** real calls and returns `''` as the final answer. Under
`finish`-as-a-tool the same empty reply dispatches nothing the registry knows,
comes back as `error: unknown tool ''`, and the run continues to dispatch all
**5** and finish with `138.0`.

**FINDING: `finish`-as-a-tool is the safer one, because absence is the default
state of a broken parse.** Every failure mode that produces no parsed call --
a truncated stream, a schema the adapter does not recognise, a refusal -- is
indistinguishable from a deliberate stop under the first design. Under the
second, stopping requires a positive act that names a registered tool, and
**1** of the **6** replies is rejected instead of honoured.

**FINDING: the shipped loop's `finish` is neither of the two.** It is a third
reply kind, so `tools.names()` holds **3** entries and `finish` is not one of
them -- `ToolRegistry.dispatch` can never produce a final answer. It is also
the only place the loop indexes instead of `.get`-ing: a finish reply with no
`content` raises `KeyError`, while an action reply with no `args` is fine.

**FINDING: three terminations, one turn kind.** A real finish, an exhausted
script and an exhausted budget all append `Turn(kind='final')`. The contents
differ -- `the total including 15% tax is 138.0`, `no more actions`, `budget
exhausted` -- but the *kind* does not, so any caller that branches on the kind
sees one outcome where there are three.

Structure: `NoCallLLM` and `FinishToolLLM` are the two designs; both drive the
lesson's own `AgentLoop` unmodified.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "01-the-agent-loop"
REPLIES = (
    {"action": "kv_set", "args": {"key": "base", "value": "120"}},
    {"action": "calculator", "args": {"expr": "120 * 0.15"}},
    {"action": "kv_set", "args": {"key": "tax", "value": "18.0"}},
    {},                                      # nothing parsed out of this turn
    {"action": "calculator", "args": {"expr": "120 + 18.0"}},
    {"action": "finish", "args": {"answer": "138.0"}},
)


class Scripted:
    """Both designs at once; they differ only on what an unparsed reply means."""

    def __init__(self, replies, stop_on_absence):
        self.replies, self.cursor = list(replies), 0
        self.stop_on_absence, self.answer = stop_on_absence, None

    def respond(self, history):
        if self.answer is not None:
            return {"kind": "finish", "content": self.answer}
        reply = self.replies[self.cursor]
        self.cursor += 1
        if self.stop_on_absence and "action" not in reply:
            return {"kind": "finish", "content": ""}
        return {"kind": "action", "thought": "", "action": reply.get("action", ""),
                "args": reply.get("args", {})}

    def finish(self, answer):
        """`finish` as an explicit tool: stopping has to be asked for by name."""
        self.answer = answer
        return f"final: {answer}"


class Bare:
    """A finish reply with no content -- the shape the loop indexes into."""

    def respond(self, history):
        return {"kind": "finish"}


def registry(ref, finisher=None):
    tools, store = ref.ToolRegistry(), ref.KVStore()
    tools.register("calculator", ref.calculator)
    tools.register("kv_set", store.set)
    tools.register("kv_get", store.get)
    if finisher is not None:
        tools.register("finish", finisher)
    return tools


def drive(ref, llm, tools, message="total 120 plus 15% tax", turns=12):
    loop = ref.AgentLoop(llm=llm, tools=tools, max_turns=turns)
    final = loop.run(message)
    return final, [t.observation for t in loop.history if t.kind == "action"], loop


def terminations(ref):
    """A real finish, an exhausted script, an exhausted budget."""
    spin = [{"kind": "action", "action": "calculator", "args": {"expr": "1+1"}}] * 3
    loops = [ref.build_demo_agent(),
             ref.AgentLoop(llm=ref.ToyLLM([]), tools=registry(ref), max_turns=5),
             ref.AgentLoop(llm=ref.ToyLLM(spin), tools=registry(ref), max_turns=2)]
    for loop in loops:
        loop.run("go")
    return [(loop.history[-1].kind, loop.history[-1].content) for loop in loops]


def keyerror(ref):
    try:
        drive(ref, Bare(), registry(ref))
    except KeyError as exc:
        return str(exc)
    return "no error"


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    early_final, early_seen, _ = drive(ref, Scripted(REPLIES, True), registry(ref))
    finisher = Scripted(REPLIES, False)
    late_final, late_seen, _ = drive(ref, finisher, registry(ref, finisher.finish))
    ends = terminations(ref)
    return {
        "early_dispatched": len(early_seen), "early_final": early_final,
        "late_dispatched": len(late_seen), "late_final": late_final,
        "late_errors": [o for o in late_seen if o.startswith("error:")],
        "tool_names": registry(ref).names(), "keyerror": keyerror(ref),
        "end_kinds": sorted({kind for kind, _ in ends}),
        "end_contents": [content for _, content in ends],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the same six replies, two stop paths, two different endings",
            all([result["early_dispatched"] == 3, result["early_final"] == "",
                 result["late_dispatched"] == 6, result["late_final"] == "138.0"]),
            f"no_tool_calls -> done dispatches {result['early_dispatched']} and returns "
            f"{result['early_final']!r}; finish-as-a-tool dispatches "
            f"{result['late_dispatched']}, the empty reply among them, and returns "
            f"{result['late_final']!r}. One unparsed reply decides the whole run",
        ),
        practice.Check(
            "FINDING: finish-as-a-tool is safer because absence is the default failure",
            all([len(result["late_errors"]) == 1,
                 result["late_errors"][0].startswith("error: unknown tool"),
                 result["early_dispatched"] < result["late_dispatched"] - 1]),
            f"the unparsed reply comes back as {result['late_errors'][0]!r} and the run "
            "continues. A truncated stream, an unknown schema and a refusal all produce "
            "no parsed call, so each reads as a deliberate stop under the first design",
        ),
        practice.Check(
            "FINDING: the shipped finish is a reply kind, not a tool, and it indexes",
            all(["finish" not in result["tool_names"], len(result["tool_names"]) == 3,
                 result["keyerror"] == "'content'"]),
            f"the registry holds {result['tool_names']}, so dispatch has no path to a "
            f"final answer, and a finish reply with no content raises KeyError "
            f"{result['keyerror']} where an action reply with no args is read through "
            "reply.get('args', {}) -- the stop path is the one place the loop indexes",
        ),
        practice.Check(
            "FINDING: three terminations arrive as one turn kind",
            all([result["end_kinds"] == ["final"], len(result["end_contents"]) == 3,
                 len(set(result["end_contents"])) == 3,
                 "budget exhausted" in result["end_contents"]]),
            f"a real finish, an exhausted script and an exhausted budget all append "
            f"kind={result['end_kinds']}, differing only in content: "
            f"{result['end_contents']}. Anything branching on the kind sees one outcome "
            "where there are three, two of them failures",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
