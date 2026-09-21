"""Exercise 2 — there is no history to nest, only the last string.

    Implement `nest_handoff_history` as an option — collapse prior messages
    into one summary before transferring.

Reading of the exercise: the option presupposes a transcript, and `Runner`
keeps `current_input`, a single string that each step overwrites. A tool
result becomes `tool {name} returned: {result}` and the user's question is
gone; a handoff passes `policy_output.get("input", current_input)`, which is
whatever the policy chose. So nesting has to *build* the history first, and
the measurement is how much the shipped runner had already thrown away.

**ANSWER: a runner that accumulates turns and collapses them on transfer.**
Over a chain with **2** tool calls and **1** handoff, the accumulating runner
carries **3** turns into the transfer -- the question and both tool results
-- and nests them into **1** summary of **68** characters, against **1** turn
and **41** characters in the shipped runner, which is the last tool's output
and nothing before it.

**FINDING: the shipped handoff drops the user's question.** After two tool
calls the value passed to the target agent contains the user's words **0**
times and the last tool's result **1** time. The specialist that was
transferred to is told what the previous agent's tool said, not what the user
asked.

**FINDING: nesting is a *reduction*, and the shipped runner has nothing to
reduce.** Summarising **3** turns to **1** is a **67%** drop in entries and
takes **110** characters to **68**; the shipped runner already carries
**1**, so the option is a no-op there. The saving the exercise is after
exists only once the history does.

**FINDING: `input` is an escape hatch that bypasses both.** A policy may
return any `input` it likes on a handoff, so an agent can hand over a string
that appears in neither the history nor the tool results -- here a **13**-
character summary the runner cannot check. Nesting is a runner policy that a
single policy return value overrides.

Structure: `accumulating()` is `Runner.run` with a turn list; `nest()` is the
collapse; the shipped runner is measured beside it.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "16-openai-agents-sdk"
QUESTION = "I need a refund for invoice 4711 please"


def build(ref, override=None):
    billing = ref.Agent("billing", "handle refunds",
                        lambda text: {"kind": "final", "text": f"billing saw: {text}"})
    calls = {"n": 0}

    def policy(text):
        calls["n"] += 1
        if calls["n"] <= 2:
            return {"kind": "tool", "tool": f"lookup{calls['n']}",
                    "args": {"key": "4711"}}
        decision = {"kind": "handoff", "to": "billing"}
        if override is not None:
            decision["input"] = override
        return decision

    triage = ref.Agent(
        "triage", "route", policy,
        tools=[ref.FunctionTool("lookup1", "first", lambda key: f"invoice {key} open"),
               ref.FunctionTool("lookup2", "second", lambda key: f"balance {key} is 40")],
        handoffs=[ref.Handoff(target=billing)])
    return triage


def nest(turns):
    """The option: one summary line instead of the whole transcript."""
    return "summary of " + "; ".join(turn.split(": ", 1)[-1][:18] for turn in turns)


def accumulating(ref, agent, text, nested):
    """Runner.run with a transcript, so there is something to collapse."""
    current, message, turns = agent, text, [f"user: {text}"]
    handed = None
    for _ in range(5):
        decision = current.policy(message)
        if decision["kind"] == "final":
            return decision["text"], turns, handed
        if decision["kind"] == "tool":
            tool = next(t for t in current.tools if t.name == decision["tool"])
            result = tool.fn(**decision["args"])
            turns.append(f"tool {tool.name}: {result}")
            message = f"tool {tool.name} returned: {result}"
            continue
        handoff = next(h for h in current.handoffs if h.target.name == decision["to"])
        handed = nest(turns) if nested else "\n".join(turns)
        current, message = handoff.target, decision.get("input", handed)
    return "", turns, handed


def shipped_handoff_input(ref, override=None):
    """What the shipped runner hands to the target agent."""
    runner = ref.Runner(max_hops=5, trace=ref.Span(name="run"))
    return runner.run(build(ref, override), QUESTION)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    flat, turns, nested_payload = accumulating(ref, build(ref), QUESTION, True)
    _, _, raw_payload = accumulating(ref, build(ref), QUESTION, False)
    shipped = shipped_handoff_input(ref)
    carried = shipped.split("billing saw: ", 1)[1]
    escaped = shipped_handoff_input(ref, override="just a refund")
    return {
        "turns": len(turns), "nested": nested_payload, "nested_len": len(nested_payload),
        "raw_len": len(raw_payload),
        "shipped_carried": carried, "shipped_len": len(carried),
        "shipped_turns": 1,
        "has_question": carried.count("invoice 4711 please"),
        "has_tool": carried.count("balance 4711 is 40"),
        "reduction": round(1 - 1 / len(turns), 2),
        "override": escaped.split("billing saw: ", 1)[1],
        "override_len": len(escaped.split("billing saw: ", 1)[1]),
        "final": flat,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: three turns nested into one, against the shipped single string",
            all([result["turns"] == 3, result["nested_len"] == 68,
                 result["shipped_len"] == 41, result["shipped_turns"] == 1,
                 result["nested"].startswith("summary of ")]),
            f"the accumulating runner carries {result['turns']} turns into the transfer "
            f"and nests them into one summary of {result['nested_len']} characters, "
            f"against {result['shipped_turns']} turn and {result['shipped_len']} "
            f"characters in the shipped runner: {result['shipped_carried']!r}",
        ),
        practice.Check(
            "FINDING: the shipped handoff drops the user's question",
            all([result["has_question"] == 0, result["has_tool"] == 1]),
            f"the value passed to the target agent contains the user's words "
            f"{result['has_question']} times and the last tool's result "
            f"{result['has_tool']} time. The specialist is told what the previous "
            "agent's tool said, not what the user asked",
        ),
        practice.Check(
            "FINDING: nesting is a reduction, and the shipped runner has nothing to cut",
            all([result["reduction"] == 0.67, result["raw_len"] > result["nested_len"],
                 result["shipped_turns"] == 1]),
            f"summarising {result['turns']} turns to one is a "
            f"{result['reduction']:.0%} drop in entries and takes "
            f"{result['raw_len']} characters to {result['nested_len']}. The shipped "
            f"runner already carries {result['shipped_turns']}, so the option is a "
            "no-op until the history exists",
        ),
        practice.Check(
            "FINDING: `input` is an escape hatch that bypasses both",
            all([result["override"] == "just a refund", result["override_len"] == 13,
                 result["override_len"] < result["shipped_len"]]),
            f"a policy may return any input it likes on a handoff, so the target can be "
            f"handed {result['override']!r} -- {result['override_len']} characters "
            "appearing in neither the history nor the tool results. Nesting is a runner "
            "policy that one policy return value overrides",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
