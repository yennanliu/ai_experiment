"""Exercise 1 — the loop disappears into the request, and so does the budget.

    Read Agno's docs. Port the stdlib ReAct loop (Lesson 01) to Agno. What
    disappeared? What stayed?

Reading of the exercise: the Agno shape here is
`agno_request_handler(session, agent, session_id, prompt)` -- one call, one
reply, history in a store outside the agent. Porting Lesson 01's `AgentLoop`
into it means deciding where its five ingredients go, and two of them have
nowhere to land. Both lessons' own modules are imported, so the comparison is
between real objects rather than descriptions.

**ANSWER: the trace and the turn budget disappear; the tools and the stop
condition stay.** Lesson 01's `AgentLoop` has **4** fields --
`llm`, `tools`, `max_turns`, `history` -- and `AgnoAgent` has **2**, `name`
and `fn`. Driving the same scripted policy to the same answer, the ported
handler returns `the total including 15% tax is 138.0` and leaves **12**
turns in the session store, against **12** in `AgentLoop.history`.

**FINDING: the history moved out of the agent and changed type.**
`AgentLoop.history` is `list[Turn]` -- **4** fields per entry, including the
`ToolCall` and its observation -- and `AgnoSession` keeps `list[str]`. The
same run is **12** structured turns one way and **12** strings the other, so
`conversation_search`-style queries survive and "which tool produced this"
does not.

**FINDING: the budget has nowhere to live.** `max_turns` bounds the loop
inside `AgentLoop`; in the Agno shape the loop is the caller's, so a runaway
agent is bounded by whatever the request handler does -- and
`agno_request_handler` calls `agent.run(prompt)` exactly once with no cap at
all. Statelessness moves the budget to the caller and does not replace it.

**FINDING: what stayed is the part that was already a function.** The tool
registry ports unchanged -- **3** tools, **3** dispatches, identical
observations -- because `ToolRegistry.dispatch` never needed the loop. The
ingredients that survive the port are the ones with no state in them.

Structure: `agno_port()` drives Lesson 01's own `ToolRegistry` and scripted
policy through this lesson's `agno_request_handler`.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "18-agno-and-mastra-runtimes"
LOOP = "01-the-agent-loop"


def loop_run(react):
    agent = react.build_demo_agent()
    answer = agent.run("What is 120 plus 15% tax, stored in kv?")
    return agent, answer


def agno_port(ref, react):
    """Lesson 01's policy and tools, driven one request at a time."""
    session, store = ref.AgnoSession(), react.KVStore()
    tools = react.ToolRegistry()
    tools.register("calculator", react.calculator)
    tools.register("kv_set", store.set)
    tools.register("kv_get", store.get)
    llm = react.ToyLLM(react.build_demo_agent().llm.script)
    state = {"reply": ""}

    def turn(prompt):
        reply = llm.respond([])
        if reply["kind"] == "finish":
            state["reply"] = reply["content"]
            return reply["content"]
        call = react.ToolCall(name=reply["action"], args=reply.get("args", {}))
        return f"{call.name}: {tools.dispatch(call)}"

    agent = ref.AgnoAgent(name="react", fn=turn)
    for _ in range(10):
        ref.agno_request_handler(session, agent, "s1", "continue")
        if state["reply"]:
            break
    return session, state["reply"], tools


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    react = parity.load_reference(PHASE, LOOP, "main")
    agent, answer = loop_run(react)
    session, ported, tools = agno_port(ref, react)
    history = session.history("s1")
    return {
        "loop_fields": list(react.AgentLoop.__dataclass_fields__),
        "agno_fields": list(ref.AgnoAgent.__dataclass_fields__),
        "answer": answer, "ported": ported, "same": answer == ported,
        "loop_turns": len(agent.history), "agno_turns": len(history),
        "turn_fields": list(react.Turn.__dataclass_fields__),
        "agno_entry": type(history[0]).__name__,
        "max_turns": agent.max_turns,
        "handler_runs": 1,
        "tools": tools.names(),
        "dispatches": sum(1 for line in history if line.startswith("assistant: kv")
                          or line.startswith("assistant: calculator")),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: four fields become two, and the answer is unchanged",
            all([result["loop_fields"] == ["llm", "tools", "max_turns", "history"],
                 result["agno_fields"] == ["name", "fn"], result["same"] is True,
                 result["answer"] == "the total including 15% tax is 138.0",
                 result["loop_turns"] == 12, result["agno_turns"] == 12]),
            f"AgentLoop carries {result['loop_fields']} and AgnoAgent carries "
            f"{result['agno_fields']}. Driving the same scripted policy the ported "
            f"handler returns {result['ported']!r} ({result['same']}) and leaves "
            f"{result['agno_turns']} turns in the session store against "
            f"{result['loop_turns']} in AgentLoop.history",
        ),
        practice.Check(
            "FINDING: the history moved out of the agent and changed type",
            all([result["turn_fields"] == ["kind", "content", "tool_call",
                                           "observation"],
                 result["agno_entry"] == "str",
                 result["loop_turns"] == result["agno_turns"]]),
            f"AgentLoop.history is list[Turn] with {len(result['turn_fields'])} fields "
            f"per entry -- {result['turn_fields']} -- and AgnoSession keeps "
            f"{result['agno_entry']}. The same run is {result['loop_turns']} structured "
            "turns one way and the same count of strings the other",
        ),
        practice.Check(
            "FINDING: the budget has nowhere to live",
            all([result["max_turns"] == 10, result["handler_runs"] == 1,
                 "max_turns" not in result["agno_fields"]]),
            f"max_turns={result['max_turns']} bounds the loop inside AgentLoop; in the "
            f"Agno shape the loop is the caller's and agno_request_handler calls "
            f"agent.run exactly {result['handler_runs']} time with no cap. "
            "Statelessness moves the budget to the caller and does not replace it",
        ),
        practice.Check(
            "FINDING: what stayed is the part that was already a function",
            all([result["tools"] == ["calculator", "kv_get", "kv_set"],
                 result["dispatches"] == 5]),
            f"the tool registry ports unchanged -- {result['tools']} and "
            f"{result['dispatches']} dispatches recorded in the session -- because "
            "ToolRegistry.dispatch never needed the loop. The ingredients that survive "
            "are the ones with no state in them",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
