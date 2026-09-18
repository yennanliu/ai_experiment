"""Exercise 1 — ten checkpoints against a threshold of four.

    **Easy.** Implement the four-node ReAct graph above with a calculator tool
    and a web-search tool. Verify that `list(app.get_state_history(config))`
    returns at least four checkpoints for a two-turn conversation.

Reading of the exercise: the graph is wired exactly as the lesson wires it --
`agent` -> `tools` -> `agent`, `add_messages` as the reducer, a `MemorySaver`
and `interrupt_before=["tools"]` -- but the agent node returns a scripted
`AIMessage` instead of calling `ChatAnthropic`, because the exercise's claim is
about checkpoints and there is no API key here. The lesson's own `calculator`
and `web_lookup` are the tools, unmodified, and they run for real.

**ANSWER: a two-turn conversation leaves 10 checkpoints, and the threshold is
four.** The history grows 3, 5, 8, 10 across the four `stream` calls the two
turns actually need, so "at least four" is satisfied before the first turn has
even finished. A test written to that threshold passes whether the graph
checkpoints once per node or once per turn.

**FINDING: "four-node" counts the two nodes LangGraph adds.**
`graph.add_node` is called twice, for `agent` and `tools`; the compiled graph
reports `['__end__', '__start__', 'agent', 'tools']` and four edges, two of them
conditional. The number in the exercise is the number after compilation, not
the number you write.

**FINDING: two turns take four `stream` calls.** `interrupt_before=["tools"]`
pauses before every tool call, so each turn is a stream that ends in an
`__interrupt__` event with `next == ('tools',)` and a second stream carrying
`Command(resume=True)`. Exactly 2 of the 10 snapshots are paused at `tools`,
which is the count a test should assert -- it is a property of the interrupt,
not of the traffic.

**FINDING: `web_lookup` cannot answer the question the lesson asks it.** The
facts dict is keyed on `"anthropic headquarters"` and matched by exact equality
after `strip().lower()`, so the demo's own "Where is Anthropic headquartered?"
returns `"unknown"`. The tool works only if the model emits the dictionary key
verbatim, and the lesson's prompt never says what the keys are.

**FINDING: the calculator's allow-list admits exponentiation.**
`set(expression) <= set("0123456789+-*/(). ")` passes `"9**9**9"`, because `*`
is in the list and `**` is two of them -- so a tool advertised as "digits and
+ - * / ( )" will accept an expression that does not terminate. `__import__`
*is* blocked, and `1/0` returns `ERROR: ZeroDivisionError(...)` rather than
raising, so the two guards that exist work; the gap is the one the character
class cannot see.

Structure: `SCRIPT` is the four canned agent replies, `build` wires the
lesson's graph with a scripted agent node, `drive` runs one turn to its
interrupt and resumes it, and `stages` records the checkpoint count after each
`stream` call.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "16-langgraph-state-machines"
ALLOWED = "0123456789+-*/(). "
QUESTIONS = ("Where is Anthropic headquartered?", "What is 17 * 23?")
SCRIPT = [
    AIMessage(content="", tool_calls=[{"name": "web_lookup", "id": "c1",
                                       "args": {"query": "anthropic headquarters"}}]),
    AIMessage(content="Anthropic is headquartered in San Francisco."),
    AIMessage(content="", tool_calls=[{"name": "calculator", "id": "c2",
                                       "args": {"expression": "17 * 23"}}]),
    AIMessage(content="17 * 23 = 391."),
]


def build(ref, script):
    """The lesson's graph, with the model replaced by the next scripted reply."""
    step = {"index": 0}

    def agent_node(state):
        reply = script[step["index"] % len(script)]
        step["index"] += 1
        return {"messages": [reply]}

    def should_continue(state):
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    graph = StateGraph(ref.State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ref.TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=MemorySaver(), interrupt_before=["tools"])


def drive(app, config, question):
    """One turn: stream to the interrupt, then resume past it."""
    opened = list(app.stream({"messages": [HumanMessage(question)]}, config,
                             stream_mode="updates"))
    paused = app.get_state(config)
    at_pause = len(list(app.get_state_history(config)))
    closed = list(app.stream(Command(resume=True), config, stream_mode="updates"))
    return {"events": [next(iter(event)) for event in opened + closed],
            "next": paused.next, "checkpoints_at_pause": at_pause}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    app = build(ref, SCRIPT)
    config = {"configurable": {"thread_id": "practice"}}
    stages, turns = [], []
    for question in QUESTIONS:
        turn = drive(app, config, question)
        stages += [turn["checkpoints_at_pause"], len(list(app.get_state_history(config)))]
        turns.append(turn)
    history = list(app.get_state_history(config))
    shape = ref.build_app()[0].get_graph()
    return {
        "checkpoints": len(history), "stages": stages, "threshold": 4,
        "messages": len(app.get_state(config).values["messages"]),
        "paused_at_tools": sum(1 for snap in history if snap.next == ("tools",)),
        "events": [turn["events"] for turn in turns], "next_at_pause": turns[0]["next"],
        "nodes": sorted(shape.nodes), "edges": len(shape.edges),
        "conditional": sum(1 for edge in shape.edges if edge.conditional),
        "declared": len(ref.TOOLS), "lookup_demo": ref.web_lookup.invoke({"query": QUESTIONS[0]}),
        "lookup_key": ref.web_lookup.invoke({"query": "anthropic headquarters"})[:24],
        "exponent_allowed": set("9**9**9") <= set(ALLOWED),
        "import_blocked": ref.calculator.invoke({"expression": "__import__('os')"})[:20],
        "divide_by_zero": ref.calculator.invoke({"expression": "1/0"})[:26],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a two-turn conversation leaves 10 checkpoints against a threshold of 4",
            all([result["checkpoints"] == 10, result["stages"] == [3, 5, 8, 10],
                 result["stages"][1] > result["threshold"]]),
            f"the history grows {result['stages']} across the four stream calls the two "
            f"turns need, ending at {result['checkpoints']} snapshots over "
            f"{result['messages']} messages. 'At least {result['threshold']}' is satisfied "
            "before the first turn has finished",
        ),
        practice.Check(
            "FINDING: 'four-node' counts the two nodes LangGraph adds",
            all([result["nodes"] == ["__end__", "__start__", "agent", "tools"],
                 result["edges"] == 4, result["conditional"] == 2]),
            f"add_node is called twice and the compiled graph reports {result['nodes']} with "
            f"{result['edges']} edges, {result['conditional']} of them conditional. The "
            "number in the exercise is the number after compilation",
        ),
        practice.Check(
            "FINDING: two turns take four stream calls",
            all([result["paused_at_tools"] == 2, result["next_at_pause"] == ("tools",),
                 result["events"][0] == ["agent", "__interrupt__", "tools", "agent"]]),
            f"each turn streams to an {result['events'][0][1]} event with next == "
            f"{result['next_at_pause']} and needs a second stream carrying "
            f"Command(resume=True): {result['events'][0]}. Exactly "
            f"{result['paused_at_tools']} of the {result['checkpoints']} snapshots are "
            "paused at tools, which is a property of the interrupt and not of the traffic",
        ),
        practice.Check(
            "FINDING: web_lookup cannot answer the question the lesson asks it",
            all([result["lookup_demo"] == "unknown", result["lookup_key"].startswith("Anthropic")]),
            f"the facts dict is matched by exact equality after strip().lower(), so the "
            f"demo's own {QUESTIONS[0]!r} returns {result['lookup_demo']!r} while "
            f"'anthropic headquarters' returns {result['lookup_key']!r}. The tool works only "
            "if the model emits the dictionary key verbatim, and the prompt never says what "
            "the keys are",
        ),
        practice.Check(
            "FINDING: the calculator's allow-list admits exponentiation",
            all([result["exponent_allowed"], result["import_blocked"].startswith("ERROR: only"),
                 "ZeroDivisionError" in result["divide_by_zero"]]),
            f"`set(expression) <= set({ALLOWED!r})` passes '9**9**9', because * is in the "
            "list and ** is two of them -- a tool advertised as digits and + - * / ( ) will "
            f"accept an expression that does not terminate. __import__ is blocked "
            f"({result['import_blocked']!r}) and 1/0 returns {result['divide_by_zero']!r} "
            "rather than raising, so the guards that exist work",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
