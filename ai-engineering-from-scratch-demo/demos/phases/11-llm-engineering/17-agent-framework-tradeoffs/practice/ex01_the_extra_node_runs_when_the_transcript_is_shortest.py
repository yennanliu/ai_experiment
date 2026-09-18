"""Exercise 1 — the extra node runs when the transcript is shortest.

    **Easy.** Take the same task -- "research Anthropic's headquarters, write a
    200-word brief, cite sources" -- and implement it in LangGraph (four nodes:
    plan, search, write, cite) and in CrewAI (three roles: researcher, writer,
    editor). Report token cost per run and lines of code.

Reading of the exercise: the LangGraph arm is the real library -- a compiled
four-node `StateGraph` whose nodes record every prompt they are handed -- so
"token cost per run" is measured rather than reasoned about. CrewAI is not
installed here, so its arm is a transcription of the documented shape: three
role-prompted steps, each receiving the previous step's output as context. Both
arms produce the same four artifacts from the same strings, so the comparison
is of the wiring and nothing else.

**ANSWER: LangGraph 4 calls / 337 input tokens / 9 lines; CrewAI 3 calls /
283 tokens / 7 lines.** 19% apart on tokens and two lines apart on code.

**MECHANISM: 160 of those tokens are the 200-word brief.** It is 47% of the
LangGraph total and 57% of the CrewAI total, and both arms have to carry it
into the last prompt. The cost of this task is set by the artifact, not by the
framework -- and the two numbers the exercise asks for are the two least
sensitive to the choice it is asking about.

**FINDING: the extra call is nearly free.** LangGraph's `plan` node runs when
the transcript is one line long: 29 tokens of the 337. So 33% more calls buys
19% more input tokens, and the extra 54 tokens are mostly the plan's own text
appearing in three later prompts. Call count and prompt size are inversely
ordered along a chain, which is why counting calls ranks frameworks backwards.

**MEASUREMENT: the real graph hands every node the whole accumulated list.**
The four prompts measure 29, 42, 53 and 213 tokens -- `add_messages` appends
and nothing prunes, so the last node reads everything the first three wrote.
This is the library's behaviour, recorded from inside the node functions, not a
property of the transcription.

**FINDING: "cite sources" has no channel to be cited in.** The state schema has
one key, `messages`, and CrewAI's context is a single accumulating string, so
the citation is a sentence inside a reply. Neither shape can check that the
brief's claims match the source it names, which is the only part of the task
that would have needed structure.

Structure: `TASK` and the four reply strings are shared by both arms.
`build_langgraph` compiles the real four-node graph and `run_langgraph` drives
it, recording the prompt handed to each node; `crewai_shape` is the transcribed
three-role pipeline, since CrewAI is not installed here. `tokens` is the lesson
family's `words * 4 // 3` estimator and `body_lines` counts the non-blank lines
of a builder.
"""

from __future__ import annotations

import inspect
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "17-agent-framework-tradeoffs"
TASK = "research Anthropic's headquarters, write a 200-word brief, cite sources"
SYSTEM = "You are a research assistant. Work step by step and cite every claim."
PLAN = "Plan: 1 search, 2 draft 200 words, 3 attach citations."
FOUND = "Anthropic is headquartered in San Francisco, California (anthropic.com/company)."
BRIEF = "Anthropic is an AI safety company headquartered in San Francisco. " * 12
CITE = "Sources: anthropic.com/company (accessed 2026-09-18)."
NODES = (("plan", PLAN), ("search", FOUND), ("write", BRIEF), ("cite", CITE))
ROLES = (("researcher", "verifies every fact against a primary source", FOUND),
         ("writer", "produces tight 200-word briefs", BRIEF),
         ("editor", "checks citations and trims to length", CITE))


def tokens(text):
    return max(1, len(text.split()) * 4 // 3)


def build_langgraph(recorder):
    graph = StateGraph(TypedDict("Brief", {"messages": Annotated[list, add_messages]}))
    for name, reply in NODES:
        graph.add_node(name, recorder(name, reply))
    graph.set_entry_point("plan")
    for (source, _), (target, _) in zip(NODES, NODES[1:]):
        graph.add_edge(source, target)
    graph.add_edge("cite", END)
    return graph.compile(checkpointer=MemorySaver())


def crewai_shape():
    context, calls = "", []
    for name, backstory, reply in ROLES:
        calls.append(tokens(f"You are the {name}. A senior professional who {backstory}.\n"
                            f"Overall task: {TASK}\nContext:\n{context}"))
        context += reply + "\n"
    return calls


def body_lines(function):
    return len([line for line in inspect.getsource(function).splitlines() if line.strip()])


def run_langgraph():
    seen = []

    def recorder(name, reply):
        def node(state):
            seen.append((name, tokens(SYSTEM + "\n"
                                      + "\n".join(str(m.content) for m in state["messages"]))))
            return {"messages": [AIMessage(content=reply)]}
        return node

    app = build_langgraph(recorder)
    list(app.stream({"messages": [HumanMessage(TASK)]},
                    {"configurable": {"thread_id": "brief"}}, stream_mode="updates"))
    return seen, app


def solve():
    parity.load_reference(PHASE, LESSON, "main")
    seen, app = run_langgraph()
    crew = crewai_shape()
    return {
        "graph_calls": [count for _, count in seen], "graph_nodes": [name for name, _ in seen],
        "graph_total": sum(count for _, count in seen),
        "crew_calls": crew, "crew_total": sum(crew),
        "brief_tokens": tokens(BRIEF),
        "graph_lines": body_lines(build_langgraph), "crew_lines": body_lines(crewai_shape),
        "state_keys": sorted(app.get_state({"configurable": {"thread_id": "brief"}}).values),
        "cite_in_messages": any(CITE in str(m.content) for m in
                                app.get_state({"configurable": {"thread_id": "brief"}})
                                .values["messages"]),
    }


def verify(result):
    graph, crew = result["graph_total"], result["crew_total"]
    return [
        practice.Check(
            "ANSWER: LangGraph 4 calls / 337 tokens / 9 lines, CrewAI 3 / 283 / 7",
            all([result["graph_calls"] == [29, 42, 53, 213], graph == 337,
                 crew == 283, result["graph_lines"] == 9, result["crew_lines"] == 7]),
            f"the real four-node graph is handed {result['graph_calls']} tokens across "
            f"{result['graph_nodes']}, {graph} in total, in {result['graph_lines']} lines; "
            f"the transcribed three-role pipeline is handed {result['crew_calls']}, "
            f"{crew} in total, in {result['crew_lines']}. {graph / crew - 1:.0%} apart on "
            "tokens and two lines apart on code",
        ),
        practice.Check(
            "MECHANISM: 160 of those tokens are the 200-word brief",
            all([result["brief_tokens"] == 160,
                 result["brief_tokens"] / graph > 0.4, result["brief_tokens"] / crew > 0.5]),
            f"the brief alone is {result['brief_tokens']} tokens -- "
            f"{result['brief_tokens'] / graph:.0%} of the LangGraph total and "
            f"{result['brief_tokens'] / crew:.0%} of the CrewAI one -- and both arms carry "
            "it into their last prompt. The cost is set by the artifact, and the two numbers "
            "the exercise asks for are the least sensitive to the choice it asks about",
        ),
        practice.Check(
            "FINDING: the extra call is nearly free",
            all([result["graph_calls"][0] == 29, len(result["graph_calls"]) == 4,
                 len(result["crew_calls"]) == 3, graph / crew < 1.25]),
            f"the plan node runs when the transcript is one line long -- "
            f"{result['graph_calls'][0]} tokens of {graph} -- so "
            f"{len(result['graph_calls']) / len(result['crew_calls']) - 1:.0%} more calls "
            f"buys {graph / crew - 1:.0%} more input. Call count and prompt size are "
            "inversely ordered along a chain, so counting calls ranks frameworks backwards",
        ),
        practice.Check(
            "MEASUREMENT: the real graph hands every node the whole accumulated list",
            all([result["graph_calls"] == sorted(result["graph_calls"]),
                 result["graph_calls"][-1] > 4 * result["graph_calls"][0]]),
            f"the prompts measure {result['graph_calls']} -- monotonically increasing, the "
            f"last {result['graph_calls'][-1] / result['graph_calls'][0]:.1f}x the first -- "
            "because add_messages appends and nothing prunes. Recorded from inside the node "
            "functions, so it is the library's behaviour and not the transcription's",
        ),
        practice.Check(
            "FINDING: 'cite sources' has no channel to be cited in",
            all([result["state_keys"] == ["messages"], result["cite_in_messages"]]),
            f"the state schema has one key, {result['state_keys']}, and CrewAI's context is "
            "a single accumulating string, so the citation is a sentence inside a reply. "
            "Neither shape can check that the brief's claims match the source it names, "
            "which is the only part of the task that needed structure",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
