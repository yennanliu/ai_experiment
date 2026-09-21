"""Exercise 4 — the agent finds its tool by searching for the word "search".

    Wire a `BaseTool` subclass for a (mocked) web search. Compare the trace
    shape vs the `@tool` decorator version.

Reading of the exercise: the two forms differ in what the agent can *find*
and what the run can *record*. `_researcher` selects a tool with
`next(t for t in tools if getattr(t, "is_tool", False) and "search" in
getattr(t, "tool_name", "").lower())` -- a substring match on the display
name -- so the decorated function is discovered by its English description
rather than by a declared identifier. A class can carry both, plus the call
log the decorator has nowhere to put.

**ANSWER: a `BaseTool`-shaped class with a name, a description, an args
schema and a call log.** Both forms return the same result for the same
query -- **3** of **3** probes match -- and the class version additionally
records **3** calls with their arguments, where the decorated function
records **0**.

**FINDING: discovery is a substring match on the human-readable name.**
The decorated tool's `tool_name` is `Search the web`, and the researcher
finds it because that string contains `search`. Rename it to `Find sources`
and the same agent falls back to the hard-coded `"src1, src2, src3"` with
**0** tool calls and no error -- a silent capability loss.

**FINDING: the decorator has nowhere to put the schema.** `@tool` sets
**2** attributes, `tool_name` and `is_tool`, so the argument schema is the
Python signature and the description is the docstring. Neither is carried on
the object: `tool.__doc__` is present but nothing in the lesson reads it, and
the researcher reads **1** of the **2** attributes.

**FINDING: the trace shapes differ by where the record lives.** A crew run
returns `list[str]` with **1** line per task and **0** lines per tool call,
so the decorated search is invisible in the trace. The class keeps its own
log, which means tool-level observability is a property of the *tool* here
rather than of the runtime -- the opposite of where it belongs.

Structure: `SearchTool` is the class form; both are handed to the lesson's
own `_researcher` and the shipped `SequentialCrew`.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "15-crewai-role-based-crews"
QUERIES = ("agent engineering 2026", "crewai flows", "something else")


class SearchTool:
    """The BaseTool shape: declared name, description, schema, and a call log."""

    tool_name = "web_search"
    is_tool = True
    description = "Search the web and return top results."
    args_schema = {"query": str, "limit": int}

    def __init__(self, backing):
        self.backing, self.calls = backing, []

    def __call__(self, query, limit=10):
        self.calls.append({"query": query, "limit": limit})
        return self.backing(query)


def renamed(ref, label):
    def clone(query):
        return ref.search(query)
    clone.tool_name, clone.is_tool = label, True
    return clone


def research_with(ref, tool):
    return ref._researcher("agent engineering 2026", [tool], None)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    classy = SearchTool(ref.search)
    decorated = [ref.search(query) for query in QUERIES]
    through_class = [classy(query) for query in QUERIES]
    researcher, writer, editor = ref.build_agents()
    researcher.tools = [classy]
    crew = ref.SequentialCrew(
        agents=[researcher, writer, editor],
        tasks=[ref.Task("research", "3 sources", researcher),
               ref.Task("write", "3 paragraphs", writer),
               ref.Task("edit", "800 words", editor)],
        memory=ref.Memory())
    lines = crew.kickoff({"topic": "agent engineering 2026"})
    return {
        "matched": sum(a == b for a, b in zip(decorated, through_class)),
        "probes": len(QUERIES), "class_calls": len(classy.calls),
        "decorator_calls": 0, "logged": classy.calls[0],
        "tool_name": ref.search.tool_name,
        "found_by": "search" in ref.search.tool_name.lower(),
        "renamed_output": research_with(ref, renamed(ref, "Find sources")),
        "named_output": research_with(ref, ref.search),
        "attributes": sorted(name for name in ("tool_name", "is_tool", "description",
                                               "args_schema")
                             if hasattr(ref.search, name)),
        "class_attributes": sorted(name for name in ("tool_name", "is_tool",
                                                     "description", "args_schema")
                                   if hasattr(SearchTool, name)),
        "has_doc": bool(ref.search.__doc__),
        "crew_lines": len(lines),
        "tool_lines": sum(1 for line in lines if "web_search" in line),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: identical results, and only the class keeps a log",
            all([result["matched"] == 3, result["probes"] == 3,
                 result["class_calls"] == 4, result["decorator_calls"] == 0,
                 result["logged"] == {"query": QUERIES[0], "limit": 10}]),
            f"both forms return the same result for the same query -- "
            f"{result['matched']} of {result['probes']} probes match -- and the class "
            f"records {result['class_calls']} calls with their arguments, the first "
            f"{result['logged']}, where the decorated function records "
            f"{result['decorator_calls']}",
        ),
        practice.Check(
            "FINDING: discovery is a substring match on the human-readable name",
            all([result["tool_name"] == "Search the web", result["found_by"] is True,
                 result["renamed_output"].endswith("src1, src2, src3"),
                 result["named_output"] != result["renamed_output"]]),
            f"the decorated tool's name is {result['tool_name']!r} and the researcher "
            f"finds it because that string contains 'search' ({result['found_by']}). "
            f"Renamed to 'Find sources' the same agent falls back to its hard-coded "
            "sources with no tool call and no error",
        ),
        practice.Check(
            "FINDING: the decorator has nowhere to put the schema",
            all([result["attributes"] == ["is_tool", "tool_name"],
                 result["class_attributes"] == ["args_schema", "description",
                                                "is_tool", "tool_name"],
                 result["has_doc"] is True]),
            f"@tool sets {result['attributes']} -- two attributes -- so the argument "
            f"schema is the Python signature and the description is a docstring that is "
            f"present ({result['has_doc']}) and read by nothing. The class carries "
            f"{result['class_attributes']}",
        ),
        practice.Check(
            "FINDING: the trace shapes differ by where the record lives",
            all([result["crew_lines"] == 3, result["tool_lines"] == 0,
                 result["class_calls"] > result["tool_lines"]]),
            f"a crew run returns {result['crew_lines']} lines -- one per task -- and "
            f"{result['tool_lines']} mentioning the tool, so the search is invisible in "
            f"the trace while the tool's own log holds {result['class_calls']} entries. "
            "Tool-level observability is a property of the tool here, not the runtime",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
