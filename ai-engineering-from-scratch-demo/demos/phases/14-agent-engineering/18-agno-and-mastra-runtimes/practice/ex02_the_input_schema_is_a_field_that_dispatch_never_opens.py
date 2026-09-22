"""Exercise 2 — the input schema is a field that dispatch never opens.

    Read Mastra's docs. Port the same loop to Mastra. What changed in tool
    typing (Zod vs nothing)?

Reading of the exercise: `MastraTool` has an `input_schema`, which is the
whole point of the Zod comparison, and `MastraAgent.run` calls
`tool.fn(**args)` without looking at it. So the typed half of the port is
declarative and inert, and the measurement is what a schema would have caught
that the shipped call does not. Lesson 01's loop is ported into
`MastraAgent` plus a `MastraWorkflow` so both primitives carry part of it.

**ANSWER: the loop becomes an agent plus a workflow, and the schema becomes
a docstring.** The three tool calls of Lesson 01's demo run through
`MastraAgent.run` producing a **3**-entry trace and the same observations,
with the turn sequencing moved into a `MastraWorkflow` of **3** steps. Every
tool declares an `input_schema`, and `run` reads it **0** times.

**FINDING: a wrong argument name is a crash, not a validation error.**
Calling the search tool with `{"q": "tax"}` instead of `{"query": "tax"}`
raises `TypeError` out of `MastraAgent.run`, because `fn(**args)` binds
directly. Lesson 01's `ToolRegistry.dispatch` catches the same mistake and
returns `error: bad args` as an observation -- the untyped runtime is the one
that survives it.

**FINDING: the schema would have caught three of four probe arguments.**
Checked against the declared `input_schema`, **3** of **4** probes are
rejected -- a missing key, an unknown key and a wrong type -- and **1**
passes. Unchecked, **2** of the **4** raise `TypeError` and **2** run,
including the wrong-typed one. Typing moves failures earlier and makes one of
them visible at all.

**FINDING: the workflow has no failure path.** `MastraWorkflow.run` threads
`current = fn(current)` through every step with no try, no predicate and no
early exit, so a step that raises takes the workflow with it and a step that
returns nonsense is passed on. Of the three primitives, **0** have a
validation seam.

Structure: `check()` is the schema validator the runtime does not call;
`mastra_port()` splits Lesson 01's loop across the agent and the workflow.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "18-agno-and-mastra-runtimes"
LOOP = "01-the-agent-loop"
SCHEMA = {"query": str}
PROBES = ({"query": "tax"}, {}, {"query": "tax", "limit": 3}, {"query": 5})


def _mistyped(schema, args):
    return [key for key, value in args.items()
            if key in schema and not isinstance(value, schema[key])]


def check(schema, args):
    """What a Zod-typed tool would do before the call."""
    problems = {"missing": [key for key in schema if key not in args],
                "unknown": [key for key in args if key not in schema],
                "wrong type": _mistyped(schema, args)}
    return [f"{label} {keys}" for label, keys in problems.items() if keys]


def call(agent, tool_name, args):
    try:
        return agent.run("go", [(tool_name, args)])[1][0][1], None
    except TypeError as exc:
        return None, type(exc).__name__


def mastra_port(ref, react):
    """Lesson 01's tools as MastraTools; its turn order as a MastraWorkflow."""
    store = react.KVStore()
    tools = [ref.MastraTool("calculator", {"expr": str}, react.calculator),
             ref.MastraTool("kv_set", {"key": str, "value": str}, store.set),
             ref.MastraTool("kv_get", {"key": str}, store.get)]
    agent = ref.MastraAgent("react", "run the tools", tools)
    calls = [("kv_set", {"key": "base", "value": "120"}),
             ("calculator", {"expr": "120 * 0.15"}),
             ("kv_get", {"key": "base"})]
    output, trace = agent.run("total with tax", calls)
    workflow = ref.MastraWorkflow(
        steps=[(name, (lambda value, n=name: f"{n} done")) for name, _ in calls])
    return agent, output, trace, workflow.run("start")


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    react = parity.load_reference(PHASE, LOOP, "main")
    agent, output, trace, steps = mastra_port(ref, react)
    search = ref.MastraAgent("search", "find things",
                             [ref.MastraTool("search", SCHEMA, ref._mastra_tool_fn)])
    typed = [check(SCHEMA, args) for args in PROBES]
    untyped = [call(search, "search", args) for args in PROBES]
    registry = react.ToolRegistry()
    registry.register("calculator", react.calculator)
    shipped = registry.dispatch(react.ToolCall("calculator", {"expression": "1+1"}))
    return {
        "trace": len(trace), "observations": [result for _, result in trace],
        "output": output, "steps": len(steps),
        "schema_reads": inspect.getsource(ref.MastraAgent.run).count("input_schema"),
        "tool_fields": list(ref.MastraTool.__dataclass_fields__),
        "typed_rejected": sum(1 for notes in typed if notes),
        "typed_passed": sum(1 for notes in typed if not notes),
        "untyped_raised": sum(1 for _, error in untyped if error),
        "untyped_ran": sum(1 for value, _ in untyped if value is not None),
        "wrong_type_ran": untyped[3][0] is not None,
        "shipped_error": shipped,
        "workflow_guards": inspect.getsource(ref.MastraWorkflow.run).count("try"),
        "agent_guards": inspect.getsource(ref.MastraAgent.run).count("try"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: an agent plus a workflow, and the schema is inert",
            all([result["trace"] == 3, result["steps"] == 3,
                 result["output"] == "react processed 3 tools",
                 result["schema_reads"] == 0,
                 result["observations"][1] == "18.0"]),
            f"the three tool calls run through MastraAgent.run producing a "
            f"{result['trace']}-entry trace -- {result['observations']} -- with the turn "
            f"sequencing in a {result['steps']}-step MastraWorkflow. Every tool declares "
            f"an input_schema and run reads it {result['schema_reads']} times",
        ),
        practice.Check(
            "FINDING: a wrong argument name is a crash, not a validation error",
            all([result["untyped_raised"] == 2,
                 result["shipped_error"].startswith("error: bad args"),
                 result["tool_fields"] == ["name", "input_schema", "fn"]]),
            f"MastraAgent.run binds fn(**args) directly, so {result['untyped_raised']} "
            f"of the four probes raise TypeError out of the agent, while Lesson 01's "
            f"registry returns {result['shipped_error']!r} as an observation. The "
            "untyped runtime is the one that survives the mistake",
        ),
        practice.Check(
            "FINDING: the schema would have caught three of four probes",
            all([result["typed_rejected"] == 3, result["typed_passed"] == 1,
                 result["untyped_ran"] == 2, result["wrong_type_ran"] is True]),
            f"checked against the declared input_schema, {result['typed_rejected']} of "
            f"four probes are rejected and {result['typed_passed']} passes. Unchecked, "
            f"{result['untyped_ran']} run -- including the wrong-typed one "
            f"({result['wrong_type_ran']}), which no exception would ever have caught",
        ),
        practice.Check(
            "FINDING: the workflow has no failure path",
            all([result["workflow_guards"] == 0, result["agent_guards"] == 0]),
            f"MastraWorkflow.run threads current = fn(current) through every step with "
            f"{result['workflow_guards']} try blocks, and MastraAgent.run has "
            f"{result['agent_guards']}. A step that raises takes the workflow with it "
            "and a step that returns nonsense is passed on; none of the three "
            "primitives has a validation seam",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
