"""Exercise 4 — every turn picks the right tool, and six answers are wrong.

    Read BFCL V4 description. Pick one category (e.g. "multi-turn") and run 10
    example prompts through your agent. Report pass rate.

Reading of the exercise: the category picked is **multi-turn**, because it is
the one the shipped registry has no mechanism for -- `dispatch` takes a
`ToolCall` and returns a `ToolResult`, and nothing carries a result into the
next turn. Ten two-turn prompts are scored V3-style on the *final value*
rather than on the calls emitted, which is the distinction V3 introduced.

**ANSWER: pass rate **4/10** for an agent that does not carry state, **10/10**
for one that does.** Each prompt adds two numbers, then multiplies the result
by a third. **6** of the ten refer to the first answer as "that"; **4**
restate it. The stateless agent passes exactly the **4** it was handed the
number for.

**FINDING: the failures are silent.** All **20** turns dispatch with
`ok=True` and **0** validation errors -- a missing carried value becomes a
plausible default, not an error, so the wrong answers arrive as successful
tool results. This is the case BFCL's state-based evaluation exists for.

**FINDING: AST-style scoring would call it perfect.** Comparing the tool
*name* chosen on each turn, the stateless agent matches **20/20**. Comparing
final values, it matches **4/10**. The calls are right and the answers are
wrong, which is why V3 stopped matching call trees.

**FINDING: the registry has nowhere to put the carry.** `ToolDef` has **5**
fields, `ToolCall` **3**, `ToolResult` **3**, and none is a session or state
handle; `dispatch_many` maps a list to a list with no accumulator. Multi-turn
state is the caller's job by construction -- which is why single-turn calling
is near-solved and the agentic and multi-turn categories are 70% of V4.

Structure: `run()` plays one scenario through the lesson's own registry; the
two agents differ only in where turn two's first argument comes from.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "06-tool-use-and-function-calling"
INTS = {"type": "object", "required": ["a", "b"],
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}
DEFAULT = 0
SCENARIOS = (                      # (a, b, k, the prompt restates the first answer)
    (2, 3, 4, False), (10, 5, 3, False), (7, 8, 2, False), (1, 1, 9, False),
    (6, 6, 5, False), (4, 9, 7, False),
    (3, 3, 6, True), (12, 8, 2, True), (5, 5, 4, True), (11, 4, 3, True),
)


def registry(ref):
    tools = ref.ToolRegistry()
    tools.register(ref.ToolDef(name="add", description="Add two integers a and b.",
                               input_schema=INTS, executor=ref.add))
    tools.register(ref.ToolDef(name="multiply", description="Multiply two integers.",
                               input_schema=INTS, executor=ref.multiply))
    return tools


def run(ref, tools, scenario, carries):
    """Turn one adds; turn two multiplies the first answer by k."""
    a, b, k, restates = scenario
    first = tools.dispatch(ref.ToolCall("t1", "add", {"a": a, "b": b}))
    if carries:
        carried = int(first.content)
    else:
        carried = a + b if restates else DEFAULT
    second = tools.dispatch(ref.ToolCall("t2", "multiply", {"a": carried, "b": k}))
    return {"results": [first, second], "names": ["add", "multiply"],
            "final": second.content, "want": str((a + b) * k)}


def ast_matches(rows):
    """V2-style scoring: was the right tool named on each turn?"""
    return sum(1 for entry in rows for name, want in
               zip(entry["names"], ("add", "multiply")) if name == want)


def carry_failures(passed):
    return sum(1 for good, scenario in zip(passed, SCENARIOS)
               if not good and not scenario[3])


def report(ref, carries):
    rows = [run(ref, registry(ref), scenario, carries) for scenario in SCENARIOS]
    results = [row for entry in rows for row in entry["results"]]
    passed = [entry["final"] == entry["want"] for entry in rows]
    ok = sum(row.ok for row in results)
    return {"passed": sum(passed), "turns": len(results), "ok": ok,
            "errors": len(results) - ok, "ast": ast_matches(rows),
            "failed_carry": carry_failures(passed)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    stateless, stateful = report(ref, False), report(ref, True)
    return {
        "stateless": stateless, "stateful": stateful,
        "cases": len(SCENARIOS),
        "implicit": sum(1 for scenario in SCENARIOS if not scenario[3]),
        "tooldef": len(ref.ToolDef.__dataclass_fields__),
        "toolcall": list(ref.ToolCall.__dataclass_fields__),
        "toolresult": list(ref.ToolResult.__dataclass_fields__),
        "state_fields": [field for field in
                         (*ref.ToolCall.__dataclass_fields__, *ref.ToolResult.__dataclass_fields__)
                         if "state" in field or "session" in field],
    }


def verify(result):
    stateless, stateful = result["stateless"], result["stateful"]
    return [
        practice.Check(
            "ANSWER: multi-turn pass rate is 4/10 without carried state and 10/10 with",
            all([stateless["passed"] == 4, stateful["passed"] == 10,
                 result["cases"] == 10, result["implicit"] == 6,
                 stateless["failed_carry"] == 6]),
            f"over {result['cases']} two-turn prompts the stateless agent passes "
            f"{stateless['passed']} and the carrying agent {stateful['passed']}. "
            f"{result['implicit']} prompts refer to the first answer as 'that', and "
            f"{stateless['failed_carry']} of them are exactly the failures -- the agent "
            "passes every prompt it was handed the number for",
        ),
        practice.Check(
            "FINDING: the failures are silent",
            all([stateless["turns"] == 20, stateless["ok"] == 20,
                 stateless["errors"] == 0]),
            f"all {stateless['turns']} turns dispatch with ok=True and "
            f"{stateless['errors']} validation errors. A missing carried value becomes a "
            "plausible default rather than a malformed argument, so six wrong answers "
            "arrive as successful tool results",
        ),
        practice.Check(
            "FINDING: AST-style scoring would call it perfect",
            all([stateless["ast"] == 20, stateless["ast"] == stateful["ast"],
                 stateless["passed"] < stateful["passed"]]),
            f"scored on the tool name chosen each turn the stateless agent matches "
            f"{stateless['ast']}/20, the same as the carrying agent; scored on the final "
            f"value it matches {stateless['passed']}/10. The calls are right and the "
            "answers are wrong, which is the case state-based evaluation exists for",
        ),
        practice.Check(
            "FINDING: the registry has nowhere to put the carry",
            all([result["tooldef"] == 5,
                 result["toolcall"] == ["tool_use_id", "name", "args"],
                 result["toolresult"] == ["tool_use_id", "ok", "content"],
                 result["state_fields"] == []]),
            f"ToolCall carries {result['toolcall']} and ToolResult carries "
            f"{result['toolresult']} -- {len(result['state_fields'])} session or state "
            "handles between them -- and dispatch_many maps a list to a list with no "
            "accumulator. Multi-turn state is the caller's job by construction",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
