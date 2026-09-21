"""Exercise 1 — declining and inventing a tool name are the same event.

    Add a "no-op" tool that lets the model explicitly refuse to use any other
    tool. Measure on a BFCL-like hallucination test.

Reading of the exercise: BFCL's hallucination category scores whether the
model declines when no tool fits, so scoring it requires a *correct action*
for the no-tool cases. The shipped registry has none: `catalog()` advertises
only executable tools and `dispatch` answers an unregistered name with an
error. The measurement is therefore a ten-case fixture -- five that need a
tool, five that need none -- run against the registry with and without a
`no_tool` entry.

**ANSWER: a `no_tool` tool with a `reason` string, and the score moves 5 to
10.** A model that must always call something scores **5/10** overall and
**0/5** on the hallucination half. With `no_tool` registered, a model that
declines correctly scores **10/10** and **5/5**, and all **10** dispatches
return `ok=True` with a recorded reason.

**FINDING: before the no-op exists, declining is inventing.**
`dispatch` on `no_tool` in the unregistered registry returns
`error: unknown tool 'no_tool'` -- the same shape, the same `ok=False`, the
same `ToolResult` as the demo's bogus `subtract` call. A correct refusal and
a hallucinated tool name are one event class, so the category cannot be
scored at all.

**FINDING: refusing in the runtime instead leaves holes in the trace.** A
threshold that suppresses the call produces **5** `ToolResult`s for **10**
prompts while `catalog()` still advertises **3** tools. The model was never
told refusal was permitted, and nothing records why five prompts produced
nothing.

**FINDING: the no-op's cost is that refusal becomes a success.** With
`no_tool` registered, a correct decliner and a model that declines
*everything* both return **10/10** `ok=True` results; on task accuracy they
score **10/10** and **5/10**. Any metric built on dispatch success cannot
tell them apart.

Structure: `score()` runs one model's proposed calls through the lesson's own
`ToolRegistry`; the models differ only in what they propose.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "06-tool-use-and-function-calling"
NO_TOOL_SCHEMA = {"type": "object", "required": ["reason"],
                  "properties": {"reason": {"type": "string"}}}
CASES = (
    ("add 2 and 3", "add", {"a": 2, "b": 3}),
    ("what is 4 plus 5", "add", {"a": 4, "b": 5}),
    ("multiply 6 and 7", "multiply", {"a": 6, "b": 7}),
    ("multiply 3 by 3", "multiply", {"a": 3, "b": 3}),
    ("classify this ticket as open", "classify", {"status": "open"}),
    ("tell me a joke", None, None),
    ("what do you think of the weather", None, None),
    ("summarise the last meeting", None, None),
    ("who won the 1998 world cup", None, None),
    ("explain why the sky is blue", None, None),
)


def registry(ref, with_no_tool):
    tools = ref.ToolRegistry()
    for name, executor, schema in (
            ("add", ref.add, {"a": "integer", "b": "integer"}),
            ("multiply", ref.multiply, {"a": "integer", "b": "integer"})):
        tools.register(ref.ToolDef(
            name=name, description=f"{name} two integers a and b.", executor=executor,
            input_schema={"type": "object", "required": sorted(schema),
                          "properties": {k: {"type": v} for k, v in schema.items()}}))
    tools.register(ref.ToolDef(
        name="classify", description="Classify a status as one of the allowed labels.",
        executor=ref.classify, input_schema={
            "type": "object", "required": ["status"],
            "properties": {"status": {"type": "string",
                                      "enum": ["open", "closed", "pending"]}}}))
    if with_no_tool:
        tools.register(ref.ToolDef(
            name="no_tool", description="Decline: no registered tool applies. Give a reason.",
            input_schema=NO_TOOL_SCHEMA, executor=lambda reason: f"declined: {reason}"))
    return tools


def propose(case, style):
    """What the model emits for one case: a real call, a decline, or a guess."""
    prompt, tool, args = case
    if style == "decline_all":
        return "no_tool", {"reason": prompt}
    if tool is None:
        return ("no_tool", {"reason": prompt}) if style == "declines" else ("add", {"a": 1, "b": 1})
    return tool, args


def score(ref, tools, style):
    """BFCL-shaped: overall accuracy, plus the hallucination half scored apart."""
    chosen, results = [], []
    for index, case in enumerate(CASES):
        name, args = propose(case, style)
        chosen.append(name)
        results.append(tools.dispatch(ref.ToolCall(f"u{index:02d}", name, args)))
    return {"results": results, "ok": sum(1 for row in results if row.ok),
            "correct": sum(1 for case, name in zip(CASES, chosen)
                           if name == (case[1] or "no_tool")),
            "hallucination": sum(1 for case, name in zip(CASES, chosen)
                                 if case[1] is None and name == "no_tool")}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    bare, full = registry(ref, False), registry(ref, True)
    must = score(ref, bare, "must_call")
    declines = score(ref, full, "declines")
    always = score(ref, full, "decline_all")
    unknown = bare.dispatch(ref.ToolCall("u", "no_tool", {"reason": "nothing fits"}))
    bogus = bare.dispatch(ref.ToolCall("u", "subtract", {"a": 1, "b": 2}))
    return {
        "must": {k: must[k] for k in ("correct", "ok", "hallucination")},
        "declines": {k: declines[k] for k in ("correct", "ok", "hallucination")},
        "always": {k: always[k] for k in ("correct", "ok", "hallucination")},
        "decline_content": declines["results"][5].content,
        "unknown": (unknown.ok, unknown.content), "bogus": (bogus.ok, bogus.content),
        "same_shape": unknown.content.split()[:3] == bogus.content.split()[:3],
        "bare_catalog": len(bare.catalog()), "full_catalog": len(full.catalog()),
        "suppressed": sum(1 for case in CASES if case[1] is not None),
        "cases": len(CASES),
    }


def verify(result):
    must, declines, always = result["must"], result["declines"], result["always"]
    return [
        practice.Check(
            "ANSWER: a no_tool entry moves the score from 5/10 to 10/10",
            all([must["correct"] == 5, must["hallucination"] == 0,
                 declines["correct"] == 10, declines["hallucination"] == 5,
                 declines["ok"] == 10,
                 result["decline_content"].startswith("declined:")]),
            f"a model that must always call something scores {must['correct']}/10 "
            f"overall and {must['hallucination']}/5 on the hallucination half. With "
            f"no_tool registered a correct decliner scores {declines['correct']}/10 and "
            f"{declines['hallucination']}/5, with {declines['ok']} ok results carrying "
            f"a reason: {result['decline_content']!r}",
        ),
        practice.Check(
            "FINDING: before the no-op exists, declining is inventing",
            all([result["unknown"][0] is False, result["bogus"][0] is False,
                 result["same_shape"] is True,
                 result["unknown"][1].startswith("error: unknown tool")]),
            f"dispatching no_tool against the bare registry returns "
            f"{result['unknown']!r} and the demo's bogus subtract returns "
            f"{result['bogus']!r} -- same ok flag, same message shape "
            f"({result['same_shape']}). A correct refusal and a hallucinated name are "
            "one event class, so the category cannot be scored",
        ),
        practice.Check(
            "FINDING: refusing in the runtime leaves holes in the trace",
            all([result["suppressed"] == 5, result["cases"] == 10,
                 result["bare_catalog"] == 3, result["full_catalog"] == 4]),
            f"suppressing the call for the no-tool prompts yields "
            f"{result['suppressed']} ToolResults for {result['cases']} prompts while "
            f"catalog() still advertises {result['bare_catalog']} tools. Registering the "
            f"no-op instead makes it {result['full_catalog']}, and every prompt leaves a "
            "record of what was decided and why",
        ),
        practice.Check(
            "FINDING: the no-op's cost is that refusal becomes a success",
            all([always["ok"] == 10, declines["ok"] == 10, always["correct"] == 5,
                 declines["correct"] == 10]),
            f"a correct decliner and a model that declines everything both return "
            f"{always['ok']}/10 ok results, while scoring {declines['correct']}/10 and "
            f"{always['correct']}/10 on the task. Any metric built on dispatch success "
            "cannot tell them apart, which is why BFCL scores the category separately",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
