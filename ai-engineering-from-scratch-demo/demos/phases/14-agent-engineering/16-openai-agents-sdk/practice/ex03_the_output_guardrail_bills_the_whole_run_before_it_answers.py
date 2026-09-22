"""Exercise 3 — the output guardrail bills the whole run before it answers.

    Write a blocking output guardrail. Compare latency on prompts that would
    trip it vs ones that pass.

Reading of the exercise: an output guardrail runs after the loop, so a prompt
that trips it has already paid for every agent turn, tool call and handoff.
Latency is therefore reported as *work done before the verdict* -- policy
calls, tool calls and spans -- rather than in seconds, which would describe
this machine rather than the design.

**ANSWER: a blocking output guardrail, and the tripping prompt costs the
same as the passing one.** Both prompts run **2** agent turns, **1** tool
call and **1** handoff before the check: **7** spans each. The tripping
prompt raises `GuardrailTripped` after all of it, so the refusal costs
**100%** of the work the answer would have cost.

**FINDING: the input guardrail costs nothing by comparison.** Blocking the
same content on the way in raises after **0** agent turns, **0** tool calls
and **1** span. Anything a guardrail can decide from the input belongs on the
input side, and the shipped `_pii_check` reads only the user's text -- it
would work unchanged as an output check, at **7x** the span count.

**FINDING: the trip leaves no final output anywhere.** `Runner.run` raises
instead of returning, so the caller gets `GuardrailTripped('output', ...)`
and the text that tripped it is **not** on the exception -- **2** of its
attributes are `which` and `reason`, and neither carries the output. The
span does record it, which means the only copy is in the trace.

**FINDING: the guardrails do not re-run across a handoff.** Input checks run
once before the loop, so text that enters through `policy_output["input"]`
on a transfer is never checked: a handoff payload containing `ssn` reaches
the second agent with **1** input-guardrail span for the run and **0** for
the transfer.

Structure: `measure()` runs the shipped `Runner` and counts spans by kind;
the guardrails are the lesson's own `_pii_check` and a blocking variant.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "16-openai-agents-sdk"
CLEAN = "I need a refund for invoice 4711"
DIRTY = "I need a refund and my ssn is on file"


def blocked(text):
    """The blocking output guardrail: refuses anything mentioning an ssn."""
    if "ssn" in text.lower():
        return False, "output mentions a social security number"
    return True, "ok"


def build(ref, carry_input=False):
    """Triage looks something up, then hands the user's own words to billing."""
    billing = ref.Agent("billing", "handle refunds",
                        lambda text: {"kind": "final", "text": f"billing: {text}"})
    calls, original = {"n": 0}, {"text": ""}

    def policy(text):
        calls["n"] += 1
        if calls["n"] == 1:
            original["text"] = text
            return {"kind": "tool", "tool": "lookup", "args": {"key": "4711"}}
        payload = "please check the ssn on file" if carry_input else original["text"]
        return {"kind": "handoff", "to": "billing", "input": payload}

    return ref.Agent(
        "triage", "route", policy,
        tools=[ref.FunctionTool("lookup", "look up", lambda key: f"invoice {key} open")],
        handoffs=[ref.Handoff(target=billing)]), calls


def count(trace, prefix):
    direct = sum(1 for child in trace.children if child.name.startswith(prefix))
    nested = sum(1 for child in trace.children for grand in child.children
                 if grand.name.startswith(prefix))
    return direct + nested


def measure(ref, text, guards, carry_input=False):
    agent, calls = build(ref, carry_input)
    runner = ref.Runner(max_hops=5, trace=ref.Span(name="run"), **guards)
    try:
        output, tripped = runner.run(agent, text), None
    except ref.GuardrailTripped as exc:
        output, tripped = None, exc
    spans = sum(1 for child in runner.trace.children) + sum(
        1 for child in runner.trace.children for _ in child.children)
    return {"output": output, "tripped": tripped, "turns": calls["n"],
            "tools": count(runner.trace, "tool."),
            "handoffs": count(runner.trace, "handoff."),
            "inputs": count(runner.trace, "input_guardrail."),
            "outputs": count(runner.trace, "output_guardrail."), "spans": spans}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    out_guard = {"output_guardrails": [ref.OutputGuardrail("ssn_block", blocked)]}
    in_guard = {"input_guardrails": [ref.InputGuardrail("pii_block", ref._pii_check)]}
    passing, tripping = measure(ref, CLEAN, out_guard), measure(ref, DIRTY, out_guard)
    early = measure(ref, DIRTY, in_guard)
    across = measure(ref, CLEAN, in_guard, carry_input=True)
    return {
        "passing": {k: tripping[k] for k in ("turns", "tools", "handoffs", "spans")},
        "tripping": {k: tripping[k] for k in ("turns", "tools", "handoffs", "spans")},
        "same_work": all(passing[k] == tripping[k]
                         for k in ("turns", "tools", "handoffs", "spans")),
        "tripped": tripping["tripped"].which, "reason": tripping["tripped"].reason,
        "attributes": sorted(k for k in vars(tripping["tripped"])),
        "early_turns": early["turns"], "early_tools": early["tools"],
        "early_spans": early["spans"],
        "ratio": round(tripping["spans"] / early["spans"], 1),
        "across_inputs": across["inputs"],
        "across_output": across["output"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the refusal costs exactly what the answer would have cost",
            all([result["same_work"] is True, result["tripping"]["turns"] == 2,
                 result["tripping"]["tools"] == 1, result["tripping"]["handoffs"] == 1,
                 result["tripping"]["spans"] == 7, result["tripped"] == "output"]),
            f"both prompts run {result['tripping']['turns']} agent turns, "
            f"{result['tripping']['tools']} tool call and "
            f"{result['tripping']['handoffs']} handoff before the check -- "
            f"{result['tripping']['spans']} spans each ({result['same_work']}) -- and "
            "the tripping one raises after all of it",
        ),
        practice.Check(
            "FINDING: the input guardrail costs nothing by comparison",
            all([result["early_turns"] == 0, result["early_tools"] == 0,
                 result["early_spans"] == 1, result["ratio"] == 7.0]),
            f"blocking the same content on the way in raises after "
            f"{result['early_turns']} agent turns and {result['early_tools']} tool "
            f"calls, at {result['early_spans']} span against "
            f"{result['tripping']['spans']} -- {result['ratio']}x. The shipped "
            "_pii_check reads only the user's text, so it would work unchanged on "
            "either side",
        ),
        practice.Check(
            "FINDING: the trip leaves no final output anywhere",
            all([result["attributes"] == ["reason", "which"],
                 "social security" in result["reason"]]),
            f"Runner.run raises instead of returning, and GuardrailTripped carries "
            f"{result['attributes']} -- {result['reason']!r} -- with the offending text "
            "on neither. The span records it, so the only copy of what was blocked is "
            "in the trace",
        ),
        practice.Check(
            "FINDING: the guardrails do not re-run across a handoff",
            all([result["across_inputs"] == 1,
                 "ssn" in result["across_output"]]),
            f"input checks run once before the loop, so text entering through a "
            f"handoff's input is never checked: the run has {result['across_inputs']} "
            f"input-guardrail span and the final output is "
            f"{result['across_output']!r} -- content the guardrail was configured to "
            "refuse",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
