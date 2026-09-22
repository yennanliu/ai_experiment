"""Exercise 1 — max_hops counts tool calls, so it is not a hop counter.

    Add a handoff hop counter: refuse after N transfers. Trace the behavior.

Reading of the exercise: `Runner` already has `max_hops=3`, and the name is
wrong -- the loop it bounds runs once per *policy step*, so a tool call
spends the same budget as a transfer. A handoff counter is therefore a second
counter, and the reason to want one is visible the moment an agent uses a
tool before transferring.

**ANSWER: a transfer counter that refuses after N.** With `max_handoffs=1` a
chain that transfers twice stops at the second with
`error: handoff limit 1 reached`, after **1** transfer and **2** agent spans.
With the limit at **2** the same chain completes and returns the third
agent's answer.

**FINDING: two tool calls exhaust the budget before the transfer lands.**
An agent that calls a tool twice and then transfers uses **3** of
`max_hops=3` on its own turns, so the target agent never runs. The trace
shows **3** agent spans, **2** tool spans and **1** handoff span -- the
transfer is recorded on the way out.

**FINDING: an exhausted run returns the empty string and passes the output
guardrail.** Falling out of `for hop in range(self.max_hops)` leaves
`final_output` at its initial `""`, which the shipped `length_cap` check
accepts at **0** characters. A run that did no work is indistinguishable from
one that answered briefly.

**FINDING: the trace records a transfer that never arrived.** There is a
`handoff.transfer_to_billing` span and **0** `agent.billing` spans, and the
run returns `''`. Read the handoffs and the transfer happened; read the agent
spans and nobody received it. Nothing in the trace says the loop ran out.

Structure: `bounded()` is `Runner.run` with a second counter; the agents and
policies are the lesson's own.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "16-openai-agents-sdk"


def chain(ref):
    """triage -> billing -> escalation, each transfer declared as a Handoff."""
    escalation = ref.Agent("escalation", "final stop",
                           lambda text: {"kind": "final", "text": "escalated"})
    billing = ref.Agent("billing", "handle refunds",
                        lambda text: {"kind": "handoff", "to": "escalation",
                                      "input": text},
                        handoffs=[ref.Handoff(target=escalation)])
    triage = ref.Agent("triage", "route",
                       lambda text: {"kind": "handoff", "to": "billing",
                                     "input": text},
                       handoffs=[ref.Handoff(target=billing)])
    return triage


def tool_heavy(ref):
    """An agent that spends two policy steps on tools before transferring."""
    target = ref.Agent("billing", "handle refunds",
                       lambda text: {"kind": "final", "text": "billing handled"})
    calls = {"n": 0}

    def policy(text):
        calls["n"] += 1
        if calls["n"] <= 2:
            return {"kind": "tool", "tool": "lookup", "args": {"key": "x"}}
        return {"kind": "handoff", "to": "billing", "input": text}

    triage = ref.Agent("triage", "route", policy,
                       tools=[ref.FunctionTool("lookup", "look up", lambda key: key)],
                       handoffs=[ref.Handoff(target=target)])
    return triage


def bounded(ref, runner, agent, text, max_handoffs):
    """Runner.run with a transfer counter beside the step counter."""
    current, message, transfers, output = agent, text, 0, ""
    for hop in range(runner.max_hops):
        span = ref.Span(name=f"agent.{current.name}", attributes={"hop": hop})
        runner.trace.children.append(span)
        decision = current.policy(message)
        if decision["kind"] == "final":
            output = decision["text"]
            break
        if transfers >= max_handoffs:
            output = f"error: handoff limit {max_handoffs} reached"
            break
        handoff = next(h for h in current.handoffs
                       if h.target.name == decision["to"])
        span.children.append(ref.Span(name=f"handoff.{handoff.tool_name}"))
        current, message, transfers = handoff.target, decision["input"], transfers + 1
    return output, transfers, runner.trace


def fresh(ref):
    return ref.Runner(output_guardrails=[ref.OutputGuardrail("length_cap",
                                                             ref._length_check)],
                      max_hops=3, trace=ref.Span(name="run"))


def spans(trace, prefix):
    return sum(1 for child in trace.children if child.name.startswith(prefix)) + sum(
        1 for child in trace.children for grand in child.children
        if grand.name.startswith(prefix))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tight, transfers, tight_trace = bounded(ref, fresh(ref), chain(ref), "refund", 1)
    loose, loose_transfers, _ = bounded(ref, fresh(ref), chain(ref), "refund", 2)
    runner = fresh(ref)
    exhausted = runner.run(tool_heavy(ref), "refund please")
    return {
        "tight": tight, "transfers": transfers,
        "tight_agents": spans(tight_trace, "agent."),
        "loose": loose, "loose_transfers": loose_transfers,
        "exhausted": exhausted,
        "agent_spans": spans(runner.trace, "agent."),
        "handoff_spans": spans(runner.trace, "handoff."),
        "tool_spans": spans(runner.trace, "tool."),
        "max_hops": runner.max_hops,
        "guardrail_passed": ref._length_check(exhausted),
        "guardrail_spans": spans(runner.trace, "output_guardrail."),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the second transfer is refused, and lifting the limit completes it",
            all([result["tight"] == "error: handoff limit 1 reached",
                 result["transfers"] == 1, result["tight_agents"] == 2,
                 result["loose"] == "escalated", result["loose_transfers"] == 2]),
            f"with max_handoffs=1 the chain stops at the second transfer with "
            f"{result['tight']!r} after {result['transfers']} transfer and "
            f"{result['tight_agents']} agent spans; at 2 the same chain completes with "
            f"{result['loose']!r} after {result['loose_transfers']} transfers",
        ),
        practice.Check(
            "FINDING: two tool calls exhaust the budget before the transfer lands",
            all([result["agent_spans"] == 3, result["tool_spans"] == 2,
                 result["handoff_spans"] == 1, result["max_hops"] == 3]),
            f"an agent that calls a tool twice and then transfers uses "
            f"{result['agent_spans']} of max_hops={result['max_hops']} on its own turns: "
            f"{result['tool_spans']} tool spans, then {result['handoff_spans']} handoff "
            "span on the last one. The cap counts policy steps, so the transfer is "
            "recorded and the target agent never runs",
        ),
        practice.Check(
            "FINDING: an exhausted run returns '' and passes the output guardrail",
            all([result["exhausted"] == "", result["guardrail_passed"][0] is True,
                 result["guardrail_spans"] == 1]),
            f"falling out of range(max_hops) leaves final_output at its initial "
            f"{result['exhausted']!r}, and length_cap accepts it "
            f"({result['guardrail_passed']}). A run that did no work is "
            "indistinguishable from one that answered briefly",
        ),
        practice.Check(
            "FINDING: the trace records a transfer that never arrived",
            all([result["handoff_spans"] == 1, result["agent_spans"] == 3,
                 result["guardrail_spans"] == 1, result["exhausted"] == ""]),
            f"the trace holds {result['handoff_spans']} handoff span, "
            f"{result['agent_spans']} agent spans -- none of them the target -- and "
            f"{result['guardrail_spans']} guardrail span, with a final output of "
            f"{result['exhausted']!r}. Reading the trace, the transfer happened; "
            "reading the agent spans, nobody received it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
