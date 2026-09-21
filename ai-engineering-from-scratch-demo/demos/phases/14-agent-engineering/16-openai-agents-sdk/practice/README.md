<!-- generated:start -->
# 14-agent-engineering / 16-openai-agents-sdk

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/16-openai-agents-sdk/) · upstream spec
`phases/14-agent-engineering/16-openai-agents-sdk/docs/en.md`

```bash
uv run demo practice run 16-openai-agents-sdk --ex 1
uv run demo explain 16-openai-agents-sdk --ex 1
uv run pytest demos/phases/14-agent-engineering/16-openai-agents-sdk
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a handoff hop counter: refuse after N transfers. Trace the behavior. | code | T0 | `ex01_max_hops_counts_tool_calls_so_it_is_not_a_hop_counter.py` |
| 2 | Implement `nest_handoff_history` as an option — collapse prior messages into one summary befo… | code | T0 | `ex02_there_is_no_history_to_nest_only_the_last_string.py` |
| 3 | Write a blocking output guardrail. Compare latency on prompts that would trip it vs ones that… | code | T0 | `ex03_the_output_guardrail_bills_the_whole_run_before_it_answers.py` |
| 4 | Wire `add_trace_processor` to a JSON logger. What shape does it emit per span? | code | T0 | `ex04_the_span_has_no_id_no_parent_and_no_clock.py` |
| 5 | Read the SDK docs. Port your stdlib toy to `openai-agents-python`. What did you model wrong? | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

All five are T0 and stdlib. Four ship code; exercise 5 is a port to the real
SDK, so it is answered in prose.

The recurring finding is that **the run's state is one string and the trace
is the only history**. `Runner` carries `current_input`, which every step
overwrites: after a tool call it reads `tool lookup returned: …` and the
user's question is gone, so a handoff hands the specialist what the previous
agent's *tool* said rather than what the user asked (exercise 2). The span
tree is the only place the earlier turns survive, and a `Span` has **3**
fields — `name`, `attributes`, `children` — with no id, no parent and no
clock (exercise 4).

The second thread is that **the two counters in the system count the wrong
things**. `max_hops=3` bounds *policy steps*, so two tool calls exhaust the
budget before a transfer lands and the run returns `""` — which the shipped
`length_cap` guardrail accepts at 0 characters (exercise 1). And the output
guardrail runs after the whole chain, so a refusal costs **100%** of what the
answer would have cost, against **1** span for the same check on the input
side (exercise 3).

What holds: the primitive set. Agent, Handoff, Guardrail, Span and a Runner
that raises a structured exception rather than crashing — all five are there
and all five behave as named, which is what makes the gaps measurable.

### 1 — `max_hops` counts tool calls, so it is not a hop counter

**ANSWER: a transfer counter that refuses after N.** With `max_handoffs=1` a
chain that transfers twice stops at the second with
`error: handoff limit 1 reached` after **1** transfer and **2** agent spans;
at **2** the same chain completes and returns the third agent's answer.

**FINDING: two tool calls exhaust the budget before the transfer lands.** An
agent that calls a tool twice and then transfers spends **3** of
`max_hops=3` on its own turns. The trace shows **3** agent spans, **2** tool
spans and **1** handoff span — the transfer is recorded on the way out — and
the target agent never runs. A cap named for hops that counts steps is a cap
whose meaning depends on how tool-happy the agent is.

**FINDING: an exhausted run returns `''` and passes the output guardrail.**
Falling out of `for hop in range(self.max_hops)` leaves `final_output` at its
initial `""`, and `length_cap` accepts it at **0** characters. A run that did
no work is indistinguishable from one that answered briefly — and the check
that was supposed to catch bad output is the one that waves it through.

**FINDING: the trace records a transfer that never arrived.** There is a
`handoff.transfer_to_billing` span and **0** `agent.billing` spans. Read the
handoffs and the transfer happened; read the agent spans and nobody received
it. Nothing anywhere says the loop ran out.

### 2 — there is no history to nest, only the last string

**ANSWER: a runner that accumulates turns and collapses them on transfer.**
Over a chain with **2** tool calls and **1** handoff it carries **3** turns
into the transfer — the question and both tool results — and nests them into
**1** summary of **68** characters, against the shipped runner's **1** turn
and **41** characters.

**FINDING: the shipped handoff drops the user's question.** The value passed
to the target contains the user's words **0** times and the last tool's
result **1** time. The SDK's own description of the handoff — "copy the
conversation context (or collapse it via `nest_handoff_history`)" —
presupposes a context to copy; this runtime has a variable.

**FINDING: nesting is a reduction, and the shipped runner has nothing to
reduce.** **3** turns to **1** is a **67%** drop in entries and **110**
characters to **68**. On a single-turn payload the option is a no-op, which
is the honest answer to "implement it as an option": the option is cheap and
the accumulation it presupposes is the work.

**FINDING: `input` is an escape hatch that bypasses both.** A policy may
return any `input` it likes on a handoff — here **13** characters appearing
in neither the history nor the tool results. Nesting is a *runner* policy
that a single *agent* return value overrides, which is worth knowing before
relying on it for context control.

### 3 — the output guardrail bills the whole run before it answers

**ANSWER: a blocking output guardrail, and the refusal costs exactly what
the answer would have cost.** Both prompts run **2** agent turns, **1** tool
call and **1** handoff before the check — **7** spans each — and the
tripping one raises after all of it.

**FINDING: the input guardrail costs nothing by comparison.** Blocking the
same content on the way in raises after **0** agent turns and **0** tool
calls, at **1** span: a **7.0x** difference on this chain. The lesson's own
guardrail modes are about the *guardrail LLM's* latency; this measurement is
about the *main* run's work, and it points the same way — anything decidable
from the input belongs on the input side. The shipped `_pii_check` reads
only the user's text and would work unchanged on either.

**FINDING: the trip leaves no final output anywhere.** `Runner.run` raises
rather than returning, and `GuardrailTripped` carries `which` and `reason`
and not the text that tripped it. The span records it, so the only copy of
what was blocked is in the trace — which matters because the trace is also
the thing most likely to be sampled away.

**FINDING: the guardrails do not re-run across a handoff.** Input checks run
once before the loop, so text entering through a handoff's `input` is never
checked: the run has **1** input-guardrail span and returns content the
guardrail was configured to refuse. The SDK's own scoping — "input
guardrails run on the first agent's input, output guardrails on the last
agent's output" — is a deliberate choice, and this is the hole it leaves.

### 4 — the span has no id, no parent and no clock

**ANSWER: a processor that walks the tree and emits one JSON object per
span.** The refund case yields **7** spans, each emitted as `name`, `depth`,
`parent`, `attributes`, with **1** root and depths **0–2**. Every line parses.

**FINDING: three of the four fields are reconstructed, not recorded.** `Span`
carries `name`, `attributes`, `children` and **0** of an id, a parent, a
start time or a duration. A processor attached after the run can rebuild the
tree from nesting and cannot rebuild anything about time — so `duration_ms`,
the field every tracing backend sorts by, is unavailable *in principle*
rather than merely unimplemented.

**FINDING: attribute keys are per-span-kind, so the log is not a table.**
Across **7** spans the union of attribute keys is **8** and **0** appear on
every span. JSON lines handle that; a columnar sink gets one column per span
kind, which is the practical reason tracing backends define a semantic
convention rather than accepting free-form attributes.

**FINDING: the trace is a mutable field, so two runs share it.** One run
leaves **2** children; two leave **4** on the same trace. The lesson's own
`main()` reassigns `runner.trace` before each case — the caller remembering
to do the runtime's job, which is exactly the kind of thing a real
`add_trace_processor` exists to take over.

### 5 — the handoff is a tool name and not a tool

*Cites "Handoffs as tools".*

**What I modelled wrong, in the order it would break a port.**

**1. Handoffs are a separate list and a separate `kind`, not entries in the
tool list.** The lesson is explicit: "The model sees
`transfer_to_billing_agent` in its tool list." In the toy, `Handoff` has a
`tool_name` property that nothing calls, `Agent.handoffs` is a field beside
`Agent.tools`, and the policy signals a transfer by returning
`{"kind": "handoff"}` — a control-flow branch the runtime interprets. The
model, if there were one, would never see a transfer tool at all. That
inverts the SDK's central design decision: making handoffs tools is what lets
the *model* choose them with the same machinery it uses for everything else,
and what makes `tool_choice` and tool guardrails apply to them.

**2. I modelled two guardrail flavours out of three, and the wrong axis of
the third.** The SDK has input, output *and* tool guardrails, and separately
a parallel/blocking mode. The toy has input and output only, always blocking,
and exercise 3 measures the consequence: the output check pays for the whole
chain. A tool guardrail is the thing that would have caught exercise 2's
`input` escape hatch, because that payload is an argument to a transfer.

**3. `Session` is missing entirely, which is why exercise 2 exists.** The
fourth primitive is "automatic conversation history across turns". The toy
has `current_input: str`. Every finding in exercise 2 — the dropped question,
the nothing-to-nest, the overridable payload — is a restatement of that one
omission, and `nest_handoff_history` is a *session* feature that cannot be
implemented without one.

**4. Tracing is a data structure rather than a subsystem.** No ids, no
parents, no clock, one mutable field shared across runs, and no
`add_trace_processor` — exercise 4 measures all four. The one that cannot be
retrofitted from outside is time.

**5. The counters are the caller's, not the runtime's.** `max_hops` bounds
steps and is checked by a `for` loop that silently returns `""` on exhaustion
(exercise 1). The SDK raises `MaxTurnsExceeded`, which is the difference
between a bug and an error.

**What the toy got right, and is worth keeping in a port:** guardrail trips
are structured exceptions carrying `which` and `reason` rather than sentinel
strings, and every guardrail evaluation emits a span whether it passed or
failed. Those two together are why exercise 3 could measure a refusal's cost
at all.
