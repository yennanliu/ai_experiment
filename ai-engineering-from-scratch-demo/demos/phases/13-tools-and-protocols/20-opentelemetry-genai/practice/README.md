<!-- generated:start -->
# 13-tools-and-protocols / 20-opentelemetry-genai

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/20-opentelemetry-genai/) · upstream spec
`phases/13-tools-and-protocols/20-opentelemetry-genai/docs/en.md`

```bash
uv run demo practice run 20-opentelemetry-genai --ex 1
uv run demo explain 20-opentelemetry-genai --ex 1
uv run pytest demos/phases/13-tools-and-protocols/20-opentelemetry-genai
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Count the spans and identify which is CLIENT vs INTERNAL. | code | T0 | `ex01_the_kind_says_where_the_work_goes_not_what_the_code_does.py` |
| 2 | Turn on content capture (env var) and confirm `gen_ai.content.prompt` and `gen_ai.content.com… | code | T0 | `ex02_truncation_is_not_redaction_and_the_completion_is_a_literal.py` |
| 3 | Add the tool-execution metric `gen_ai.tool.execution.duration` and emit it as a histogram sam… | code | T0 | `ex03_the_span_already_has_the_duration_the_histogram_needs.py` |
| 4 | Propagate a traceparent from a parent agent span into an MCP request's `_meta.traceparent` fi… | code | T0 | `ex04_the_traceparent_is_recorded_and_never_sent.py` |
| 5 | Read the OTel GenAI semconv spec. Identify one attribute listed in the semconv that this less… | code | T0 | `ex05_the_missing_attribute_is_the_one_that_says_it_went_wrong.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib, and all five ship code — including exercises 1 and
5, which read as observation and reading tasks.

Telemetry is a contract between what a producer emits and what a reader can
ask, and the five exercises keep finding the same shape of gap: **the value
exists and is in the wrong place.** A traceparent on a span instead of in a
request; a duration in nanoseconds for a metric specified in seconds; a
completion field holding a constant.

### 1 — the kind says where the work goes, not what the code does

**ANSWER: 9 spans in 1 trace — 5 CLIENT, 4 INTERNAL.** CLIENT is `llm.chat`
×2 and `mcp.call` ×3; INTERNAL is `agent.invoke_agent` and `tool.execute` ×3.

**FINDING: the rule is whether the work leaves the process.** `llm.chat` is
CLIENT although its body only sleeps; `tool.execute` is INTERNAL although it
is the slowest work here, because the crossing happens in its `mcp.call`
child. A parent can be INTERNAL and contain CLIENT.

**FINDING: the two spans per tool call are one operation named twice.** **6**
spans carry `operation.name == "execute_tool"` where **3** executions
happened.

**FINDING: `SPANS` is a module-level list with no reset.** A second
`agent_loop` leaves **18** spans in **2** traces.

### 2 — truncation is not redaction, and the completion is a literal

**ANSWER: the events appear — 2 per `llm.chat`, 4 in the trace.** Capture off
gives **0** events and an otherwise identical trace.

**PII: truncation is not redaction.** `prompt[:200]` is a length cap applied
before anything inspects it. PII in the first 200 characters is captured; PII
at character 400 is protected by accident. A slice is not a policy.

**FINDING: the completion event does not carry the completion.** Both hold
`"sample completion"` — **1** distinct output across **2** calls. The privacy
cost is paid in full by a field that does not yet contain anything.

**FINDING: the switch is read once at import, and content rides the same
span.** No per-request, per-tenant or per-span setting; no separate sink,
sampling or retention.

### 3 — the span already has the duration the histogram needs

**ANSWER: one sample per tool call, from `tool.execute`, in seconds.** The
sum equals the span durations because it *is* them.

**FINDING: timing it separately would disagree with the span.**

| source | samples | measures |
|---|---:|---|
| `tool.execute` | 3 | crossing + local execution |
| `mcp.call` | 3 | the crossing only |
| every `execute_tool` span | 6 | both, double-counted |

**FINDING: the histogram needs a unit the span does not state.** `start_ns`
against a metric specified in seconds — one factor of 1e9, in one place.

**FINDING: the attributes are a cardinality choice.** Including
`gen_ai.tool.call.id` turns **1** series into **3**.

### 4 — the traceparent is recorded and never sent

**ANSWER: the header rides `params._meta` and the server recovers the
trace.** One trace across two processes.

**FINDING: the shipped traceparent goes to the span, not to a request.**
`fake_mcp_call` returns a two-key dict — no `params`, no `_meta`, no message
at all. An attribute on the client span tells the *exporter* what the server
should have been told.

**FINDING: without it the server starts a new trace, and nothing looks
wrong.** Two internally well-formed traces that no query joins — a missing
header is a join that silently returns nothing.

**FINDING: the format carries the sampling decision.** Copying only the trace
id produces traces with holes rather than traces that are absent.

### 5 — the missing attribute is the one that says it went wrong

**ANSWER: `error.type`.** Chosen because its absence changes what a trace can
*answer*, not how much detail it has: **0** error attributes across **9**
spans means a failing call ends a span that looks successful.

**FINDING: the absence is structural.** `Span`'s **9** fields include no
status, so `error.type` lives in `attrs` and there is nowhere for the
`UNSET`/`OK`/`ERROR` it normally accompanies.

**FINDING: the request half of the semconv is missing too, and costs less.**
Temperature, max_tokens and finish_reasons absent; both models and both token
counts present — the emitted set is the billing half.

**FINDING: an error must propagate or the root stays green.** Marking only
the span that raised leaves a trace-level failure query empty. That is a
policy, not an attribute.
