<!-- generated:start -->
# 14-agent-engineering / 23-otel-genai-conventions

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/23-otel-genai-conventions/) · upstream spec
`phases/14-agent-engineering/23-otel-genai-conventions/docs/en.md`

```bash
uv run demo practice run 23-otel-genai-conventions --ex 1
uv run demo explain 23-otel-genai-conventions --ex 1
uv run pytest demos/phases/14-agent-engineering/23-otel-genai-conventions
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Instrument your Lesson 01 ReAct loop with `invoke_agent` (INTERNAL) + per-tool spans. Send to… | code | T0 | `ex01_the_span_tree_has_no_ids_so_it_cannot_leave_the_process.py` |
| 2 | Add content capture in "references only" mode: prompts to SQLite, span attributes carry only… | code | T0 | `ex02_the_attribute_key_changes_name_with_the_capture_mode.py` |
| 3 | Read the spec for `gen_ai.data_source.id`. Wire it into your Lesson 09 Mem0 search. | code | T0 | `ex03_one_attribute_names_one_store_and_the_search_reads_two.py` |
| 4 | Set `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` and verify your attributes don… | explain | T0 | prose, below |
| 5 | Build a dashboard: "which tool errors correlate with which models" from GenAI attributes alone. | code | T0 | `ex05_the_dashboard_needs_two_attributes_the_spans_do_not_carry.py` |
<!-- generated:end -->

## Answers

### 1 — the instrumenting is easy and the sending is the lesson

Wrapping Lesson 01's `ToyLLM.respond` and `ToolRegistry.dispatch` produces 12
spans: one `invoke_agent research_bot` (INTERNAL, because the loop is
in-process), 6 `chat` spans and 5 `tool_call` spans across three tools. The run
still answers correctly, which is the point of wrapping rather than forking.

Sending it to Jaeger is where the design shows. `Span` carries `name`, `kind`,
`attributes`, `children`, `start_ns`, `end_ns` — six fields, none of them an
identity. Flattening the tree for export gives 12 rows and zero recoverable
parent links, because the only thing connecting a tool span to its agent is a
Python object reference. An OTLP exporter needs `trace_id`, `span_id` and
`parent_span_id`. That is the lesson's own "spans without parent links" pitfall,
sitting in the emitter rather than in the caller — and it is the constraint that
exercises 3 and 5 both run into from different directions.

Two smaller defects. `end_span()` takes no argument and pops whatever is on top,
so ending one span too many pops the root and the *next* `start_span` raises
`IndexError` from an empty stack: the failure surfaces one call later, in
unrelated code. A real tracer takes the span to end, or returns a context manager.
And `start_ns` comes from `perf_counter_ns`, whose zero is arbitrary — the module
never calls `time_ns`, so every `start_ns` is smaller than the wall clock and no
span can be placed on a timeline beside a log line or another service's trace.
Durations are recoverable; positions are not, and `start_time_unix_nano` cannot
be reconstructed from what is stored.

### 2 — references-only is the default; the SQLite swap exposes the naming problem

`Tracer` already defaults to `capture_inline=False` and writes content to an
external store, so what the exercise adds is durability. Reimplementing
`ExternalContentStore`'s three methods over stdlib `sqlite3` is a drop-in: a
3-span trace writes 6 rows, references are integer row ids, and grepping the
whole trace for the card number in the prompt returns 0 hits where inline capture
returns 3 — while `SELECT` finds it in 3 rows, which is the point. The content
still exists; it is somewhere with an access-control story, and the trace ops can
read does not have it.

The finding worth carrying out of this is the attribute name. Inline capture
writes `gen_ai.input.messages`; references-only writes
`gen_ai.input.messages.reference_id`. The two content key sets share zero
members, while the non-content keys are identical. So a dashboard query, an alert
or a tail-sampling rule written against one mode silently matches nothing under
the other — and the mode is a constructor flag that nothing downstream can see.
Flipping `capture_inline` in a config change is enough to empty a dashboard with
no error anywhere. The convention's own attribute names do not encode the
distinction, so it has to be encoded somewhere a consumer can query.

Inline capture is also lossy without saying so: a 516-character prompt is kept as
200, dropping 316, with zero attributes recording that anything was dropped. A
truncated prompt is indistinguishable from a short one, which makes any analysis
of prompt length or content silently wrong at the tail.

And the reference id is a per-store counter, so two `ExternalContentStore`s each
mint `content_001` for different content. A reader holding a span cannot tell
which store to ask. SQLite's `rowid` is per-database rather than per-process,
which does not solve the problem but moves it somewhere that can be namespaced —
the production version of this attribute is a URI, not an integer.

### 3 — the spec says "which store was consulted" and the search consults two

`gen_ai.data_source.id` is one line of spec: for RAG, which corpus or store was
consulted. Wiring it into Lesson 09's `Mem0.search` turns that line into a
question the attribute cannot answer. The search reads the vector store *and* the
KV store on every call and fuses the results — instrumented, that is
`VectorStore.search` once and `KVStore.by_user` once for a single query. Of the 5
returned records, 1 came from the vector index and 4 were reached only through
KV. The attribute holds one string, and here the store it names supplied the
minority of the answer.

The second finding is sharper. `Mem0` owns a `GraphStore`, `add` writes 5 edges
into it, and `search` reads it zero times. A trace carrying one accurate-looking
data source id is accurate about the wrong thing: it says a corpus was consulted
and cannot say that a maintained, written-to index was not. A per-store
attribute would have made a silently unused index visible in a dashboard; a
single id makes it invisible. That is a real production failure mode — the index
you are paying to keep fresh and nothing queries.

The shipped id is also a literal written at the call site (`"corpus://mem0/default"`)
rather than a property of the store. Deriving it from the stores actually read
gives `mem0://kv` and `mem0://vector` — two values, from objects that know their
own identity, and correct without anyone maintaining them.

Finally, the attribute sits on a span that carries no model. The `tool_call` span
holds `gen_ai.data_source.id`, `gen_ai.operation.name`, `gen_ai.tool.name` and
names a model or provider zero times, so "which corpus did this model consult"
needs the parent — and `Span` has no parent field. Exercise 1's identity gap
blocks the attribution the spec exists to enable.

### 4 — the opt-in pins names; verifying it means asserting on what arrives

**Stability** states the situation plainly: most GenAI conventions are
experimental as of March 2026, and the opt-in is

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

with Datadog v1.37+ mapping GenAI attributes natively into its LLM Observability
schema, and Grafana, Honeycomb and Jaeger supporting the raw attributes.

What the variable actually controls is worth being precise about, because the
exercise's phrasing ("verify your attributes don't get renamed by the collector")
points at the right worry and the slightly wrong actor. The opt-in is read by the
*instrumentation libraries* in your process, not by the collector. It selects
which generation of attribute names the SDK emits — old, new, or both during a
migration window. A collector can also rename things, via `transform` or
`attributes` processors, but it does that because someone configured it to,
not because of this variable. So there are two renaming surfaces and the env var
addresses one.

Verifying it is therefore a test, not an inspection, and it has three layers.

**At emit.** Capture the spans your SDK produces in-process, before any exporter,
and assert on the exact attribute key set: `gen_ai.provider.name`,
`gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.agent.name`,
`gen_ai.operation.name`, `gen_ai.data_source.id`. Exercise 2 is this test in
miniature — it compares two key sets and finds zero overlap — and the same shape
catches a library upgrade that renames `gen_ai.system` to `gen_ai.provider.name`
underneath you.

**At the collector.** Run the collector against a recorded span batch and diff
the output keys against the input keys. Any rename that is not in your own
processor config is a surprise, and the diff is the assertion. Doing this in CI
is the only way to notice a collector image bump that changes defaults.

**At the backend.** Query for one known attribute after a deploy and assert the
row count is non-zero. This is the cheapest and the last to fail, and it is the
one that catches a backend-side schema mapping changing — Datadog's native
mapping is a translation layer, and translations have versions too.

The failure this prevents is the one the lesson lists: attributes renamed on a
backend upgrade, every dashboard built on them going quietly empty. Nothing
errors. The graphs just flatten to zero, which reads like a drop in traffic. The
defence is that the attribute names are an interface between your process and
your dashboards, and interfaces get contract tests.

One practical note: set the variable explicitly, including when you want current
behaviour. An unset variable means "whatever this library version defaults to",
which is a value that changes without a deploy of yours.

### 5 — the dashboard returns nothing, and the reason is structural

"From GenAI attributes alone" is what makes this a design exercise. A backend
receives flat spans and groups by attribute; it cannot walk a tree. The shipped
`tool_call` span carries `gen_ai.data_source.id`, `gen_ai.operation.name`,
`gen_ai.tool.name` — no error, no model — so grouping tool failures by model over
40 tool spans returns zero rows. Adding two attributes, `error.type` and
`gen_ai.request.model`, makes the same grouping return a full 8-row table across
4 tools and 2 models.

Both additions are corrections to real gaps. `Span` has six fields and none is a
status, and no span in the lesson's own `main()` carries an attribute matching
`error` or `status`: a trace in which nothing can fail is a trace where this
dashboard's subject does not exist. And `gen_ai.request.model` lives on `chat`
and `invoke_agent` while the error lives on `tool_call`, so joining them needs
the parent link that exercise 1 showed is absent. Propagating the model down onto
the tool span is duplication, and it is duplication on purpose — flat spans are
the interface a backend actually has.

With the attributes present the correlation is worth having. Over 40 seeded tool
calls, `claude-opus-4-6` fails 20.0% and `gpt-5-mini` 65.0%, and the failures
concentrate rather than spread: `web_search` fails 80.0% on the weaker model
against 20.0% on the stronger. A single-number tool error rate of 42.5%
describes neither model and neither tool. That is the argument for the group-by
over the headline, and it is the same argument exercise 2 of Lesson 20 makes
about mean trajectory efficiency: the aggregate is the statistic that hides the
thing you would act on.
