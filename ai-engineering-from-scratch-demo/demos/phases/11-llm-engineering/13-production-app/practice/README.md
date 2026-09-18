<!-- generated:start -->
# 11-llm-engineering / 13-production-app

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/11-llm-engineering/13-production-app/) · upstream spec
`phases/11-llm-engineering/13-production-app/docs/en.md`

```bash
uv run demo practice run 13-production-app --ex 1
uv run demo explain 13-production-app --ex 1
uv run pytest demos/phases/11-llm-engineering/13-production-app
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add RAG integration. Build a simple in-memory vector store with 20 documents. When the templa… | code | T0 | `ex01_one_word_apart_scores_lower_than_completely_unrelated.py` |
| 2 | Implement real function calling. Add a tool registry (from Lesson 09) to the service. When a… | code | T0 | `ex02_the_tool_runs_and_the_answer_is_byte_identical.py` |
| 3 | Build a cost alerting system. Track cost per user per day. When a user exceeds $0.50/day, swi… | code | T0 | `ex03_the_hundred_dollar_threshold_is_a_hundred_and_thirteen_thousand_requests.py` |
| 4 | Implement prompt versioning with rollback. Store all prompt versions with timestamps. Add an… | code | T0 | `ex04_every_error_is_filed_under_a_version_that_does_not_exist.py` |
| 5 | Add OpenTelemetry tracing. Instrument every component (cache lookup, guardrail check, LLM cal… | code | T0 | `ex05_the_span_that_is_ninety_nine_point_nine_percent_of_the_trace_is_a_random_number.py` |
<!-- generated:end -->

## Answers

The capstone is pure stdlib, so all five exercises are **T0** and run in CI. The
service is `async` with a 15% simulated failure rate, so every solution drives
it through one small helper that pins `random` to a seed and swaps
`asyncio.sleep` for a recorder — the simulated milliseconds are reported, not
slept, which is the only way to keep 200-call measurements inside a T0 budget.

Two facts run through the whole lesson. `simple_embedding` is `sha256` of the
whole string, so nothing built on it can be semantic; and
`call_llm_with_retry` picks one of **three constant strings** by testing the
prompt for `"code"`, `"review"` and `"context"`, so nothing put *into* a prompt
can change what comes out except by hitting one of those substrings.

### 1 — one word apart scores lower than completely unrelated

| | cosine against `DOCS[0]` |
|---|---:|
| six single-word rewrites of it | **0.7427** |
| the 19 documents that share nothing | **0.7560** |
| all 190 document pairs | 0.7561, sd 0.0371, max 0.8480 |

**ANSWER: recall@3 is 3 of 10 and the mean rank of the right document is 8.8 of
20**, against 10.5 for a coin. Ranks: 14, 3, 5, 13, 5, 3, 14, 1, 17, 13.

**MECHANISM: the near set scores *lower* than the far set** and the two ranges
overlap. `simple_embedding` is not a function of the words, so it cannot be a
function of the meaning.

**FINDING: 0.92 is out of reach.** The entries are `int(hex, 16) / 255`, all
non-negative, so every vector is in one corner of the space. The maximum over
190 pairs is 0.8480, below `SemanticCache`'s threshold — the "semantic" cache
can only hit an exact repeat, and reports `similarity` 1.0 when it does.

**ANSWER: "quality with and without RAG" compares two constants** — the `rag`
string whenever the prompt says "Context:", the `general` string otherwise.

**ANSWER: retrieval is under 0.1 ms; the LLM call is 130.2 ms of
`asyncio.sleep`.** The number the exercise says to track separately is the only
one measuring real work, and it is the small one.

### 2 — the tool runs and the answer is byte-identical

```text
3 tools, 20 labelled queries   router 20/20
same 20 with the tool result   answers changed: 0/20
```

**MECHANISM: the reply is a substring lookup on the prompt.** A search result
mentioning "the context of the treaty" flips it to the RAG constant; one
mentioning a "code snippet" flips it to the code-review constant. Both are
accidents of vocabulary, not the tool's data being used.

**FINDING: `tools_used` has to be added in three places.** The normal,
cache-hit and blocked responses carry **10, 6 and 5** keys and share only
`request_id`, `cost_usd`, `latency_ms`. A field added on the normal path is
missing on a cache hit — where it would have to be replayed from an entry that
stores only the response text.

**FINDING: no template has a slot for a tool result.** The three templates have
`code`, `context`, `query` between them; `template.format(**variables)` drops
an extra key silently and raises `KeyError('context')` on a missing one, before
any model is called.

**MEASUREMENT: the prompt doubles, 18 → 36 tokens, and the request goes
$0.000855 → $0.0009 (+5.3%)** for a byte-identical answer.

### 3 — the $100 threshold is 113,637 requests away

| | gpt-4o | gpt-4o-mini |
|---|---:|---:|
| one request (28 in / 81 out) | $0.00088 | $0.0000528 |
| requests for one user to reach $0.50 | **569** | 9,470 |
| requests for the service to reach $100 | **113,637** | 1,893,940 |

**ANSWER: the traffic spike the exercise asks for is 6.3 hours long** at the
lesson's own 0.2 s of simulated latency per call.

**ANSWER: the per-user switch cuts 94.0%** and is the right mitigation — the
threshold is what is in the wrong units.

**FINDING: `CostTracker` has no day in it.** Seven fields, all keyed by name
alone. "Cost per user per day" has to be rebuilt on `RequestLog.timestamp`,
the only place the day survives.

**FINDING: the configured primary model is used by 1 of the 3 templates.**
`PRIMARY_MODEL` is `claude-sonnet-5`, but `PromptTemplate.model` defaults to
`gpt-4o` and only `code_review` overrides it, so all 100 requests bill
`gpt-4o`. Routing `general_chat` to the configured primary would have *raised*
the bill by **47.6%**.

**FINDING: emergency mode's cache-only rule leaks answers between users.**
PII is redacted *before* the cache is consulted and the cache key is the
redacted text, so four users with four different SSNs produce **1 cache entry,
3 hits at similarity 1.0 and one shared answer**. There is also no length check
to extend: rejecting over 2,000 tokens means rejecting at **1,501 words**.

### 4 — every error is filed under a version that does not exist

| 400 requests, one user each | v1 | v2 | "blocked" |
|---|---:|---:|---:|
| requests | 340 | 40 | 20 |
| errors logged | 0 | 0 | **20** |
| `output_length` mean / sd | 421.0 / 0.00 | 421.0 / 0.00 | — |

**ANSWER: the per-version error rate is 0.000 for both arms and cannot be
anything else.** Every error in the log is a guardrail block, and
`_blocked_response` hardcodes `prompt_version="blocked"` — a version in no
template dict. The only other way `error` is set is the whole three-model
fallback chain failing: **p = 6.59e-15** per request. "2× the error rate" is 0
against 0, so the automatic rollback can never fire.

**FINDING: `eval_results` carries neither an error nor a rating.** Seven keys:
`request_id`, `template`, `version`, `model`, `output_length`, `latency_ms`,
`timestamp`. Of the exercise's three metrics, latency is there, the error rate
is in another list under a phantom version, and user ratings exist nowhere in
the lesson.

**FINDING: the only per-version number that varies is the clock.**
`output_length` has sd 0.00 and the same mean in both arms, because v1 and v2
both miss `"code"`, `"review"` and `"context"` and receive the same constant.
The arms' mean latency differs by less than one standard deviation of either.

**FINDING: `PromptTemplate` has no timestamp** — five fields, and
`PROMPT_TEMPLATES` is a module-level dict literal, so storing versions with
timestamps needs a field *and* a store.

**FINDING: 100 requests on the new version is 915 users away.** The experiment
gives **205 of 2,000** users the variant (10.25%), deterministically by
`md5(user:experiment) % 100 < 10`. All 7 users in the lesson's own demo land in
v1, so the rollback window never opens there.

### 5 — the span that is 99.98% of the trace is a random number

| span | ms |
|---|---:|
| `llm_call` (simulated) | **130.2** |
| `output_guardrail` | 0.017 |
| `cache_lookup` | 0.012 |
| `input_guardrail` | 0.003 |
| `cost_calculation` | 0.0003 |
| all four real spans | **0.032** |

**MECHANISM: the LLM span's duration is `random.uniform(0.1, 0.3)` plus a retry
tail.** Over 200 seeded calls: median **207.0 ms**, mean 417.3, p95 1781.9, max
4569.7, and **28 of 200 over a second** — `call_llm_with_retry` fails 15% of
the time and then sleeps `min(2 ** attempt + uniform(0, 1), 10)`.

**FINDING: the guardrail span runs before the cache span, so a hit pays for
it.** The cheapest path in the service still scans 7 injection and 4 PII
patterns. The hit's `latency_ms` is 0.03 ms against the miss's 0.4 ms.

**FINDING: `latency_ms` stops before the streaming starts.**
`handle_streaming_request` stamps `latency_ms` in `handle_request` and *then*
streams 61 words at `uniform(0.02, 0.08)` each — **2.87 s**, 99.998% of the
request, in neither `latency_ms` nor `eval_results`. The number the exercise
asks to decompose excludes the largest component of a streamed request.
