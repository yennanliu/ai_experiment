<!-- generated:start -->
# 17-infrastructure-and-production / 13-llm-observability

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/13-llm-observability/) · upstream spec
`phases/17-infrastructure-and-production/13-llm-observability/docs/en.md`

```bash
uv run demo practice run 13-llm-observability --ex 1
uv run demo explain 13-llm-observability --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/13-llm-observability
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Your team on LangChain wants OSS self-hosted observability. Pick Langfuse or Opik and justify. | code | T0 | `ex01_langfuse_fits_the_team_word_for_word_and_the_lesson_misstates_its_license.py` |
| 2 | At 5M traces/day with Datadog quotes $150K/month, compute break-even for Arize AX. | code | T0 | `ex02_arize_breaks_even_below_150k_but_against_a_sampled_datadog_it_must_be_12_8x_cheaper_not_100x.py` |
| 3 | Design an OpenTelemetry GenAI attribute set your org's guideline should mandate on every LLM… | code | T0 | `ex03_the_lessons_attribute_set_leaves_two_of_its_five_alerts_blind_and_names_a_deprecated_key.py` |
| 4 | Argue whether Phoenix alone is sufficient for production. When does it not suffice? | explain | T0 | prose, below |
| 5 | Helicone is 20ms proxy overhead. At P99 TTFT 300 ms, is that acceptable? What if SLA is 100 ms? | code | T0 | `ex05_20ms_is_fine_with_headroom_above_300ms_and_at_100ms_the_sla_is_already_lost_so_log_async.py` |
<!-- generated:end -->

## Answers

### 1 — Langfuse fits the team word for word, and the lesson misstates its license

**Langfuse, unless the team already runs Comet, in which case Opik.** The
team's three requirements are three hard filters. Both tools pass all of them:

| | LangChain integration | OSS | self-host |
|---|---|---|---|
| Langfuse | `langfuse.langchain.CallbackHandler` | MIT core | Docker |
| Opik | `opik.integrations.langchain.OpikTracer` | Apache-2.0 | Docker |
| LangSmith | native | commercial | Enterprise only |

The lesson's own sweet spots break the tie. Langfuse's is "LangSmith-class
features but must self-host or stay on OSS license", which describes this
team exactly. Opik's is "ML teams already on Comet".

**The lesson gives Langfuse's license three different ways.** The Langfuse
section says "Core Apache / MIT", the summary says "MIT-licensed core", and
Key Terms says "MIT OSS". The repository's LICENSE (read 2026-09-26) is MIT
everywhere except its `ee/` directories, which hold 13 commercially licensed
features. Eight are in the web app: admin API, audit-log viewer, billing,
multi-tenant SSO, SSO settings, Salesforce sync, UI customization and
verified domains. Five are in the worker, among them `dataRetention`.
Traces, evals, prompts and datasets are outside `ee/`, so they are MIT.
Opik's LICENSE is Apache-2.0, with no such carve-out. For a team
self-hosting on the MIT core, SSO and automatic data retention are where the
two licenses differ, and retention is what this lesson's sampling section is
about.

The 50K-event free tier the lesson quotes is not an option at this scale.
Even at one event per trace, it covers 72 minutes of the reference
simulator's 1M-trace day, or 1/600 of the month. That is why self-hosting is
the requirement.

### 2 — Arize breaks even below $150K, but against a sampled Datadog it must be 12.8x cheaper, not 100x

At the reference's 4,500 bytes a trace, 5M traces/day is 22.5 GB/day and
675 GB/month. The quote therefore works out to **$222.22/GB**, or $1.00 per
1,000 traces. Zero-copy means you also pay for your own S3, which is $15.53
at 30-day retention.

| Arize AX has to beat | that costs | break-even Arize contract |
|---|---:|---:|
| Datadog, 100% retention (the quote) | $150,000 | **$149,984** |
| Datadog under the lesson's "5% success + errors + $$$" (391,366 of 5M kept) | $11,741 | **$11,725** |
| "100x cheaper", for comparison | | $1,500 |

Sampling alone cuts the bill 92% without changing vendor. A full-retention
Arize has to be **12.8x** cheaper than the quote to beat that, not 100x.

**The 100x is a ratio of two constants.** `OBSERVABILITY_INGEST_PER_GB /
ARIZE_AX_PER_GB` = 0.50 / 0.005 = 100, at every volume and every strategy.
The `arize` column leaves out the S3 bill, which is 4.6x larger ($15.53
against $3.38). Counted in, the reference's own rates give $337.50 against
$18.90, which is 17.9x.

**The code prices Datadog 444x below the quote.** At $0.50/GB, 100%
retention of 1M traces/day costs $67.50/month, or $2.25/day, in the code's
own table. The printout's "hundreds of $/day" is true only at the quote's
rate, where it is $1,000/day.

Arize's pricing page (read 2026-09-26) lists AX Pro at $50/month for 50K
spans, which is 14.4 minutes of this load, and Enterprise as "Custom". The
break-even above is therefore the ceiling for that negotiation.

### 3 — the lesson's attribute set leaves two of its five alerts blind and names a deprecated key

Names are checked against the OpenTelemetry GenAI spans spec
(`open-telemetry/semantic-conventions-genai`, read 2026-09-26). The spec is
still at Development status.

**Mandated on every call:**

- **Spec Required:** `gen_ai.operation.name`, `gen_ai.provider.name`.
- **Model:** `gen_ai.request.model`, plus `gen_ai.response.model`, so an
  alias is priced as the snapshot that answered.
- **Usage:** `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`,
  `gen_ai.usage.cache_read.input_tokens`.
- **Response:** `gen_ai.response.finish_reasons`, `gen_ai.request.stream`,
  `gen_ai.response.time_to_first_chunk`, `gen_ai.conversation.id`, and
  `error.type` on failure.
- **Org namespace:** `org.tenant_id`, `org.user_id`, `org.task`,
  `org.prompt_version`.
- **Off by default:** the spec's Opt-In content attributes,
  `gen_ai.input.messages`, `gen_ai.output.messages`,
  `gen_ai.system_instructions` and `gen_ai.tool.definitions`.

Each attribute is there because a signal reads it. With these 16, all of the
skill file's five alerts can be computed, and so can per-tenant cost:

| attribute set | alerts it cannot compute | spec problems |
|---|---|---|
| this mandate | none | none |
| the skill file's nine | P99 TTFT, prompt-cache hit rate | no `gen_ai.operation.name` or `gen_ai.provider.name`; uses deprecated `gen_ai.system` |
| `code/main.ts` tracer | P99 TTFT, prompt-cache hit rate, per-tenant cost | `gen_ai.system`; `gen_ai.usage.cached_input_tokens` is not a spec name |

`main.ts` has two further problems. Its sampler reads missing token counts
as `?? 0`, so a span that carries no usage data is never flagged as
high-cost. Its span name `chat.completion` is also not the spec's
`{operation} {model}`.

**The mandate is cheap enough to keep for every call.** The example span is
551 bytes, 12.2% of the reference's 4,500-byte trace. Keeping that for all
1M calls a day, plus full traces under the sampling rule, costs
$12.90/month at the reference's ingest rate. That is 19% of 100% retention
($67.50) and 2.4x sampling alone ($5.28). This is the lesson's "keep
aggregates always" made concrete: under sampling alone, 92% of calls leave
no record at all.

### 4 — Phoenix alone is a production trace store, but not a gateway or an archive

*Draws on "Phoenix (Arize) — telemetry-first, OpenTelemetry-native".*

**Phoenix alone is enough for one team's OTel-instrumented app at moderate
volume.** The lesson says it is "not designed as persistent production
backend". The current code says otherwise. Phoenix's `config.py` (read
2026-09-26) accepts a PostgreSQL `PHOENIX_SQL_DATABASE_URL`, including a
read replica. It has a per-project trace retention policy
(`PHOENIX_DEFAULT_RETENTION_POLICY_DAYS`, 0 when unset), and it has
authentication with OAuth2 and LDAP settings. A single team that sends
OpenTelemetry, wants evals and RAG drift views, and needs nothing in the
request path can run Phoenix alone in production.

**It stops being enough in four cases:**

1. **Anything in the request path.** Phoenix receives spans after the fact.
   It does not fail over, cache, rate-limit or enforce per-key budgets,
   which are the jobs the lesson gives Helicone and the gateway layer. An app
   that needs those needs a gateway in front, whatever stores the traces.
2. **Volume that belongs in a lake.** At the reference's 4,500 bytes a
   trace, 1M traces/day is 4.5 GB/day, or 135 GB at 30 days, which one
   Postgres handles. At the lesson's Arize AX threshold of 10M/day it is
   1.35 TB at 30 days, and long-term analysis is what the lesson routes to
   Iceberg plus Arize AX or DuckDB.
3. **Sampling.** The lesson's rule (100% errors, 100% high-cost, 5% of
   successes) has to run before the store, in the SDK or an OpenTelemetry
   Collector. Nothing in Phoenix alone decides what not to keep, and I did
   not find a server-side sampler in its configuration.
4. **Offering it to others.** Phoenix's license is Elastic License 2.0,
   which forbids providing the software "to third parties as a hosted or
   managed service". A platform team selling observability to its customers
   cannot build on it. Self-hosting for your own use is fine.

The lesson's production pattern (gateway plus eval platform, glued by OTel)
follows from cases 1 and 2. Phoenix is the eval-and-trace half. The gateway
half and the archive are separate jobs, not missing Phoenix features. I did
not check whether Phoenix ships alerting for the skill file's five metrics.

### 5 — 20 ms is fine with headroom above 300 ms; at 100 ms the SLA is already lost, so log async

This lesson's code has no latency model, so the distribution used here is
lesson 08's `synth_workload()` (2,000 requests, seed 7), rescaled so its P99
TTFT is 300 ms. A constant 20 ms shifts every request by the same amount:

| | before | with the proxy |
|---|---:|---:|
| P99 TTFT | 300 ms | 320 ms (+6.7%) |
| requests within 300 ms | 99.05% | 97.1% |
| requests within 100 ms | 75.8% | 69.6% |

- **At 300 ms: acceptable if the SLA has 20 ms of headroom.** Against any
  SLA at or above 320 ms it passes, as does lesson 19's "for P99 < 500 ms,
  any" gateway. Against an SLA equal to the 300 ms baseline it fails.
- **At a 100 ms SLA: no, and the SLA is already missed without Helicone.**
  The baseline P99 is 3x the budget. The proxy takes 20% of the budget,
  which leaves the model 80 ms.

The remedy at 100 ms is Helicone's async integration. Its proxy-vs-async
page (read 2026-09-26) says async is "not on the critical path", with zero
propagation delay. The cost is the proxy-only features: Bucket Cache,
Retries and Custom rate limiting. Those are the features that make Helicone
"a gateway too".

**The same 20 ms costs 12x more at a 100 ms line than at a 320 ms one.** At
100 ms it pushes 6.2 points of requests over the line, because the
workload's body sits there (median 48.4 ms). At 320 ms, in the thin tail,
it pushes 0.5 points. What sets the cost is how many requests sit near the
line, not the overhead's share of the budget.

The curriculum already answers the 100 ms case. Lesson 19 says "For TTFT P99
< 100 ms SLA, Kong or Cloudflare" (3–8 and 1–3 ms of overhead), and lists
Portkey at 20–40 ms, so Helicone's 20 ms equals Portkey's floor. Helicone's
own page gives no millisecond figure, so the 20 ms comes from the lesson,
not the vendor.
