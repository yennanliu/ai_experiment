<!-- generated:start -->
# 17-infrastructure-and-production / 15-batch-apis

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/15-batch-apis/) · upstream spec
`phases/17-infrastructure-and-production/15-batch-apis/docs/en.md`

```bash
uv run demo practice run 15-batch-apis --ex 1
uv run demo explain 15-batch-apis --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/15-batch-apis
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. For a 100k-doc pipeline with 3K-token system prompt and 500-token output,… | code | T0 | `ex01_the_full_stack_saves_68_percent_and_500_output_tokens_keep_it_above_25_percent_of_sync.py` |
| 2 | Pick three features in a real product you know. Triage each into interactive/semi/batch. | code | T0 | `ex02_the_lanes_split_cleanly_until_next_hour_which_the_lesson_calls_batch_and_its_sla_cannot_meet.py` |
| 3 | A user complains their report took 3 hours. Was that a batch mis-triage or a legitimate inter… | code | T0 | `ex03_three_hours_is_a_normal_batch_and_a_normal_sync_run_so_only_the_promise_decides.py` |
| 4 | Your batch API return SLA is 24h but P99 is 20 hours. How do you communicate this to the user… | code | T0 | `ex04_promise_24_hours_and_spill_to_sync_at_hour_20_which_costs_1_percent_more.py` |
| 5 | Compute break-even: at what shared-prefix length does batch + cache become cheaper than runni… | code | T0 | `ex05_no_prefix_length_makes_batch_plus_cache_cheaper_because_its_bill_rises_with_the_prefix.py` |
<!-- generated:end -->

## Answers

### 1 — the full stack saves 68%, and 500 output tokens keep it above 25% of sync

**Batch + cache takes $2,250.00 to $720.01, a saving of $1,529.99 (68.0%).**
The exercise does not fix the per-document input, so it is swept. The
headline uses the reference's summarization value of 2000 tokens:

| per-doc tokens | batch + cache / sync |
|---:|---:|
| 0 | 25.45% |
| 300 | 26.72% |
| 2000 | 32.0% |
| 15000 | 43.41% |

**The ~10% the lesson promises is out of reach here.** Output is halved but
never cached, so with no document tokens the ratio is
0.5 × (0.3P + 15O) / (3P + 15O). That reaches 10% only at a prefix 40 times
the output, 20,000 tokens for this pipeline. The reference's own three runs
print 24.3%, 17.1% and 41.3% under a "~10%" banner. The Problem's
"$2,000 → $180, ~9%" pipeline prints $1,050.00 → $255.01 (24.3%), and 14.0%
even with no document tokens.

**The reference assumes every cache read hits.** Anthropic's batch docs say
cache hits in a batch are best-effort, typically 30% to 98%, and suggest the
1-hour cache because batches can outlast 5 minutes. Price a miss as a fresh
5-minute write, and the stack is 32.46% of sync at a 98% hit rate, 43.5% at
50% and 48.1% at 30%. Plain batch is 50%. Below a 21.7% hit rate,
`cache_control` costs more than not caching.

### 2 — the lanes split cleanly until "next hour", which the lesson calls batch and its SLA cannot meet

The product is GitHub, and each feature is reduced to how long its user will
wait:

| feature | wait | lane |
|---|---|---|
| Copilot inline completion | < 1 s | interactive |
| Copilot code review on a pull request | ~15 min | semi-interactive |
| weekly repository digest email | days | batch |

The lesson states its lane rules twice. The Concept text says "minutes" is
semi and "next hour" or "by morning" is batch. The skill refuses batch under
a 60 s P99, and batch promises only 24 hours. Both rule sets agree on these
three features.

**They disagree on every tolerance from 1 to 24 hours.** Labelling newly
opened issues "within the hour" is batch by the Concept text and semi by the
SLA. At the lesson's typical P50 of 2-6 hours, most such batches would miss
the hour.

**Once caching is on, the lane is worth exactly 2x.** `cost_batch_cache` is
`cost_sync_cache * 0.5`, so on every workload the move from cached
interactive to batch saves 50%. The skill's "~90% leaked spend" needs a sync
run that was also uncached, and even then the reference workloads leak 75.7%,
82.9% and 58.7%. The semi lane has no price of its own: none of main.py's
four configurations is an async queue.

### 3 — three hours is a normal batch and a normal sync run, so only the promise decides

**Criterion: it is a batch mis-triage exactly when the report ran on batch
and the user expected it in under 24 hours.** In full:

- **Batch, expected in under 24 h:** mis-triage. Batch promises 24 hours, so
  move the report to semi or sync.
- **Batch, expected in 24 h or more:** a legitimate batch. The complaint is
  about communication, so tell the user the 24-hour promise.
- **Sync, report too big for the wait:** the size alone takes longer than
  the user will wait, at the lesson's 12,500 documents an hour. It cannot be
  interactive at all. Precompute it on batch and serve the stored result,
  which is the partial-interactivity trap.
- **Sync, otherwise:** an incident, not a triage question.

Over 12 enumerated cases, 4 are mis-triage, and all 4 are on batch.

**The 3 hours itself decides nothing.** It is inside the lesson's 2-6 hour
batch P50, and it is what the lesson's own sync pipeline takes for 37,500
documents. The lesson's typical figures also disagree with each other: it
says "2-8 hours in practice" for OpenAI and a "Typical P50 is 2-6 hours".
Anthropic's docs say most batches finish in under an hour. That is why the
criterion compares the expectation with the 24-hour promise, not with a
typical latency.

### 4 — promise 24 hours and spill to sync at hour 20, which costs 1% more

`code/main.py` has no notion of time, so completion time is modelled as a
lognormal with P50 4 h (the lesson's 2-6) and the exercise's P99 20 h. The
pipeline is the Problem's: 50k documents, which the lesson says takes 4 hours
on sync.

**Tell the user "within 24 hours". Behind the scenes, any batch still
running at hour 20 is cancelled and rerun on sync + cache.** Hour 20 is the
P99, so 1% of nights spill. The expected night costs $257.56 against
$255.01, and every report arrives by hour 24. Both providers' docs say
cancelled or expired requests are not billed, which is why a spilled night is
priced as sync alone. The model treats a night as all-or-nothing. Without the
spillover, 0.48% of batches pass 24 hours, expire, and the report is a day
late.

**"By morning" is the expensive promise.** An 8-hour promise means spilling
at hour 4, the P50. Half the nights spill, and the expected night is $382.51,
1.5x batch. Without spillover, 15.8% of nights would miss.

**The skill's observable fires on this healthy provider.** The skill says to
alert when completion P95 exceeds 12 hours. With a 20-hour P99, the P95 is
10.19 h, 12.48 h and 14.06 h at a P50 of 2, 4 and 6 hours, so the alert
fires for any P50 above 3.5 hours.

### 5 — no prefix length makes batch + cache cheaper, because its bill rises with the prefix

The workload is exercise 1's: 100k documents, 2000 per-document and 500
output tokens. The GPU is one H100 reserved at $2.50/h, lesson 07's price, so
$60 a day. It runs 2,300 tok/s, from lesson 04's 2,200-2,400, and uses prefix
caching, so its cost is independent of the prefix. That sets a 70B open
model against the reference's $3/$15 API prices, as the exercise implies.

| prefix | batch + cache | vs sync |
|---:|---:|---:|
| 0 | $675.00 | 50.0% |
| 3,000 | $720.01 | 32.0% |
| 30,000 | $1,125.05 | 10.87% |

**The bill rises $0.015 a night per prefix token, and a zero-length prefix
already costs 11.25 GPU-days.** So there is no prefix length at which batch
+ cache becomes the cheaper option. The break-even is in volume: at a 3K
prefix, batch + cache is cheaper only below 8,333 documents a night.

**The break-even is a ceiling, and it exists only for small jobs.** At 1,000
documents a night, batch + cache stays cheaper up to a 350,960-token prefix.
A longer prefix improves only the ratio to sync, 50% down to 10.87%, which
is the number the lesson quotes. That ratio says nothing about beating a
fixed-cost GPU.

**Capacity is not the constraint.** 2,300 tok/s for 8 hours is 66.24M output
tokens, or 132,480 documents of 500 tokens. That counts decode only, so it
is an upper bound.
