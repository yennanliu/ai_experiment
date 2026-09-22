<!-- generated:start -->
# 14-agent-engineering / 24-agent-observability-platforms

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/24-agent-observability-platforms/) · upstream spec
`phases/14-agent-engineering/24-agent-observability-platforms/docs/en.md`

```bash
uv run demo practice run 24-agent-observability-platforms --ex 1
uv run demo explain 24-agent-observability-platforms --ex 1
uv run pytest demos/phases/14-agent-engineering/24-agent-observability-platforms
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Export a week of OTel traces to Langfuse cloud (free tier). Which sessions failed? Why? | code | T0 | `ex01_trace_count_counts_spans_and_the_judge_runs_twice.py` |
| 2 | Write an LLM-judge rubric for your domain (factual correctness, tone, scope adherence). Test… | code | T0 | `ex02_the_shipped_judge_never_reads_a_word_of_the_output.py` |
| 3 | Compare Langfuse prompt versioning against Phoenix's trace clustering. Which tells you what b… | code | T0 | `ex03_versioning_needs_one_group_by_and_clustering_needs_a_cluster.py` |
| 4 | Read Opik's guardrail docs. Wire a PII redaction guardrail to one of your agent runs. | code | T0 | `ex04_redaction_runs_after_the_span_already_carried_the_value.py` |
| 5 | Benchmark the three on your corpus. Ignore vendor-published numbers; measure your own. | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — 40 of 140 sessions fail, and the summary is wrong about how many traces that is

A week of 140 sessions through the shipped collector splits cleanly: 40 FAIL at
0.20, 40 WARN at 0.50, 60 PASS at 1.00. The FAIL sessions are the ones whose
`chat` span errored — 0.4 for the error plus 0.3 for the missing final answer —
and every one carries an `error.reason`: `rate_limited` 14, `tool_timeout` 13,
`context_overflow` 13. The WARN sessions are the long-context ones, penalised
0.3 for no reference id, 0.1 for no tool call and 0.1 for crossing 2000 tokens.

Three things about the summary are worth not trusting.

`trace_count` is a span count. The field is named for traces and holds
`len(spans)`, so the healthiest session reports 4 for a single `trace_id`. Across
the week that is 400 spans under 140 traces, and any per-trace rate computed from
this field is off by the 2.86x average fan-out. This is the kind of mistake that
survives for a long time, because the number looks plausible and moves in the
right direction.

The judge runs twice per session. `summarize` calls `scripted_llm_judge` and
throws away the verdict, then `main` calls it again to get the verdict back — two
call sites for one decision, 280 judge calls for 140 sessions. With the scripted
judge that is free. With a real LLM judge it is double the eval bill, and the
lesson's own pitfall list opens with "tracing without evaluation is just
expensive logging".

And a session is scored as a unit. The judge takes the whole span list and
returns one number, so a 2-span session that failed immediately and a 12-span
session that retried ten times before failing both score 0.20. Sorting by score
puts them adjacent, and the ten extra spans of wasted work — the thing you would
actually want to fix first — are invisible in the ranking.

### 2 — the rubric reads the text, which is the one thing the shipped judge never does

`scripted_llm_judge` reads `status`, `name`, and two attribute keys. Factual
correctness, tone and scope adherence are all properties of the output text, and
the judge never opens it. So the rubric is written against the answer, given a
third of the weight each, and run on 50 traces with hand labels.

The rubric agrees with the labels 40/50 at precision 0.67 and recall 0.67; the
shipped judge manages 15/50, and those 15 are just the traces that happened to be
good, because it passes all 50. They disagree on 35. That is the headline, and it
is worth stating the rubric's own 10 misses too, because they are the honest
limit of a keyword stand-in: it reads `"the refund window is not 30 days"` as
factually correct, since substring matching cannot see negation, and marks
`"that's wrong -- the warranty is actually 2 years"` as rude, since it cannot see
that the rudeness is quoted. Those are exactly the cases an LLM judge is for.

Two structural findings fall out. A structurally perfect trace can carry any text
at all: a session with a tool call, a reference id and no errors scores 1.00
whatever the referenced content says, because `has_final` tests that
`gen_ai.output.reference_id` *exists* and never dereferences it. The judge is
measuring that instrumentation ran. And the shipped score cannot reach the bottom
of its own range — the four penalties sum to 0.9, so `max(0.0, score)` is
unreachable and nothing scores below 0.10, while the rubric uses 0.00 to 1.00 on
the same data.

Finally, the reason to report criteria separately rather than blended: 10 traces
are factually wrong while in scope and in tone, and 5 state the right fact about
the wrong subject. Each scores 2/3, so all 15 clear a 0.60 threshold on the mean.
A single `eval_score_mean` hides every specific failure the rubric was written to
catch.

### 3 — each finds its own kind of break, and "faster" depends which you got

Read "faster" as *how many traces arrive before the cause is named* — the number
an on-call engineer waits through — with "named" meaning the verdict holds for
the rest of the stream rather than flickering once. On that definition, over 400
traces:

On a regression caused by a prompt edit, grouping by `prompt.version` splits the
failure rate 10.0% against 68.5% and names `v7` at trace **8**, as soon as both
groups hold four traces. On a regression caused by a changed retrieval corpus —
which carries no version difference at all — grouping by behaviour names the
affected topic at trace **42**, 5.2x later.

But the comparison that matters is not the ratio, it is the blindness. A corpus
change leaves every trace on `v7`, so the version group-by sees one group at
14.2% and returns nothing, ever. A prompt change hits every topic equally, so the
behavioural split never clears the threshold and returns nothing, ever. Versioning
finds one of the two regressions; clustering finds the other. That is why the
lesson's "picking one" table lists them for different needs instead of ranking
them, and why Phoenix is positioned as a drift tool "alongside broader
platforms".

The generalisation: versioning is fast because its candidate list is enumerated —
two versions — so it needs only enough data to fill the groups, and it is silent
rather than wrong when the cause is not on the list. Clustering has to discover
the partition, which costs data and works on causes nobody labelled. Speed and
coverage trade against each other, and the practical answer to "which tells you
faster" is "the one you already instrumented for".

Which makes the last finding the important one: neither works on the shipped
span. `SpanEvent` has five fields and the lesson's own `main()` writes zero
attributes naming a prompt or a version, so the Langfuse column does not exist
and the bisect is impossible. That is the third pitfall — "prompt versions not
tied to traces" — sitting in the emitter, where it is cheap to fix and where
nobody looks.

### 4 — the guardrail has a placement question before it has a regex question

`TraceCollector.ingest` appends whatever it is handed, so a redaction guardrail
can sit in front of it or behind it, and only one of those removes anything.
Wrapping `ingest` redacts four PII kinds across 24 spans in 8 sessions, rewriting
5 values and leaving 0 matches in the collector where the unguarded run holds 5.

Redacting *after* ingest reaches the same 0 matches — and is not redaction. All
24 spans were appended with their original values first, so anything that read
the list, forwarded it to an exporter or persisted it in between saw the PII. The
final state is identical and the exposure is total. That is why Opik places
guardrails in the call path rather than in the dashboard: a guardrail is a
transform on the way in, and "the stored copy is clean" is not the property you
need.

The regexes then have their own problems, and the interesting one is not a miss.
Of the six PII values in the fixture, four are fully redacted and one — an
internationally formatted phone number — is missed outright. The sixth comes out
as `ada+[REDACTED:email]`: a *partial* redaction that reads as a complete one.
That is worse than a miss, because a miss looks like PII and gets caught by the
next reviewer, while a partial redaction looks handled. And the guardrail's own
report of "0 remaining matches" is true of its own regexes and false of the data.
A redactor measured against itself always passes; it has to be measured against
an independently labelled corpus, which is the grounding argument the lesson
makes about LLM judges, in a different register.

One quiet consequence worth naming. Redaction touches neither `status` nor
`gen_ai.output.reference_id`, so `scripted_llm_judge` returns PASS for all 8
sessions before and after — zero verdicts changed. A judge that reads structure
is unaffected by redaction; a judge that reads content is blinded by it. If you
adopt Lesson 23's references-only capture, the guardrail runs over the external
store and the spans never had the content to begin with, and the two problems
stop competing.

### 5 — benchmark on your corpus, and the number that matters is not ingest throughput

**Comet Opik (Apache 2.0)** publishes the comparison this exercise is warning
about: Comet's own measurement puts Opik at 23.44s for logging plus evals against
Langfuse's 327.15s, roughly a 14x gap, and the lesson itself says to take vendor
benchmarks as directional. Which is the right instruction, and the reason is
structural rather than suspicious: a vendor benchmark measures the workload the
vendor chose, and ingest throughput is almost never the workload that binds.

**What to actually measure, and why.** Benchmarking three observability platforms
on your own corpus means fixing the corpus and varying the platform, and the
corpus has to be *yours* — a week of real traces with your span fan-out, your
attribute cardinality and your content sizes. Exercise 1 gives the shape of the
thing being measured: 400 spans under 140 traces, a 2.86x fan-out. A platform
priced or benchmarked per span behaves very differently on a 2.86x fan-out than
on a 10x one, and a vendor's demo corpus has neither.

Four numbers are worth taking, in descending order of how often they decide the
choice:

*Time to answer a question you actually ask.* Not ingest latency — time from "the
error rate rose" to "it was `v7`". Exercise 3 measures exactly this and finds the
answer depends on whether the platform has the column: 8 traces with prompt
versioning, 42 with behavioural clustering, and never for the wrong pairing. This
is the number that makes a platform worth its price, and no vendor publishes it,
because it depends on your instrumentation and not on their code.

*Eval cost per session at your judge's price.* Exercise 1 finds the shipped
pattern calling the judge twice per session — 280 calls for 140 sessions. At a
real judge's per-call price that doubling dominates every other cost in this
lesson, and it is a property of how the platform's SDK is wired, not of how fast
it ingests. Measure judge calls per session, not judge latency.

*Attribute cardinality limits.* Exercise 3's whole comparison rests on grouping
by `prompt.version`. Platforms differ in how many distinct values they will index
and how far back. A platform that drops high-cardinality attributes after 7 days
cannot bisect a regression that took two weeks to notice.

*Ingest throughput.* Last, because it is the one the vendors publish and the one
that is easiest to fix with a queue. Measure it, then check whether it is ever
the constraint. On the corpus in exercise 1 it is not.

**The trap in the vendor number specifically.** "Logs + evals in 23.44s vs
327.15s" packages two very different operations in one figure, and the eval half
depends on the judge model, the judge prompt and how many times the SDK calls it
— all of which are yours, not the vendor's. Reproducing that gap on your corpus
means holding the judge fixed across all three platforms, which the published
benchmark almost certainly did not do.

**What the licences do to the benchmark.** Langfuse is MIT and Opik is Apache
2.0; Phoenix is Elastic License 2.0, which constrains offering it as a service
and therefore constrains what "self-hosted at our scale" means in a benchmark.
The lesson's table already flags this as its own selection row, and it belongs in
the benchmark design too: a number produced against a managed tier is not a
number about the self-hosted one.

**The conclusion the exercise is pushing toward.** Per the lesson's industry
data, 89% of organisations already have agent observability and 32% cite quality
issues as the top production barrier — so the binding constraint is evaluation
quality, not ingest speed. Exercise 2 is the evidence: a judge that never reads
the output passes 50 of 50 traces including 25 that are wrong. Benchmark the
three on whether their eval tooling would have caught those, and the throughput
number stops mattering.
