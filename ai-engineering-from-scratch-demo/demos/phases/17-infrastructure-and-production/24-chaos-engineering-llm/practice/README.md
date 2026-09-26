<!-- generated:start -->
# 17-infrastructure-and-production / 24-chaos-engineering-llm

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/24-chaos-engineering-llm/) · upstream spec
`phases/17-infrastructure-and-production/24-chaos-engineering-llm/docs/en.md`

```bash
uv run demo practice run 24-chaos-engineering-llm --ex 1
uv run demo explain 24-chaos-engineering-llm --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/24-chaos-engineering-llm
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Which experiment trips the burn-rate gate and why? | code | T0 | `ex01_only_the_provider_429_trips_because_the_gate_is_blast_over_20_percent_and_the_80x_prompt_runs.py` |
| 2 | Design the first five chaos experiments for a vLLM-based RAG service. Include success criteria. | code | T0 | `ex02_criteria_on_error_latency_and_grounding_fail_all_five_broken_runs_and_the_lessons_gate_stops_one.py` |
| 3 | Your burn-rate alert paused an experiment. How do you determine root cause — chaos or natural? | code | T0 | `ex03_compare_tagged_traffic_with_the_untagged_control_because_the_gate_aborts_a_natural_incident_too.py` |
| 4 | Argue whether chaos should run in production or only staging. When is production the right an… | code | T0 | `ex04_production_once_the_prerequisites_hold_because_the_lessons_experiments_cost_0_08_percent_of_a_month.py` |
| 5 | Name three LLM-specific failure modes that generic network-chaos cannot reproduce. | code | T0 | `ex05_tokenizer_cost_kv_pressure_and_a_truncating_failover_follow_the_payload_and_the_lesson_names_two.py` |
<!-- generated:end -->

## Answers

### 1 — only the provider 429 trips, because the gate is blast over 20 percent and the 80x prompt runs

**Only "provider 429 fallback" trips, at 30x burn and 30% blast.** The gate is
`burn_rate > 2.0 and blast_radius_pct > 0.2`:

| experiment | burn (vs expected) | blast | service-wide burn | status |
|---|---:|---:|---:|---|
| pod kill (1 decode replica) | 4x | 5% | 1.15x | COMPLETED |
| provider 429 fallback | 30x | 30% | 9.7x | ABORTED |
| malformed prompt tokenizer stall | 80x | 10% | 8.9x | COMPLETED |

All three pass the burn half of the gate, so the blast half decides. At 20%
blast none of them trips; at 21% all three do.

**The experiment that burns fastest runs to completion.** The malformed prompt
takes the whole service to 8.9x, close to the aborted run's 9.7x. It completes
because of the blast condition, and the lesson never states that condition.
Its rule, in Guardrails, Key Terms and Numbers, is burn over 2x on its own,
which would pause all three. "20%" appears nowhere in docs/en.md. The lesson's
skill file caps blast at "< 30% of fleet", and the aborted experiment's 30%
already breaks that.

**The burn rate is measured against the baseline, not the budget, and ignores
duration.** `ERROR_BUDGET_PER_DAY` is printed and never read. Measured against
the 0.1% budget, burn is 2.0x, 15x and 40x, so the pod kill sits exactly at 2
and does not exceed it. `duration_min` is only echoed: a 500-minute run gets
the same verdicts.

**Read as a daily budget, nothing trips.** The provider run spends 1.56% of
one day's budget, which puts the day at 1.03x the expected burn.

### 2 — criteria on error, latency and grounding fail all five broken runs, and the lesson's gate stops one

Five experiments, smallest blast radius first. Each has criteria on error
rate, p95 against baseline and grounded-answer rate. The expected outcomes
below are the design's hypotheses.

| # | experiment | blast | criteria (err / p95x / grounded) | naive service fails |
|---|---|---:|---|---|
| 1 | malformed prompt in 1% of requests | 5% | ≤1.5% / ≤1.2 / ≥0.95 | p95 (12 s tokenizer stalls) |
| 2 | gateway-to-vLLM partition | 5% | ≤0.5% / ≤1.5 / ≥0.90 | error, grounded (no failover) |
| 3 | vector DB +2 s, 500 ms timeout | 10% | ≤0.5% / ≤1.5 / ≥0.85 | grounded (answers with no context) |
| 4 | long-context burst, top_k 5 → 40 | 10% | ≤5% / ≤2.0 / ≥0.90 | p95 (preemption + re-prefill) |
| 5 | kill 1 of 4 decode replicas | 25% | ≤0.1% / ≤1.5 / ≥0.95 | error (no retry, 30 s to drop the endpoint) |

The baseline and the hardened service (input limit, failover, BM25 fallback,
admission control, gateway retry) fail none of the criteria. The naive service
fails every experiment, each on the SLI that experiment targets.

**The lesson's gate stops one of the five broken runs.** Given the naive error
rates, `run_experiment` aborts only the replica kill. The partition fails
every request in its blast radius, 2000x burn, and still completes because its
blast is 5%. The vector-DB run, grounding 0, and the KV run, p95 at 3.5x, both
read 1.0x and complete. **The reference `Experiment` has no field for a
criterion.** Its error rate is an input, so the outcome is set before the run.

### 3 — compare tagged traffic with the untagged control, because the gate aborts a natural incident too

The simulation is the lesson's aborted experiment: provider 429, 30% blast,
1.5% induced errors. It runs over a seeded 60,000-request window in three
worlds. Split the window by trace-ID tag. Test tagged against untagged traffic
for chaos, and the untagged control against baseline for a natural incident.

| world | z chaos | z natural | verdict | tagged share of errors |
|---|---:|---:|---|---:|
| chaos only | 23.77 | 0.23 | chaos | 0.9272 |
| natural only (+1% everywhere) | 0.29 | 94.82 | natural | 0.3064 |
| both | 13.21 | 94.82 | chaos + natural | 0.5005 |

**The lesson's gate aborts a natural incident as if it were the experiment.**
With no chaos at all, the tagged traffic shows 1.11% errors. Given that rate,
`run_experiment` reads 22x at 30% blast and aborts, with no field to say why.
**Counting tagged errors does not answer the question.** In the "both" world
the tagged share is a coin-flip, 50.05%. The natural errors inside the blast
radius are also the ones a suppression window silences, so the untagged
control is where the evidence survives. **The smallest experiment needs most
of its window to be attributed.** At z ≥ 3 the pod kill (0.2% against 0.05%,
5% blast) needs 48,394 requests. That is 4.0 of its 5 minutes at 200 req/s,
and longer than the whole run below 162 req/s.

### 4 — production once the prerequisites hold, because the lesson's experiments cost 0.08 percent of a month

**Production, once the five prerequisites hold.** At the lesson's 99.9% SLO and
0.05% expected error rate, the baseline leaves half the budget. The three
experiments together spend 0.081% of a 30-day budget. To spend the half that
is left, the malformed prompt would have to run 5,400 minutes and the provider
429 4,800. Staging cannot supply the real provider's 429s, the real prompt mix
behind a tokenizer stall, or production concurrency behind a KV storm.

**The worst case sets the blast cap and the abort time, not the plan.** If the
blast radius fails outright, the remaining budget lasts 432 minutes at 5%
blast and 72 minutes at 30%. A 5-minute abort then risks 1.2% or 6.9% of it.
The decision rule sends a team to staging in any of these cases:

- a prerequisite is missing
- there is no green staging run yet
- the worst case exceeds 10% of the budget left

It sends a team with more than 2 incidents a week to stabilise first. Where
the abort is a person paged, 15 minutes, the 5% experiment runs in production
and the 30% one goes to staging. **Nothing in the lesson's code knows where an
experiment runs.** `Experiment` has no environment field, and the skill's
refusal rules exist only as prose.

### 5 — tokenizer cost, KV pressure and a truncating failover follow the payload, and the lesson names two

Generic network chaos acts on packets and never reads their content. Each of
these three failure modes is triggered by content or by the model:

1. **Tokenizer stall.** A toy BPE that merges one pair per scan costs 2,031
   pair-steps on 1 KB of prose and 515,775 on 1 KB with no whitespace, 254x.
   Delayed, 1%-corrupted or truncated prose costs at most 2,094.
2. **KV preemption whose cost scales with context.** On a 4,096-block pool, 32
   requests of 1,024 tokens preempt 0 times, whether delayed or with 10%
   dropped. At 8,000 tokens they preempt 5 times and re-prefill 40,704 tokens,
   8,141 per preemption.
3. **Silent truncation on failover.** A 16K → 4K fallback drops 8 of 10
   retrieved chunks and still returns 200. The lesson's gate reads 0x and
   completes.

**Network chaos reaches KV preemption only by multiplying requests, and a
different guard stops that.** Duplicated 1,024-token requests preempt 21 times.
A cap of 32 running requests takes that to 0 but leaves the 8,000-token run at
5, which needs token-budget admission. **The lesson's five "LLM-specific"
experiments contain two LLM-specific mechanisms.** Network failure and
provider outage are generic network and HTTP faults. Memory overload and KV
eviction storm are both KV preemption. `code/main.py` runs none of them: its
malformed prompt is a fixed 0.04 error rate.
