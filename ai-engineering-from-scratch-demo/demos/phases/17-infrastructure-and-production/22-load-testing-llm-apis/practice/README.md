<!-- generated:start -->
# 17-infrastructure-and-production / 22-load-testing-llm-apis

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/22-load-testing-llm-apis/) · upstream spec
`phases/17-infrastructure-and-production/22-load-testing-llm-apis/docs/en.md`

```bash
uv run demo practice run 22-load-testing-llm-apis --ex 1
uv run demo explain 22-load-testing-llm-apis --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/22-load-testing-llm-apis
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Compare uniform vs realistic distribution — where is the gap? | code | T0 | `ex01_the_gap_is_only_in_the_tail_and_prompt_length_and_concurrency_never_reach_the_latency.py` |
| 2 | Write the k6 script for a CI gate: TTFT P95 < 800 ms at 100 concurrent, runtime 5 minutes. | code | T0 | `ex02_the_gate_fails_realistic_traffic_on_cache_misses_and_40_iterations_pass_an_at_sla_server_40_percent_of_the_time.py` |
| 3 | Your soak test shows memory growing 50 MB/hour. Name three causes and the instrumentation to… | code | T0 | `ex03_all_three_leaks_draw_the_same_50_mb_line_and_the_lessons_own_workload_hides_the_cache_leak.py` |
| 4 | Spike test from 10 RPS to 100 RPS. What's the expected recovery time if Karpenter + vLLM prod… | code | T0 | `ex04_recovery_takes_about_three_minutes_so_a_two_minute_spike_ends_before_the_fleet_catches_up.py` |
| 5 | GenAI-Perf reports TPOT=6ms; LLMPerf reports TPOT=11ms on the same server. Explain. | code | T0 | `ex05_the_five_ms_gap_is_ttft_spread_over_the_output_so_it_needs_a_506_ms_ttft_at_100_tokens.py` |
<!-- generated:end -->

## Answers

All five are code. Exercises 1-3 run the lesson's own `simulate` and workload
generators. Exercise 4 drives lesson 03's autoscaling simulator and reads
lesson 18's module, and exercise 5 runs lesson 08's `RequestTrace`. External
sources were read on 2026-09-26: k6's metrics reference and release-notes
index, the xk6-sse README, LLMPerf's source (`token_benchmark_ray.py` and the
OpenAI chat client), and NVIDIA's NIM benchmarking metrics page. No k6 binary
was run; the script's threshold is evaluated in Python with k6's percentile
rule.

### 1 — the gap is only in the tail, and prompt length and concurrency never reach the latency

**The gap is in the tail and the mean, not the median.**

| 500 requests | uniform | realistic |
|---|---:|---:|
| TTFT P50 | 80 ms | 80 ms |
| TTFT P99 | 80 ms | 800 ms |
| mean | 81.44 ms | 193.76 ms |
| cache hits | 499 | 421 |

TTFT takes only two values, the hit and miss constants, so the whole gap is the
miss count: 1 against 79. The realistic generator draws from 80 prefixes and
meets 79 of them in 500 requests.

**Concurrency is a dead knob.** The printout repeats the same two rows at 10,
50 and 200. A request's cache insert is visible to the rest of its own batch,
and `unique_prefixes` is computed and never read. If concurrent requests
cannot see each other's inserts, the uniform workload misses once per request
in the first batch, and the gap closes as concurrency rises:

| concurrency | uniform mean | realistic mean |
|---:|---:|---:|
| 10 | 94.4 ms | 199.52 ms |
| 50 | 152 ms | 235.52 ms |
| 200 | 368 ms | 389.6 ms |

**Prompt length never reaches the latency.** `simulate` never reads
`prompt_tokens`. The uniform prompts are 2000 tokens against a realistic mean
of 503.9, four times the prefill, and they still report faster. Setting every
realistic prompt to 2000 tokens changes no number. The generator also uses
stddev 180 where the lesson's LLMPerf example uses 150.

**Nothing measures TPOT.** "Use It" says the code "measures effective TPOT".
`TPOT_MS` and `BATCH_EFFICIENCY_SHARED_PREFIX` are defined and never read.

### 2 — the gate fails realistic traffic on cache misses, and 40 iterations pass an at-SLA server 40% of the time

The script (`K6_SCRIPT` in the file) runs a `constant-vus` scenario, 100 VUs
for `5m`. TTFT is a custom `Trend` recorded at the first SSE event through
xk6-sse. The thresholds are `ttft: p(95)<800` with `abortOnFail`, and a
`server_errors` Rate of `status >= 500` under `rate<0.05`, for the lesson's
5xx line. (`http_req_failed` also counts 4xx.) A failed threshold makes k6
exit non-zero, which breaks the build. Run it as
`k6 run -e BASE_URL=... -e MODEL=... gate.js` beside a `prompts.json` sampled
from real traffic. TTFT needs its own metric because
k6's built-in `http_req_waiting` is time to first byte, and on an SSE stream
the first byte is the response headers. k6's metrics reference lists no
time-to-first-token metric.

**Against the lesson's simulator the gate grades the cache, not the server.**
The miss TTFT is 800 ms, exactly the threshold, so `p(95)<800` passes only
while fewer than about 5% of requests miss. Uniform prompts pass at P95 80 ms.
Realistic prompts fail at P95 800 ms, with 79 misses in 500.

**30-50 iterations cannot resolve a P95 gate.** The lesson's CI gate is
"30-50 iterations". With k6's interpolated percentile (`TrendSink.P` in k6's source), 40 samples stay under
the threshold only if at most one of them exceeds it:

| true fraction above 800 ms | chance the gate passes |
|---:|---:|
| 3% | 0.662 |
| 5% (exactly at SLA) | 0.399 |
| 10% | 0.080 |

The exercise's own run, 100 VUs for 5 minutes at the lesson's 15 ms TPOT and
256 max tokens, is about 7,400 iterations.

**"k6 v2026.1.0" is not a k6 version.** k6 moved to semantic versioning at
1.0 (May 2025), and its release-notes index lists 0.47-0.57, 1.0-1.8 and
2.0-2.3. The streaming support in this script comes from the xk6-sse
extension, not from a built-in metric.

### 3 — all three leaks draw the same 50 MB line, and the lesson's own workload hides the cache leak

The three causes are the three the lesson says a soak catches:

| cause | gauge that reads it | A/B soak that zeroes it |
|---|---|---|
| unbounded prefix/response cache | cache entry count + `tracemalloc` snapshot diff | replay a fixed prompt set |
| connection-pool drift | open connections / `process_open_fds` | enable keep-alive |
| observability overflow (label cardinality) | registry series count / `prometheus_tsdb_head_series` | drop the per-request label |

Each cause is calibrated so that, on its own, it produces the observed
50 MB/h at 10 RPS. Each A/B run takes its cause to 0 MB/h and leaves the other
two at 50.0.

**The memory graph cannot tell them apart, and neither can load.** All three
give 50.0 MB/h at baseline. Doubling RPS takes all three to 100.0 MB/h. The
one knob a soak test turns moves every candidate together.

**The lesson's realistic workload hides an unbounded cache.**
`make_realistic_workload` draws from 80 prefixes and has met all 80 by request
833, which is 83 s into a 10 RPS soak. After that it adds 0 entries an hour,
and the uniform workload's single prefix never adds any. A soak has to replay
real, mostly unique prompts to see this leak at all.

### 4 — recovery takes about three minutes, so a two-minute spike ends before the fleet catches up

The spike goes through lesson 03's simulator: Karpenter 50 s, model load 45 s,
a 15 s HPA tick, a 30 s queue timeout and queue-depth HPA. That simulator
serves one request per replica per tick, so both rates are divided by 150. The
ratio stays 10x, and the fleet needs 1 → 10 replicas. Recovery is the time
from the spike's start to the last request that was dropped or waited more
than one tick.

**About 3 minutes.** On a sustained 10-minute spike, queue-depth HPA drops 55
requests and recovers at +193.5 s. The first new replica cannot serve before
15 + 50 + 45 = 110 s; after that the backlog drains. Lesson 03's own "2-5
minutes" for a from-zero request brackets it. KAI's more aggressive rule
drops 48, and Cluster Autoscaler's 110 s provisioning drops 69.

**The lesson's 2-minute spike ends before the fleet recovers.** On a 120 s
spike, 55 of the 80 spike arrivals are dropped (69%), and waits settle at
+135 s, after the spike is over. The 10-minute spike drops the same 55, so the
provisioning chain sets the loss, not the spike's length. A 2-minute spike
test measures warm headroom, not autoscaling.

**Only headroom or a shorter chain moves the number.**

| 120 s spike | dropped |
|---|---:|
| Karpenter (shipped) | 55 |
| Cluster Autoscaler | 69 |
| 4 warm replicas | 29 |
| 10 warm replicas | 0 |
| zero model load | 28 |
| zero model load and provisioning | 2 |

**Production-stack contributes no term.** Lesson 18's module has
`make_workload`, `simulate` and KV-block constants, and no router, autoscaler
or replica count. KV offload and cache-aware routing change the work each
replica does, not how fast capacity arrives.

### 5 — the five ms gap is TTFT spread over the output, so it needs a 506 ms TTFT at 100 tokens

**LLMPerf folds the TTFT into its per-token average; GenAI-Perf does not.**
NVIDIA's metrics page gives ITL = (e2e − TTFT) / (N − 1). The page now names
the tool AIPerf. LLMPerf sums a `time_to_next_token` list whose first entry is
the TTFT, then divides by the output token count. Per request, the difference
is exactly

    LLMPerf − GenAI-Perf = (TTFT − GenAI-Perf TPOT) / N

So 6 against 11 ms means a TTFT of 5N + 6 ms: 506 ms at 100 output tokens,
1006 ms at 200. Lesson 08's `RequestTrace` with a 506 ms TTFT and 594 ms of
decode over 100 tokens reports 6.0 and 11.0. It is the same server; the 5 ms
is prefill and queueing, spread across the output.

**On lesson 08's own workload the gap is 1.03 ms, not 5.** Its 1000 synthetic
requests (mean TTFT 154 ms, 50-300 tokens) give GenAI-Perf 7.70 ms and LLMPerf
8.73 ms, and the identity reproduces the difference to 1e-9. A 5 ms gap needs
TTFTs about five times longer, or much shorter outputs, so the gap itself
points at long prompts or a queue.

**Lesson 08's GenAI-Perf divides N decode gaps by N − 1.** Its trace stores
`output_tokens` decode steps after the TTFT and divides their sum by N − 1. As
a result it reports 7.70 ms against a mean decode step of 7.64 ms, 0.71% high.

LLMPerf's divisor is not the server's token count either. LLMPerf re-tokenizes
the output text with `hf-internal-testing/llama-tokenizer`, so N depends on
that tokenizer rather than on the served model's. This was read in the source,
not measured.
