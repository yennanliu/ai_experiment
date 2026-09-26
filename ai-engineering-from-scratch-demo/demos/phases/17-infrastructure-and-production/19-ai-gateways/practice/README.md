<!-- generated:start -->
# 17-infrastructure-and-production / 19-ai-gateways

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/19-ai-gateways/) · upstream spec
`phases/17-infrastructure-and-production/19-ai-gateways/docs/en.md`

```bash
uv run demo practice run 19-ai-gateways --ex 1
uv run demo explain 19-ai-gateways --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/19-ai-gateways
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Configure fallback from OpenAI→Anthropic→self-hosted. What's the expected… | code | T0 | `ex01_at_five_percent_the_chain_serves_99_9875_percent_and_the_gateway_changes_nothing_but_a_constant.py` |
| 2 | Your SLA is TTFT P99 < 200 ms on a 300 ms baseline. Which gateways stay within budget? | code | T0 | `ex02_no_gateway_fits_because_the_baseline_alone_is_100_ms_over_and_the_tail_is_the_fallback.py` |
| 3 | A healthcare customer requires self-hosted + PII redaction + audit. Pick Portkey OSS or Kong. | explain | T0 | prose, below |
| 4 | Compare LiteLLM vs Kong: at what RPS ceiling should a team migrate? | explain | T0 | prose, below |
| 5 | Design a rate-limit policy for a multi-tenant SaaS: free tier, trial tier, paid tier. Token-b… | code | T0 | `ex05_sliding_window_because_a_token_bucket_cannot_grant_the_burst_and_the_full_rate_under_one_cap.py` |
<!-- generated:end -->

## Answers

Sources for exercises 3 and 4 were fetched on 2026-09-26: the Kong benchmark
post, the Portkey gateway repository (README and LICENSE), Kong's AI PII
Sanitizer and audit-log docs, and Maxim's Bifrost benchmark page. The lesson's
`code/main.py` is a fallback simulator; exercises 1, 2 and 5 run against it.

### 1 — at 5% the chain serves 99.9875%, and the gateway changes nothing but a constant

The shipped `PROVIDERS` list already is OpenAI → Anthropic → self-hosted, so
"configure" means setting each error rate to 0.05. "Hit rate" has two
readings, and both are computed:

| | exact | simulated, 200,000 requests | simulated, n = 1000 |
|---|---:|---:|---:|
| served | 0.999875 | 0.9999 | 100.0% |
| fallback calls per request | 0.0525 | 0.0522 | 0.060 |

A request fails only when all three providers fail, so the exact failure rate
is 0.05³. 5% of requests hit a fallback and 0.25% reach self-hosted. The
simulator's own 1000-request run expects 0.125 total failures, so it cannot
show the number the exercise asks for.

**The four gateways differ only by a constant.** On the shipped run every
gateway reports 100.0%, 34 retries and 34 fallbacks, because each replays
seed 7. Mean latency is 183.048 ms plus a fixed overhead of 10, 30, 5 or 2 ms.
The gateway does not change who is called, how often, or what fails.

**Every "retry" is a fallback, and there is no backoff.** No provider is ever
called twice and nothing waits, so the lesson's "exponential backoff, bounded
attempts" has no code. The two counters differ only by the requests that
failed every provider: 10,454 − 10,434 = 20 = 200,000 × (1 − 0.9999).

Two smaller things. A failed call costs 0.3 of the provider's latency (54 ms
for OpenAI), although the comment says "half-done". And "Six core features"
numbers seven items.

### 2 — no gateway fits, because the baseline alone is 100 ms over, and the tail is the fallback

**None.** A 300 ms provider P99 under a 200 ms P99 SLA leaves a budget of
−100 ms before any gateway is added. Read the other way, as 200 ms of headroom
on top of 300 ms, which is the lesson's "For P99 < 500 ms, any", every gateway
fits: the worst is Portkey at 300 + 40 = 340 ms. The gateway choice decides
nothing under either reading. (The P99 overheads are the top of each range in
"Latency budget": 15, 40, 8, 3 ms.)

**On the lesson's simulator the tail comes from the fallback.** Its path
latencies are fixed:

| path | latency |
|---|---:|
| OpenAI succeeds | 180 ms |
| OpenAI fails, Anthropic succeeds | 54 + 220 = 274 ms |
| both fail, self-hosted succeeds | 54 + 66 + 100 = 220 ms |

3% of requests fall back, which is more than 1%, so the P99 is 274 ms with no
gateway. With a gateway it is 284 ms (LiteLLM), 304 ms (Portkey), 279 ms (Kong)
and 276 ms (Cloudflare). The fallback adds 94 ms to the tail and the gateway
adds 2 to 30 ms. The code reports only the mean, 185 to 213 ms, which hides
this.

**The lesson's "P99 < 100 ms: Kong or Cloudflare" rule assumes a baseline of
85 to 91 ms.** Only then do Kong and Cloudflare fit while LiteLLM and Portkey
miss. The simulator's fastest provider is 100 ms before any gateway.

### 3 — Portkey, but neither open-source build ships PII redaction

*Draws on "Portkey — control plane positioning".*

**Pick Portkey,** with its self-hosted enterprise deployment rather than the
open-source gateway as it ships today. The reasoning:

- **The lesson's framing favours Portkey.** It positions Portkey as the
  guardrails, PII redaction and audit-trail gateway, and Kong as the
  high-throughput choice. A healthcare tenant is a compliance problem long
  before it is a 1000 RPS problem.
- **But "Portkey OSS" does not include PII redaction today.** The
  `Portkey-AI/gateway` main branch carries an **MIT** licence dated 2024, not
  the "Apache 2.0 as of March 2026" the lesson states. Its README lists "PII
  Redaction" only under "Gateway Enterprise Version", next to "SOC2, ISO,
  HIPAA, GDPR Compliances". The open-source gateway does ship guardrails ("40+
  pre-built guardrails"). A banner says the "core enterprise gateway is
  merging into open-source with our 2.0 release", which is a pre-release
  branch. I did not verify which features that branch contains, or when it
  becomes a release.
- **Kong fails the same test.** Its AI PII Sanitizer plugin is "only
  available as part of our AI Gateway Enterprise offering" (Gateway 3.10+).
  It runs as a self-hosted anonymizer container, which suits self-hosting.
  Kong's audit logging records Admin API requests and database changes, not
  proxied LLM traffic, so a per-call audit trail would have to come from its
  logging plugins.

So neither free build meets all three requirements. Both vendors meet them
with a paid, self-hostable enterprise tier. Portkey bundles PII redaction and
lists HIPAA in that tier. Kong is the better choice only if the customer
already runs Kong, or will exceed Portkey's throughput.

### 4 — plan the move at 500 RPS, because 2000 is where it breaks, not where to start

*Draws on "LiteLLM — MIT OSS, Python".*

**Start the migration when sustained peak traffic passes about 500 RPS, and
finish it before 1000 RPS.** The lesson gives three different numbers. LiteLLM
is the "best fit" below 500 RPS, Kong's "best fit" is above 1000 RPS, and
LiteLLM "breaks down around 2000 RPS". A breaking point is not a migration
threshold. A gateway move is a staged rollout (the lesson's skill file asks
for a 1% canary), so it has to begin while there is still headroom.
Migrating at 500 RPS leaves 4x headroom below the break. Waiting until
1000 RPS leaves 2x, which one traffic spike can use up.

The numbers the lesson cites do not all come from where it says:

- **The Kong benchmark does not report a 2000 RPS break or 8 GB of memory.**
  The post (July 2025) measured throughput under a 12-CPU cap, with k6 at 400
  VUs against a WireMock OpenAI mock. It reports that Kong reached "28,000+"
  RPS and was "228% faster than Portkey and 859% faster than LiteLLM", with
  "86% lower latency than LiteLLM". It gives no per-gateway RPS for LiteLLM
  (1.63.7) or Portkey. If "859% faster" means 9.59x throughput, LiteLLM
  reached roughly 2,900 RPS on 12 CPUs. That is my inference from the
  headline numbers, not a figure in the post.
- **The "~2000 RPS, 8 GB, cascading failures" claim comes from third-party
  write-ups.** It appears in DEV Community posts comparing LiteLLM with
  Bifrost. Maxim's own Bifrost benchmark page tests LiteLLM at 500 VUs on a
  2-vCPU t3.medium, where it reports a P99 of 90.72 s, 88.78% success and
  372 MB of memory. It never tests 2000 RPS.

The ceiling scales with hardware and worker count, so the numbers only apply
to the hardware they were measured on. The one published measurement of
LiteLLM failing (on a 2-vCPU box) is already at 500. Measure your own P99 at
2x your current peak before you trust any of these figures.

### 5 — sliding window, because a token bucket cannot grant the burst and the full rate under one cap

The design gives free, trial and paid tenants quotas of 20, 100 and 1000
requests per minute. The lesson's code has no rate limiter (no name in
`code/main.py` matches limit, bucket, window or throttle), so four were built.
Each one faced a tenant that sits idle for 30 s and then floods at 4x its
quota for 90 s.

| algorithm | worst 60 s admitted (20 / 100 / 1000) | total admitted in the flood |
|---|---:|---:|
| sliding-window log | **20 / 100 / 1000** | 40 / 200 / 2000 |
| token bucket, capacity = quota | 39 / 199 / 1999 | 49 / 249 / 2499 |
| token bucket, half size | 19 / 99 / 999 | 24 / 124 / 1249 |
| fixed window | 40 / 200 / 2000 | 40 / 200 / 2000 |
| sliding-window counter | 30 / 150 / 1500 | 40 / 200 / 2000 |

**Policy: a sliding-window log per tenant, with the tier setting the quota.**
It is the only limiter that holds the quota in every 60 s span. It still lets
a tenant spend a whole minute's quota at once, and sustain the full rate.

**A token bucket can have the burst or the full rate, not both.** Its worst
minute is capacity plus 60 s of refill. Sized in the natural way it admits
2L − 1. Halve it and it holds the cap, but the flood then gets 1249 paid
requests through where the log serves 2000. The lesson's "sliding-window +
burst allowance" is this trade-off: the log's burst allowance is the whole
quota.

**The cheap sliding-window counter overshoots by half.** It weights the
previous window's count as if that traffic were spread evenly. After an idle
start it was not, and the counter admits 1.5L. Its memory is two integers per
tenant, against up to L timestamps for the log (1000 per paid tenant). Choosing
it at scale means accepting a 1.5x tolerance, and that should be a deliberate
choice.

The lesson's vendor attributions were only partly checkable. Kong's AI Rate
Limiting Advanced plugin is Enterprise-only and limits by tokens or cost, not
requests. Its docs mention `window_type=sliding`. I could not confirm from
LiteLLM's docs which algorithm its `rpm_limit`/`tpm_limit` use.
