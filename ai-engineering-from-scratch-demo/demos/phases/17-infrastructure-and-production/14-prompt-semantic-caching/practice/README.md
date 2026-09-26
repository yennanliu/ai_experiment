<!-- generated:start -->
# 17-infrastructure-and-production / 14-prompt-semantic-caching

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/14-prompt-semantic-caching/) · upstream spec
`phases/17-infrastructure-and-production/14-prompt-semantic-caching/docs/en.md`

```bash
uv run demo practice run 14-prompt-semantic-caching --ex 1
uv run demo explain 14-prompt-semantic-caching --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/14-prompt-semantic-caching
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Toggle the parallelization flag. How much does the bill change? | code | T0 | `ex01_the_flag_moves_the_bill_17_percent_not_5x_and_the_whole_penalty_is_paid_in_the_first_106_requests.py` |
| 2 | Your system prompt has a date. Move it out. Show before/after hit rate math. | code | T0 | `ex02_a_request_id_in_the_prefix_takes_hit_rate_from_99_to_0_and_bills_more_than_no_cache.py` |
| 3 | Calculate break-even for 1-hour TTL (2x write) vs 5-minute TTL (1.25x write) given your reque… | code | T0 | `ex03_one_hour_wins_above_one_request_per_110_minutes_and_the_simulator_never_expires_an_entry.py` |
| 4 | Semantic cache at 0.95 threshold hits 20%. At 0.85 it hits 50% but you see incorrect cached r… | code | T0 | `ex04_keep_0_95_because_behind_a_prompt_cache_a_semantic_hit_saves_0_43_cents.py` |
| 5 | You batch 10 parallel sub-queries per user question. Rewrite for cache-friendliness without a… | code | T0 | `ex05_let_the_planner_call_write_the_cache_and_the_fan_out_reads_it_5_5x_cheaper_at_zero_added_latency.py` |
<!-- generated:end -->

## Answers

### 1 — the flag moves the bill 17%, not 5x, and the whole penalty is paid in the first 106 requests

**$6.84 → $5.85, a 17% (1.17x) change.** Toggling `parallel_penalty` turns
65 cache writes into reads (77 → 12). On input alone the bill goes
$2.93 → $1.94 (1.51x). The $3.91 of output is the same in both rows, because
every request pays a flat 200 output tokens.

**Every penalised write lands in the first 106 requests.** The simulated cache
never expires, and single (non-wave) requests fill it. Once each of the 12
prefixes has been seen by a single request, no wave can miss again. So the
penalty is a cold-start cost, and it shrinks as traffic grows while the write
count stays at 77:

| iterations | ratio | writes |
|---:|---:|---:|
| 500 (shipped) | 1.17x | 77 |
| 2000 | 1.04x | 77 |
| 5000 | 1.02x | 77 |

**The waves do not share a prompt, and "500 requests" is really 1304.**
`make_workload(500)` loops 500 times and emits 5 requests per wave, so it
produces 1304 requests, 1005 of them in 201 waves. Each wave member draws its
own prefix: none of the 201 waves shares one, and a wave averages 4.2
distinct prefixes. The lesson describes 10 tool calls sharing one system
prompt, which is not what this code simulates.

**5x needs 8 cold parallel calls on one prefix, and 10x needs 46.** Compare N
simultaneous misses with 1 write plus N−1 reads. Input then costs
N·1.25 / (1.25 + 0.1(N−1)):

| N | multiple |
|---:|---:|
| 5 (the simulator's wave) | 3.79x |
| 8 | 5.13x |
| 10 (the lesson's example) | 5.81x |
| 46 | 10x |

The ceiling is 12.5x (1.25 / 0.1).

### 2 — a request ID in the prefix takes hit rate from 99% to 0% and bills more than no cache

The simulator stores only a hash per prefix, so I model the dynamic content
by appending a stamp to that hash, built from `arrived_at`. Hit rate is then
1 − distinct prefixes / requests:

| stamp in prefix | distinct prefixes | hit rate | bill |
|---|---:|---:|---:|
| request ID | 1304 | 0.0% | $26.06 |
| time to the second | 1091 | 16.3% | $22.58 |
| time to the minute | 108 | 91.7% | $7.23 |
| day, or moved out | 12 | 99.1% | $5.85 |

Moving the stamp out gives 1 − 12/1304 = 99.1%. The minute row follows
1 − 1/(r·g). Each prefix sees r = 1304 / 12 / 519 s = 0.209 requests per
second. With g = 60 s, that is 12.6 requests per stamp value, so the formula
predicts 92.0% against the 91.7% measured.

**With a request ID in the prefix, caching costs 20% more than no caching.**
Every request misses and pays the 1.25x write, for $26.06 against $21.63
with caching off.

**The same timestamp costs 7 points or 83, depending on traffic.** What
matters is how many requests share each stamp value, not whether a stamp is
there at all. A day stamp costs nothing within a single day.

**ProjectDiscovery's post is dated 2026-04-10, not 2025-11, and ends at 84%.**
I fetched it on 2026-09-26. Moving dynamic context to the conversation tail
took them from "under 8% to 74% overnight", and later optimisations reached
84%.

### 3 — one-hour wins above one request per 110 minutes, and the simulator never expires an entry

I model the TTL the way Anthropic documents it: "The cache is refreshed for no
additional cost each time the cached content is used" (fetched 2026-09-26).
With Poisson reuse at rate L per minute, a request misses with probability
e^(−L·TTL). Per request:

- 5-min costs 1.25e^(−5L) + 0.1(1 − e^(−5L))
- 1-hour costs 2e^(−60L) + 0.1(1 − e^(−60L))

The two are equal at L = ln(1.9/1.15)/55 = 0.00913 per minute, which is **one
reuse every 109.5 minutes**. Above that rate 1-hour is cheaper; below it,
5-min is. A seeded simulation agrees: 1-hour is cheaper at a 100-minute mean
gap and 5-min at 120.

Periodic traffic gives a sharper answer. At any fixed gap from 5 to 60
minutes, 1-hour reads at 0.1x while 5-min rewrites at 1.25x every time,
which is 12.3x cheaper over 1000 requests at a 30-minute gap. Past 60 minutes
both TTLs always miss, and 5-min wins at 1.25x against 2x.

**The lesson's own traffic never reaches either TTL.** The longest gap between
two uses of a prefix is 57 seconds, so 1-hour buys no extra hits and only
raises the bill, $5.85 → $5.96.

**The simulator's TTL is a price, not a lifetime.** `simulate()` never reads
`arrived_at` and never evicts anything. In the reference, 1-hour therefore
loses at every arrival rate, so the break-even cannot be computed from it.

### 4 — keep 0.95, because behind a prompt cache a semantic hit saves 0.43 cents

On the lesson's workload, with L2 underneath, raising L1 from 20% to 50% hits
cuts the bill from $4.75 to $3.07. That is 389 extra hits at $0.0043 each,
$0.0030 of which is output.

Let e be the fraction of those extra hits that are wrong, and E what one wrong
answer costs you. Then 0.85 pays only if e·E < $0.0043:

- at E = $1, e must stay below 0.43%, or 1 wrong in 232
- at E = $10, e must stay below 1 wrong in 2325

The exercise says you already *see* incorrect responses at 0.85. An error
rate you can notice is far above either bound, so keep 0.95. If you want to
lower it, go down in steps with a labelled eval of each new band, and stop
where the band's error rate crosses $0.0043 / E.

**The prompt cache makes a semantic hit worth 3.8x less.** Without L2, a hit
also skips 4529 fresh input tokens and saves $0.0163. Stacking the two layers
shrinks exactly the margin that a lower threshold spends.

**`l1_threshold` does nothing in the reference.** `simulate()` never reads it.
An L1 hit is a coin flip, and every hit is counted as free and correct, so
the simulator cannot show the tradeoff this exercise asks about.

### 5 — let the planner call write the cache, and the fan-out reads it 5.5x cheaper at zero added latency

The setup is one question: a planner call, then 10 sub-queries. All calls
share a 5000-token prefix (a 4000-token system prompt plus 1000 tokens of
question context). A cache entry becomes visible 300 ms after the request
that writes it starts. Latencies are fixed virtual values (planner 1.5 s,
sub-query 1.0 s), so nothing depends on this machine.

**The fix: put the cache breakpoint at the end of the shared prefix in the
planner call, then fan out after the planner returns.** The fan-out has to
wait for the planner anyway, so the planner becomes the "sequential first"
request at no extra cost:

| strategy | input bill | writes / reads | end-to-end |
|---|---:|---:|---:|
| no caching | $0.1686 | 0 / 0 | 2.5 s |
| naive parallel | $0.2061 | 10 / 0 | 2.5 s |
| sequential-first (lesson) | $0.0509 | 1 / 9 | 2.8 s |
| planner-primed | $0.0374 | 1 / 10 | 2.5 s |

Planner-primed is 5.5x cheaper than naive and 27% cheaper than the lesson's
fix. It adds no latency, while the lesson's fix adds 300 ms. Naive parallel
caching also costs 22% more than not caching at all.

**Without a planner, keep only the system prompt warm.** If the sub-queries
are templated client-side, there is no earlier call to write the cache for
free. In that case:

- move the question context after the breakpoint
- keep the 4000-token system prompt warm, through normal traffic or a
  keepalive sent at least every 5 minutes

That gives 10 reads at $0.045 against $0.1905 cold, 4.2x cheaper, with no
added latency.

**The reference's serialize-first fix costs no latency.** The simulator has no
clock, so it can't model the wait. Anthropic's docs say "a cache entry only
becomes available after the first response begins", so the wait is time to
first token, and the planner's reply already takes longer than that.
