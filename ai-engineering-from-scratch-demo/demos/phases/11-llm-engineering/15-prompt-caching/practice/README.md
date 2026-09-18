<!-- generated:start -->
# 11-llm-engineering / 15-prompt-caching

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/11-llm-engineering/15-prompt-caching/) · upstream spec
`phases/11-llm-engineering/15-prompt-caching/docs/en.md`

```bash
uv run demo practice run 15-prompt-caching --ex 1
uv run demo explain 15-prompt-caching --ex 1
uv run pytest demos/phases/11-llm-engineering/15-prompt-caching
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Take a 10-turn conversation with a 5,000-token system prompt against Claude. Run it wit… | code | T0 | `ex01_the_token_bill_is_identical_and_only_the_price_changes.py` |
| 2 | Medium. Write a test harness that, given a prompt template and a request log, computes the ex… | code | T0 | `ex02_the_cheapest_provider_reports_the_worst_savings.py` |
| 3 | Hard. Build a layout optimizer: given a prompt and a list of fields marked `stable=True/False… | code | T0 | `ex03_the_simulator_caches_a_prefix_the_endpoint_would_refuse.py` |
<!-- generated:end -->

## Answers

The lesson is pure stdlib, so all three exercises are **T0** and run in CI.
Exercise 3 asks for verification against a real Anthropic endpoint; no
`ANTHROPIC_API_KEY` is available, so that step is documented and the layout is
priced against the lesson's own accountant instead.

Two multipliers explain most of the lesson: a cached prefix costs **1.25×**
base to write and **0.10×** base to read. Since 1.25 + 0.10 < 2.00, the
break-even is two requests against the same prefix inside the TTL — and a
layout that writes more often than it reads costs *more* than not caching.

### 1 — the token bill is identical and only the price changes

| 10 turns, 5,000-token system prompt | input tokens | dollars |
|---|---:|---:|
| no `cache_control` | 72,000 | $1.0800 |
| breakpoint after the system prompt | 72,000 | **$0.4912** (−54.5%) |
| breakpoint after the transcript | 72,000 | **$1.3350** (+23.6%) |

**ANSWER: caching changes the price of a token, never the count.** Any report
phrased in input tokens shows no effect at all — `simulate_anthropic` bills
every prefix token on every request and only picks between three rates.

**FINDING: the layout that maximises the cached prefix is the one that loses
money.** Caching the whole transcript changes the prefix key every turn, so all
10 turns are writes and none is a read. The +25% write premium is paid ten
times for nothing.

**FINDING: `ProviderStats.misses` is dead.** None of the four simulators ever
increments it — a miss is counted as a `write` — so `print_report` has printed
`misses 0` on every line it has ever printed.

### 2 — the cheapest provider reports the worst savings

| arm | writes | reads | hit rate | cost | saves |
|---|---:|---:|---:|---:|---:|
| Anthropic 5m | 21 | 479 | 0.958 | $19.6838 | 83.0% |
| Anthropic 1h | 3 | 497 | 0.994 | $15.5325 | **86.6%** |
| OpenAI automatic | 3 | 497 | 0.994 | $19.8625 | 48.4% |
| Gemini 1h explicit | 3 | 497 | 0.994 | **$2.6365** | 72.6% |

**FINDING: the hit rate carries no information the cost does not.**
`writes + reads == 500` in every arm and `misses` is never set, so the hit rate
is exactly `1 − writes/500`. Three arms share a hit rate and their costs differ
by 7.5×.

**FINDING: the TTL is never refreshed on a read**, so the 5-minute arm rewrites
each prefix on a 300-second clock whatever the traffic does — 1,996 seconds of
log, 7 writes per prefix. Anthropic renews the window on every hit:

```text
ttl  300   shipped: 21 writes, $19.6838   renewed: 3 writes, $15.0263
ttl 3600   shipped:  3 writes, $15.5325   renewed: 3 writes, $15.5325
```

The option the shipped model prices as the expensive one is the cheaper one,
and the harness recommends paying the 2× write premium for nothing.

**FINDING: Gemini's storage is billed once per surviving entry.** The loop runs
over `cache.values()` *after* the simulation, so at a 300-second TTL the arm
records 21 writes and bills 3 tenancies — 18 free. And `simulate_openai`
hardcodes `3600` in its body, so the one arm described as "automatic" is the
one that cannot be run at another TTL.

**FINDING: dollar savings are not comparable across providers.** Each arm is
divided by its own baseline, so the percentage measures how cacheable a price
list is, not how cheap it is. Gemini is 5.9× cheaper than Anthropic 1h and
reports the worse saving. A harness that ranks by savings picks the dearest
provider.

### 3 — the simulator caches a prefix the endpoint would refuse

```text
original order   role(800) tools(3200) | name(20) style(1500) date(15) docs(2000) schema(600) q(120)
                 ^ cacheable prefix: 4,000 tokens
rewritten        role(800) tools(3200) style(1500) schema(600) | name date docs q
                 ^ cacheable prefix: 6,100 tokens   (+52.5%)
```

**ANSWER: 4,000 → 6,100 tokens**, every label preserved and the total unchanged
at 8,255. Over 100 requests: **$7.1205 → $4.3580** against a $12.3825 baseline,
42.5% saved becoming 64.8% — the 2,100 tokens moved are worth $2.76 per hundred
requests at Opus prices.

**MECHANISM: no number of breakpoints substitutes for the reorder.** A
breakpoint caches a *prefix*, and the first unstable field truncates every
prefix after it, so the 2,100 stable tokens behind `user display name` are
unreachable however Anthropic's four breakpoints are spent.

**FINDING: the simulator has no minimum cacheable prefix.** Anthropic will not
cache below **1,024 tokens**; `simulate_anthropic` will. A 900-token stable
block is modelled as 2 writes and 98 reads and reported as **9.6% saved** — a
saving the endpoint would refuse. The optimizer has to carry the minimum itself
and decline the breakpoint.

**FINDING: "without losing information" holds for the fields and not for the
order.** The rewrite moves `style guide` and `output schema` — the latter
conventionally last, so the model reads it closest to generation — in front of
the retrieved documents and the question. Labels and token counts survive;
position does not, and position is instruction in a prompt. That cost falls on
2 of the 4 stable fields, and the exercise does not mention it.
