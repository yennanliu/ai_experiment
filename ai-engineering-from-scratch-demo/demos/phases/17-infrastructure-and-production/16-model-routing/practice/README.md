<!-- generated:start -->
# 17-infrastructure-and-production / 16-model-routing

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/16-model-routing/) · upstream spec
`phases/17-infrastructure-and-production/16-model-routing/docs/en.md`

```bash
uv run demo practice run 16-model-routing --ex 1
uv run demo explain 16-model-routing --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/16-model-routing
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. At what accuracy floor does cascade beat pre-route? | code | T0 | `ex01_the_shipped_classifier_is_perfect_and_cascade_wins_on_quality_only_below_71_percent_accuracy.py` |
| 2 | Your user base is 30% enterprise (complex queries), 70% free tier (simple). Design the routin… | code | T0 | `ex02_route_free_tier_through_a_cascade_because_tier_pre_routing_hides_complex_free_queries.py` |
| 3 | A route drops quality by 2% but saves 40%. Is that a ship? Depends on product — argue both. | code | T0 | `ex03_ship_only_if_a_bad_answer_costs_under_17_5_cents_and_the_2_percent_is_4_percent_on_medium_queries.py` |
| 4 | Implement a confidence check using logprobs from OpenAI / Anthropic APIs. What's the threshol… | code | T0 | `ex04_start_at_the_budget_quantile_of_mean_token_logprob_because_sum_escalates_by_length_and_anthropic_returns_none.py` |
| 5 | Over six months, escalation rate climbs from 8% to 22%. Diagnose three causes and the fix for… | code | T0 | `ex05_each_cause_leaves_its_own_fingerprint_and_neither_of_the_lessons_escalation_alarms_fires_on_the_climb.py` |
<!-- generated:end -->
## Answers

The lesson's `code/main.py` is a stdlib routing simulator: 1000 seeded
requests (629 simple, 276 medium, 95 hard), a per-token price for each model,
and a quality score per difficulty. Every solution reuses its `Query`,
`cost_of`, `quality` and `make_workload` rather than copying them.

### 1 — the shipped classifier is perfect, and cascade wins on quality only below 71% accuracy

The shipped run:

| pattern | cost | quality | escalated |
|---|---:|---:|---:|
| NO_ROUTE | $7.52 | 100.0% | 0 |
| PRE_ROUTE | $5.38 | 99.37% | 0 |
| CASCADE | $4.42 | 98.24% | 229 |

PRE_ROUTE has no accuracy knob. It reads `q.difficulty` directly, so the
printout shows a perfect classifier. With a classifier that is right with
probability *a*, cost and quality are linear in *a*:

- **Quality.** Pre-route drops to cascade's 98.24% at **a = 0.713**.
- **Cost.** Pre-route drops to cascade's $4.42 at **a = 0.641**. Misrouting
  sends hard queries to the cheap model, and that is cheaper as well as
  worse.

So cascade beats pre-route on both axes only for classifiers between 64.1%
and 71.3% accurate. Above that range the choice is a trade.

**The cascade is not the "quality floor".** The module ends by saying
CASCADE "guarantees quality floor", and the lesson calls it the "Best
quality floor". Yet it scores below perfect pre-route. Its confidence check
also reads the label: simple is always kept, hard always escalated, and
medium is a 50% coin. That keeps 142 of 276 medium queries on the cheap
model at 0.92. It is pre-route with a worse classifier, plus the cheap call
on every escalation: $0.29 of its $4.42. A pre-route making the same
choices costs $4.13.

"Use It" says the code simulates an ensemble. It has none, and
`simulate("ENSEMBLE", ...)` silently returns cost 0 and quality 0.

### 2 — route the free tier through a cascade, because tier pre-routing hides complex free queries

**Design: free tier → cascade, enterprise → frontier. Gate on the free
tier's escalation rate, plus a 5% judged sample of cheap answers.**

| split | premise exact | 10% of free queries complex |
|---|---|---|
| all frontier | $6.61, 100% | $7.48, 100% |
| tier: free → cheap | $4.28, 99.31%, *no signal* | $4.35, 98.43%, *no signal* |
| free → cascade | $4.28, 99.31%, escalation 0% | $5.24, 99.16%, escalation 7.0% |
| cascade for all | $3.45, 98.34%, escalation 19.1% | $4.42, 98.20% |

When the premise holds, the cascade split costs exactly the same as the tier
split, because simple queries never escalate. When the premise leaks, the
tier split loses quality and nothing in its metrics moves. The cascade
split's escalation rate goes from 0 to 7.0% at a 10% leak and 13.7% at 20%.
That rate is the online gate.

The saving is **35.2%, not the ~65%** the Problem section gives for this
exact scenario. Free-tier traffic is 70% of requests but only 38.0% of
frontier spend. On it the cheap model costs 7.5% of frontier, not 3%.

### 3 — ship only if a bad answer costs under 17.5 cents, and the 2% is 4% on medium queries

The lesson's own CASCADE is this route: 41.2% saved, 1.77% quality lost.
Read the loss as a probability of a worse answer. The route saves $0.0031
per request, so it pays off when a degraded answer costs the product less
than **$0.0031 / 0.01765 = $0.175**. At the lesson's $80k/month that is
$32,940 saved against about 187,800 degraded answers a month.

- **Ship** for consumer chat or rephrasing. A weak answer costs a re-ask,
  a fraction of a cent.
- **Don't ship** for a coding assistant or support agent. A wrong answer
  costs an engineer's minute or an escalated ticket: dollars, not cents.

The average also hides where the loss lands. Simple queries lose 1.0%,
medium queries 4.1%, hard queries 0%. Medium queries carry 64% of the loss,
and they are the paying tier's traffic. Hard queries lose nothing only
because the cascade never keeps one cheap. The cheap model's 0.75 on hard
queries is the worst number in the module, and it never enters the average.

### 4 — start at the budget quantile of mean token logprob, because sum escalates by length and Anthropic returns none

**What the APIs give you.** OpenAI Chat Completions with `logprobs=True`
returns `choices[0].logprobs.content[]`, one `{token, logprob,
top_logprobs}` per output token. Anthropic's Messages API reference lists no
logprob parameter or field (checked 2026-09-26). On Claude the check has to
be a verifier call or a hedging/refusal check instead. The solution's
`token_logprobs()` returns None on an Anthropic-shaped body.

**The starting threshold is a quantile, not a constant.** Logprob scales
differ between models, so a fixed number does not carry over. Start from the
escalation budget. The lesson's "~10% of traffic" is the 10th percentile of
the mean token logprob on a calibration sample. On synthetic OpenAI-shaped
bodies for the reference workload, that is **−0.0946, a geometric-mean
token probability of 0.910**. The bodies assume a 3% unsure-token rate in
correct answers and 6% in wrong ones; that is the solution's assumption,
not measured model behaviour.

| score, 10% budget | escalated (simple/medium/hard) | wrong caught of 49 | cost |
|---|---|---:|---:|
| mean logprob | 52 / 32 / 16 | 37 | $1.44 |
| min logprob | 27 / 39 / 34 | 14 | $2.06 |
| sum logprob | 0 / 24 / 76 | 29 | $2.99 |

The summed logprob escalates by length, not by doubt: a long answer has a
low total however sure each token is. Normalise by length. The reference
cascade's "confidence" uses no logprob at all.

### 5 — each cause leaves its own fingerprint, and neither of the lesson's escalation alarms fires on the climb

Month 0 is an 88/8/4 simple/medium/hard mix under the reference cascade
rule: 8.0% escalation, $1.76 per 1000 requests.

| cause | month 6 | fix | after |
|---|---:|---|---:|
| traffic harder: mix 68/16/16 | 21.7%, $5.37 | pre-route hard queries straight to frontier | 8.1%, $5.06 |
| cheap model regressed: medium always up, simple 11% | 22.1%, $2.52 | pin the model version, roll back, recalibrate | 8.0%, $1.76 |
| prompts longer: +3090 tokens of context, cheap gives up past 4K | 21.9%, $5.42 | length pre-route: >4K to frontier | 0.4%, $5.09 |

In the first case escalation is doing its job; the only waste is the doomed
cheap attempt on each hard query, $0.31 here. The second raises the bill 43%
on unchanged traffic.

**The log tells them apart.** Re-weight month 0's per-difficulty rates by
the new mix. If that explains the climb, the cause is mix. Otherwise the
excess escalations sit on long prompts (length) or on short ones (model).
`diagnose()` names all three correctly.

**Neither alarm fires.** The lesson flags escalation above 30%, and the
Ship It plan alerts on a climb of more than 10 points in a month. A steady
8 → 22% climb peaks at 22% and moves 2.33 points a month. Anchor the alert
to a baseline instead: +14 points since month 0.
