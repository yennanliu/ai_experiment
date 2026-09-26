<!-- generated:start -->
# 17-infrastructure-and-production / 27-finops-llms

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/27-finops-llms/) · upstream spec
`phases/17-infrastructure-and-production/27-finops-llms/docs/en.md`

```bash
uv run demo practice run 27-finops-llms --ex 1
uv run demo explain 27-finops-llms --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/27-finops-llms
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. At what z-score does the kill switch fire? How do you pick the threshold? | code | T0 | `ex01_the_kill_switch_fires_at_z_4_8_on_the_normal_tenant_for_97_cents.py` |
| 2 | Design a per-tenant, per-task cost dashboard. What are the 5 views you build first? | code | T0 | `ex02_the_dashboard_needs_a_layer_view_in_dollars_because_response_is_5pct_of_tokens_and_30pct_of_cost.py` |
| 3 | Your largest tenant is unit-economics-negative. Propose three interventions ordered by custom… | code | T0 | `ex03_caching_37pct_of_input_closes_a_30pct_loss_and_trimming_output_would_need_a_77pct_cut.py` |
| 4 | Compute cost per resolved ticket for a support product: 3M tokens/ticket, ~800 tickets/day, G… | code | T0 | `ex04_the_cached_rate_prices_a_ticket_at_0375_but_a_10pct_response_share_alone_costs_300.py` |
| 5 | Argue whether retroactive tagging can ever work. When is it acceptable? | code | T0 | `ex05_retroactive_attribution_is_exact_per_key_and_12pct_wrong_once_one_tenant_runs_agents.py` |
<!-- generated:end -->

## Answers

Prices are GPT-5 and GPT-5 mini list prices from OpenAI's model pages, read
2026-09-26: $1.25 / $0.125 cached / $10 and $0.25 / $0.025 / $2 per 1M tokens.
Every exercise runs or reads the lesson's `code/main.py`, a three-tenant
simulator of the enforcement ladder.

### 1 — the kill switch fires at z = 4.80 on the normal tenant, for 97 cents

**It fires once, at z = 4.80, on `tenant_A_normal`.** That is day 8, on $0.97
of spend against a baseline of $0.60 ± $0.08, or 1% of the tenant's $100
contract. The tenant the demo calls abusive peaks at z = 1.89 and is never
touched, and the spend cap never prints a breach.

**The "abusive" tenant is its own baseline, and it stays inside its
contract.** `tenant_C_abusive` has 25x traffic from day 1, so the abuse is
already in its history and z sees nothing unusual. It averages $15.38 a day
against a $20 contract and a $40 cap. A z-score against a tenant's own
history can only catch a *change*.

**How to pick the threshold: from a false-pause budget, and z alone is not
enough.** Over 1000 seeds × 3 steady tenants (10 days, armed from day 6):

| z | steady tenants paused | 2x day caught | 3x day caught |
|---:|---:|---:|---:|
| 2 | 30.2% | 64.9% | 81.5% |
| 3 | 11.1% | 55.1% | 85.2% |
| 4 | 5.2% | 42.2% | 80.5% |
| 5 | 1.8% | 30.1% | 72.0% |
| 6 | 0.8% | 19.8% | 62.4% |
| 8 | 0.3% | 8.8% | 42.5% |

At z = 4 the switch pauses one steady tenant in 19 and misses most doublings.
Also requiring spend over contract cuts false pauses to 1.5%, all of them
tenant C, the only tenant that ever spends near its contract. Surprise and
dollars are separate tests, and a pause should need both.

The ladder's first rung is missing: `simulate_day` never reads
`rate_limit_per_min` or `minute_count`.

### 2 — the dashboard needs a layer view in dollars, because response is 5% of tokens and 30% of cost

The five views, built over a seeded 3000-trace log in the lesson's trace
shape:

1. **Tenant spend vs contract**, carrying the enforcement-ladder state.
2. **Cost per resolved outcome, by task**, which is the lesson's unit metric.
3. **Layer dollars, by task**, which is where the levers get chosen.
4. **Per-user spend, by tenant**, for power users and abuse inside a tenant.
5. **Route × cache split**, to show whether routing and caching are engaged.

Each view partitions the same bill, and all five sum to the log's $19.35.

**The layer view has to be in dollars.** The lesson's sample trace is
61.0 / 20.3 / 13.6 / 5.1% of tokens (prompt / tool / memory / response), so it
falls outside the lesson's own "typical %" table for prompt (40-60%) and
response (10-30%). Priced at GPT-5 it is 45 / 15 / 10 / 30% of dollars.
Output costs 8x input, so a token-share chart shows the output-length lever
at about a sixth of its real size.

The reference records enough for view 1 only. `TenantState` has no user,
task, route or layer field. The lesson's skill file lists five views with no
layer view, even though the same file hard-rejects single-bucket billing.

### 3 — caching 37% of input closes a 30% loss; trimming output would need a 77% cut

Take a tenant whose cost is 1.3x its revenue, with each call shaped like the
sample trace at GPT-5 prices ($0.0035 input + $0.0015 output). Ordered by
customer impact, with the share each lever needs *on its own* to break even:

| impact | intervention | break-even share |
|---|---|---:|
| none | cache input prefixes | 36.6% of input tokens |
| none–low | route calls to GPT-5 mini | 28.8% of calls |
| visible | batch async calls | 46.2% of calls |
| visible | trim response tokens | 76.9% of response |
| contractual | reprice at renewal | +30% |

The levers customers never see are also the easiest to reach. Output
trimming is the most visible lever and the weakest one.

The lesson's "~5-10% of baseline" holds only if every call is batched.
Stacking cache, batch and route (60%) gives 7.4% of the bill. Without batch,
as an interactive product has to run, it gives 14.8%.

In the reference, reading `contracted_daily_usd` as revenue, the ladder
pauses the most profitable tenant and never alerts on the losing one. Seed-7
margins are 99.3% (A), 96.7% (B) and 23.1% (C). The kill switch pauses A.
Over 1000 seeds, C spends over its contract on 17.0% of days and never
crosses its 2x cap.

### 4 — the cached rate prices a ticket at $0.375, but a 10% response share alone costs $3.00

**$0.375 per ticket, $300 a day, $9,000 a month at the cached rate, and that
is a floor.** Uncached input is $3.75 a ticket. With 90% of input hitting the
cache and a 10 / 20 / 30% response share, a ticket costs $3.64 / $6.57 /
$9.50. At 10%, output alone is $3.00, 8x the literal answer.

"Per resolved" needs a resolution rate the exercise does not give.
Unresolved tickets burn tokens too, so at $3.64 a handled ticket the cost
per resolved ticket is $3.64, $4.55 and $6.07 at 100%, 80% and 60%
resolution.

The reference `simulate_day` prices one 3M-token request at $30.00, because
its flat $10/M is GPT-5's output price. That is 80x the cached-rate answer.
The lesson's sample trace fits neither price: its `cost_usd: 0.0135` for 2950
tokens compares with $0.0050 at GPT-5 uncached, $0.00185 cached, and $0.0295
at the flat $10/M.
And with a 400k context window, 3M tokens is at least 8 calls, so the cache
hit rate depends on each ticket's prefix staying stable across turns.

### 5 — retroactive attribution is exact per key and 12% wrong once one tenant runs agents

Measured on a seeded 29,457-call log in the reference's traffic model:

| method | misattributed |
|---|---:|
| API key per tenant | 0.0% |
| shared key, split by daily request counts | 1.5% |
| shared key, split by request counts over the period | 1.1% |
| same, one tenant at 4x tokens per request (daily) | 12.2% |
| same, one tenant at 4x tokens per request (period) | 12.0% |

**Retroactive attribution works when the dimension is already on the billing
record, and approximately when requests are alike.** It is acceptable for
showback, or when every tenant has its own key or project. It is not
acceptable for invoicing, for kill switches, or when request sizes differ by
tenant, which is the agent workload. Daily buckets do not help, because the
error comes from request size and not from timing.

Two corrections:

- Allocation conserves the bill to the cent. The skill file's "retroactive
  tagging loses ~10-30% of spend" describes misattribution, and the
  measured misattribution runs from 0% to 12% depending on how alike the
  tenants' requests are.
- The reference simulator cannot show the failure at all, because its
  tenants all draw tokens from one distribution.
