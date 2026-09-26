<!-- generated:start -->
# 17-infrastructure-and-production / 21-ab-testing-llm-features

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/21-ab-testing-llm-features/) · upstream spec
`phases/17-infrastructure-and-production/21-ab-testing-llm-features/docs/en.md`

```bash
uv run demo practice run 21-ab-testing-llm-features --ex 1
uv run demo explain 21-ab-testing-llm-features --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/21-ab-testing-llm-features
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. For an expected 5% lift with baseline 3% conversion, what sample size to… | code | T0 | `ex01_207702_per_arm_and_at_that_n_the_sequential_rule_the_code_demos_has_20_percent_power.py` |
| 2 | Pick Statsig or GrowthBook for a healthcare-regulated on-prem customer. | code | T0 | `ex02_growthbook_self_hosted_and_on_prem_its_cuped_and_sequential_need_an_enterprise_license.py` |
| 3 | Design an A/B that tests GPT-4 vs GPT-3.5 on cost-per-resolved-ticket. What's the primary met… | code | T0 | `ex03_counting_only_token_spend_gpt_3_5_wins_44x_and_loaded_with_escalations_gpt_4_wins_16_percent.py` |
| 4 | Your canary passes but A/B shows -1.2% conversion. Do you ship? Write the escalation criteria. | code | T0 | `ex04_no_at_the_planned_n_minus_1_2_percent_is_z_minus_0_68_and_the_canary_never_measured_conversion.py` |
| 5 | Apply CUPED to a pre-period with 60% of the variance of post. Compute the effective-sample-si… | code | T0 | `ex05_2_5x_if_the_pre_period_is_part_of_post_and_the_variance_ratio_alone_does_not_fix_it.py` |
<!-- generated:end -->

## Answers

### 1 — 207,702 per arm, and at that n the sequential rule the code demos has 20% power

**207,702 users per arm, 415,404 in total.** With the lesson's x1.4
non-determinism buffer that is 290,782 per arm. `main.py` prints the number
as "fixed sample size" and never says it is per arm. Exact normal quantiles
give 207,938, 0.1% more, and 207,702 per arm has power 0.7995.

- **`alpha` and `power` are accepted and ignored.** The function hardcodes
  1.96 and 0.84, so `power=0.9` and `alpha=0.01` both return 207,702.
- **The lesson's own 5%-lift run only reaches 72% of that n.**
  `simulate(0.03, 0.0315)` stops at 300,000 users in total, 150,026 of them
  in A. At the end its fixed-horizon z is 2.55, which a fixed test would
  call significant. The sequential boundary there is 4.31, so the sequential
  rule never stops.
- **At the 80%-power horizon the sequential rule has 20% power.** Run to
  2 x 207,702 users it stops in 8 of 40 seeds. Its boundary,
  sqrt(2 ln 20 + ln n), is 4.35 at that n, and the expected z is only 2.80.
  The lesson says the simulation "shows how sequential lets you stop
  early". Its 10%-lift demo stops at 159,731 users, 1.5x the 106,300 a
  fixed test needs.

### 2 — GrowthBook, self-hosted, and on-prem its CUPED and sequential need an Enterprise license

**GrowthBook, self-hosted.** It ships as one Docker image plus MongoDB and
reads metrics from the customer's own warehouse, so no data leaves the
network. If no data reaches GrowthBook Inc., there is no vendor to sign a
BAA with. Statsig fails on the on-prem requirement. Its Warehouse Native
mode runs compute in your warehouse, but you still manage experiments in
the Statsig console, and its docs describe no self-hosted control plane.
All vendor facts come from the vendors' own docs and pricing pages, fetched
2026-09-26.

**The lesson's two criteria never ask the deciding question.** It says to
pick on warehouse-SQL preference and on whether OpenAI ownership matters. A
team with no warehouse preference that doesn't mind OpenAI gets Statsig
from those two, and Statsig can't be installed on-prem.

**"Open-source (MIT) ... CUPED, SRM, Bonferroni, BH" is not one product.**

- The GrowthBook repo is MIT except for three `enterprise` directories.
- The docs say "CUPED is available on Pro and Enterprise plans", and the
  same for sequential testing.
- Pro is cloud-only. Self-hosting means either the free open-source edition
  (one project) or a custom Enterprise agreement.
- Audit logs, custom OIDC SSO and the BAA are Enterprise too.
- The MIT build keeps SRM checks, multiple-testing corrections and both
  engines.

So this customer's four needs (CUPED, sequential testing, audit logs, SSO)
mean a self-hosted Enterprise license, or CUPED written in its own SQL. The
reference code offers no CUPED either. A secondhand comparison lists Statsig
as HIPAA-eligible with a BAA; that could not be confirmed on Statsig's own
pages, and it does not change the pick.

### 3 — counting only token spend GPT-3.5 wins 44x, and loaded with escalations GPT-4 wins 16%

The design is simulated on 10,000 seeded tickets per arm, randomized by
ticket. Prices are OpenAI's legacy list prices: GPT-4 8k at $30/$60 per M
tokens, gpt-3.5-turbo at $0.50/$1.50. These inputs are **assumptions**:
60% vs 50% resolution, 2,000 input and 400 output tokens per ticket, $4 per
human-handled escalation.

- **Primary: fully loaded cost per resolved ticket.** That is LLM spend plus
  the human cost of escalations, divided by tickets closed. Every ticket
  eventually closes, so this is a per-ticket mean and a two-sample z-test
  applies. GPT-4 comes in at $1.67 and GPT-3.5 at $1.98, z = -11.0.
- **Guardrails: 7-day reopen rate per resolved ticket, CSAT and P99
  latency.** None may be significantly worse. The simulated reopen z is
  -0.17.
- **Secondary: bot resolution rate and tokens per ticket.** The simulated
  resolution z is 13.9.

**The metric's definition picks the winner before the test runs.** Counting
only LLM spend per bot-resolved ticket, GPT-4 costs $0.139 and GPT-3.5
$0.0032, so GPT-3.5 is 44x cheaper and you don't need an A/B to learn that.
Adding the cost of escalations makes GPT-4 16% cheaper. For model selection
the lesson suggests "accuracy + cost/request + latency", which is the first
definition.

**Power the test on the break-even gap.** GPT-4 pays for itself when its
resolution rate beats GPT-3.5's by ($0.084 - $0.0016) / $4 = **2.06 points**.
The reference `fixed_sample_size` needs 9,232 tickets per arm to detect that
gap at 50%. The reference's only test, `z_statistic`, takes success counts,
so it can size the resolution rate but cannot test the cost metric.

**Divide the reopen guardrail by resolved tickets, not all tickets.** Both
arms reopen 5% of what they resolve. GPT-4 resolves more tickets, so counted
per ticket it reopens 1.18x as often (z = 1.96), right at the point of
tripping the guardrail on a model that is no worse.

### 4 — no: at the planned n, -1.2% is z = -0.68, and the canary never measured conversion

**Don't ship; hold and escalate.** This uses exercise 1's plan: 3% baseline
and 207,702 users per arm.

- **Read as relative (3% to 2.964%):** z = -0.68, and the 95% interval runs
  from -4.7% to +2.2% relative. That doesn't prove harm, but it can't rule
  out a 4.7% loss, and a variant with no effect at all reads -1.2% or worse
  25% of the time. The lesson's own skill says "primary significant + all
  guardrails not significant-negative -> ship", and a non-significant
  primary is not a ship.
- **Read as absolute (3% to 1.8%):** z = -25.3. Roll back.

The escalation criteria, as `decide()` runs them:

1. The interval's upper bound is below 0: roll back now.
2. The interval's lower bound is below the harm margin (-1% relative):
   hold, extend the test, and escalate to the metric owner with the sample
   size needed.
3. Otherwise, ship only if the primary is significantly positive, or if a
   non-inferiority test at that margin passes and there is a written
   sign-off.

**The canary can't see conversion, so its "pass" was guaranteed.** Lesson
20's five canary gates are latency, cost, error rate, output length and
thumbs-down. None is conversion, and a regression-free rollout passes all
six stages. **Confirming a 1.2% drop takes 3,500,264 users per arm**, 16.9x
the plan. The sequential rule the lesson demos needs |z| > 4.35 at the
planned n, so it can't fire on an effect this size.

### 5 — 2.5x if the pre-period is part of post, and the variance ratio alone does not fix it

**2.5x.** CUPED leaves var(Y)(1 - rho^2) of the variance, so the
effective-sample-size boost is 1 / (1 - rho^2). Take post = pre + fresh
noise: then cov(X, Y) = var(X), so rho^2 = 0.6 and the boost is
1 / 0.4 = 2.5. On 20,000 seeded Gaussian users CUPED keeps 0.401 of the
variance, a 2.49x boost. That takes exercise 1's 207,702 users per arm to
83,081, and the buffered 290,782 to 116,313, so CUPED more than pays for the
lesson's x1.4 buffer.

**The correlation sets the boost, not the variance ratio.** Keep
var(X) = 0.6 var(Y) but set rho = 0.5 and the boost is 1.33x. Make X
independent of Y and it is 1.00x. Multiply X by 10 and nothing changes,
because CUPED doesn't depend on the covariate's scale. The 60% only
determines the answer because post contains pre. The lesson's "30-70%"
variance reduction corresponds to a 1.43x-3.33x boost.

**The reference code has no CUPED.** Its docstring says it "Illustrates
CUPED-style variance reduction". That sentence is the only occurrence of
"CUPED" in the module, and no function takes a pre-period.
