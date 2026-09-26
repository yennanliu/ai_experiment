<!-- generated:start -->
# 17-infrastructure-and-production / 01-managed-llm-platforms

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/01-managed-llm-platforms/) · upstream spec
`phases/17-infrastructure-and-production/01-managed-llm-platforms/docs/en.md`

```bash
uv run demo practice run 01-managed-llm-platforms --ex 1
uv run demo explain 01-managed-llm-platforms --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/01-managed-llm-platforms
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. At what sustained utilization does Azure PTU beat on-demand for a 70B cla… | code | T0 | `ex01_the_50_percent_break_even_holds_only_because_every_token_is_billed_at_the_output_rate.py` |
| 2 | Your product needs Claude 3.7 Sonnet and GPT-4o. Design a two-provider deployment — which goe… | code | T0 | `ex02_the_lessons_pair_gives_each_model_one_host_so_its_failover_swaps_the_model.py` |
| 3 | A regulated healthcare customer requires BAAs, US-East data residency, and sub-100ms P99 TTFT… | code | T0 | `ex03_only_reserved_capacity_meets_100ms_and_the_comparator_prices_it_away.py` |
| 4 | You discover your Bedrock bill is up 4x this month with no traffic change. Without Applicatio… | code | T0 | `ex04_profiles_find_the_culprit_in_one_group_by_but_only_for_traffic_already_routed_through_them.py` |
| 5 | Read the Azure OpenAI and Bedrock pricing pages. For a 100M-token/month Claude workload, whic… | code | T0 | `ex05_on_demand_wins_and_provisioned_throughput_cannot_break_even_even_at_full_load.py` |
<!-- generated:end -->

## Answers

### 1 — the 50% break-even holds only because every token is billed at the output rate

**It breaks even at exactly 50%, the middle of the band. PTU first wins at
60%.** One Azure PTU costs $240/day and delivers 48M tokens/day. At $10/M
that much traffic would cost $480 on-demand, so u* = 0.5. At 50% the two tie
at $240, and the sweep's strict `<` gives the tie to on-demand. PTU wins
every row from 60% up. The largest saving is 50%, at 100% utilization, so
the lesson's "up to ~70%" can't happen in this model.

**That 50% comes from one pricing shortcut.** `break_even_demo()` bills
every token at the output rate. Add the $2.50 input rate back in, at the
mixes `simulate()` itself uses:

| token mix (input:output) | $/M blended | break-even |
|---|---:|---:|
| output only (the sweep) | 10.00 | 50% |
| 2:1 | 5.00 | 100% |
| 3:1 | 4.375 | 114.3% |

At both real mixes, one PTU never strictly beats on-demand. The module
contradicts itself on this: at 45M tokens/day, 93.75% of one PTU,
`simulate()` picks on-demand ($225 vs $240), while the sweep says PTU wins
at 90%.

**There is no 70B-class model to run.** The sweep is labelled "GPT-4o
class", and `PLATFORMS` has no 70B entry. Break-even moves inversely with
the token price. At this PTU price the 40–60% band corresponds to
$12.50–$8.33/M blended, and any rate below $5/M puts break-even above 100%.
The reference's Bedrock row breaks even at 116.7% even at its output rate.

### 2 — the lesson's pair gives each model one host, so its failover swaps the model

**The design:**

- **Placement:** Claude on Bedrock, GPT-4o on Azure OpenAI. The split is
  forced, because GPT-4o is only in Azure's catalog and Claude is not in it.
- **Gateway:** one self-hosted, LiteLLM-style gateway in front, with a key
  and a budget per team.
- **Failover:** same model first, a different model last. Claude goes
  Bedrock → Claude on Vertex Model Garden → GPT-4o on Azure. GPT-4o goes
  Azure → Claude on Bedrock.

The lesson gives no availability figure, so each provider is assumed to be
up 99.9% of the time. Enumerating all 8 up/down states:

| routing table | Claude answered by Claude | Claude downtime / month | GPT-4o requests answered |
|---|---:|---:|---:|
| pinned (no failover) | 0.999 | 43.2 min | 0.999 |
| lesson's pair, cross-model failover | 0.999 | 43.2 min | 0.999999 |
| design above | 0.999999 | 0.04 min | 0.999999 |

The lesson's "Claude from one, GPT from the other, failover between them"
serves Claude users GPT-4o during a Bedrock outage. Same-model availability
does not move. Only a second Claude host fixes that. GPT-4o has no second
hyperscaler in the lesson's catalogs, so its prompts have to work on Claude
too. Failing over to GPT-4o costs less, not more: $4.375/M against $6.00/M
at 3:1.

**Redundancy is not 13% of spend.** `lock_in_cost()` charges $195/month on
$1,500. That includes 10% for "idle secondary headroom", but an on-demand
secondary costs nothing while idle, which leaves the 3% gateway, $45. If the
headroom is reserved capacity, 10% of it absorbs only 10% of a failed-over
load as large as the primary's. The real limit is the secondary's
rate-limit quota, which the module does not model.

Claude 3.7 Sonnet itself is no longer on Anthropic's price list (read
2026-09-26), so a real deployment would pin a current Sonnet.

### 3 — only reserved capacity meets 100 ms, and the comparator prices it away

**Azure OpenAI on a PTU in an East US region.** The three features:

1. **BAA.** The lesson lists HIPAA for Azure OpenAI and says all three
   platforms offer BAAs.
2. **US-East residency.** A regional deployment in East US, not a global
   one.
3. **Dedicated capacity.** Nothing on shared capacity meets the SLA. The
   best on-demand P99 is 140 ms.

On the reference's 30M-in / 15M-out daily workload, choosing the path that
meets the SLA first and then the cheapest:

| platform | path | $/day | P99 |
|---|---|---:|---:|
| Azure OpenAI | 1 PTU | 240 | 57 ms |
| Bedrock | 2 PT units | 1,008 | 82.5 ms |
| Vertex | none (no reserved path in the code) | — | 160 ms, fails |

**The reference's `simulate()` fails the customer on all three platforms.**
It picks the cheapest path before it checks the SLA. It takes Azure
on-demand at $225, even though the PTU that passes costs only $15 (6.7%)
more.

**The code can't evaluate two of the three requirements, and the third is
set by a fixed rule.** `Platform` has no BAA, region or residency field. PTU
P99 is `P50 × 1.5` rather than a measured spread, and `random` and
`statistics` are imported but never used.

**The lesson's latency numbers contradict each other.** It quotes Azure
OpenAI at ~50 ms "on Llama 3.1 405B equivalents", but says the Azure OpenAI
catalog has no non-OpenAI models. It also labels that ~50 ms "(shared
on-demand)" in one section and "(with PTUs)" in another. The code follows
the first: 50 ms is the shared median and 38 ms the PTU median.

### 4 — profiles find the culprit in one group-by, but only for traffic already routed through them

"No traffic change" means requests are flat, so the 4x has to come from more
tokens per request. The synthetic month has four features on one Claude
model at the reference's $3/$15 Bedrock rates, and two of those features
share one IAM role. Requests stay at 85,000/day, while `support/summarize`
starts sending the whole ticket thread: its prompt grows from 3,000 to
89,250 tokens, and the bill rises exactly 4.0x.

| view | features it could be | what it reads |
|---|---:|---|
| bill / metrics by model | 4 | invocations flat, input tokens up, so traffic is ruled out |
| invocation logs by IAM role | 2 | 2,550,000 logged requests; then the prompts themselves have to be told apart |
| application inference profiles | 1 | one cost group-by over 120 daily rows |

**Without profiles:** go model → caller role → request body. That only
works if model invocation logging was already on before the spike.
Otherwise the model view, with 4 candidates, is all you have. **With
profiles:** a single group-by. The catch is that profiles attribute only
calls that name the profile's ARN. Created on day 20 to investigate, they
cover 36.7% of the month.

The module has no attribution to run. Its Bedrock "attribution" is just the
string `"A (Application Inference Profiles)"`. Also, AWS's inference-profile
page (read 2026-09-26) sends profile *usage* metrics to CloudWatch and
*costs* to cost allocation tags in AWS Billing. That is not the lesson's
"CloudWatch breaks out cost per profile".

### 5 — on-demand wins, and Provisioned Throughput cannot break even even at full load

Pricing pages read 2026-09-26. The 100M tokens are split 3:1 input:output,
and a month is 730 hours.

| path | $/month |
|---|---:|
| direct API, Claude Sonnet 5 ($2/$10) | 400 |
| direct API, Claude Sonnet 4.6 ($3/$15) | 600 |
| Bedrock on-demand (the reference's $3/$15), global | 600 |
| Bedrock on-demand, regional endpoint (+10%) | 660 |
| Bedrock Provisioned Throughput, 1 unit at the lesson's $21/hr | 15,330 |

**On-demand wins by 25x.** This workload would fill 11.4% of one unit.
Provisioned Throughput never breaks even in the reference's numbers. At
full load, a unit's 876M tokens a month would cost $5,256 on-demand at 3:1,
and $13,140 even if every token were billed at the output rate. The unit
costs $15,330. The case for it is capacity and latency, not price.

**The pages don't carry the numbers the exercise asks for:**

- Bedrock says "For Provisioned Throughput pricing, please reach out to
  your account team", so the lesson's "$21-$50/hr" is not on the page.
- The Bedrock page's rows for current Claude models did not render in the
  fetch. The Bedrock on-demand figure above is therefore the reference's
  rate, not one read off the page.
- The Azure OpenAI page lists no Claude model. Anthropic's pricing page
  says Claude on Microsoft Foundry bills at Anthropic's own rates.
- Claude 3.7 Sonnet is gone from Anthropic's price list. The legacy Claude
  3.5 Sonnet that Bedrock still serves costs $6/$30. Bedrock's lifecycle
  page says no new Provisioned Throughput can be created for a Legacy
  model.
