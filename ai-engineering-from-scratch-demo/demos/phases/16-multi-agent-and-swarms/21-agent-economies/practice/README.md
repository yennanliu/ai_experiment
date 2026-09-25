<!-- generated:start -->
# 16-multi-agent-and-swarms / 21-agent-economies

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/21-agent-economies/) · upstream spec
`phases/16-multi-agent-and-swarms/21-agent-economies/docs/en.md`

```bash
uv run demo practice run 21-agent-economies --ex 1
uv run demo explain 21-agent-economies --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/21-agent-economies
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Confirm Shapley values sum to total value (efficiency axiom). Change the… | code | T0 | `ex01_the_fair_split_is_one_the_best_pair_would_walk_away_from.py` |
| 2 | Implement Shapley *sampling* (Monte Carlo over K orderings). How does K affect approximation… | code | T0 | `ex02_sampling_four_agents_costs_more_than_computing_them_exactly.py` |
| 3 | Implement a coalition-forming step before the auction: agents can merge into teams and bid as… | code | T0 | `ex03_every_profitable_coalition_is_a_bidding_ring.py` |
| 4 | Read the Google Research mechanism-design post. Identify one assumption that, if violated, br… | explain | T0 | prose, below |
| 5 | Read the AAMAS 2025 decentralized LaMAS paper. Implement their Shapley step over 10 agents on… | code | T0 | `ex05_a_hundred_samples_cost_more_calls_than_exact_shapley_over_ten_agents.py` |
<!-- generated:end -->

## Answers

### 1 — the fair split is one the best pair would walk away from

**Efficiency holds, and every change moves in the predicted direction.** The
exact values are coder **0.5083**, researcher **0.3333** and reviewer
**0.1583**, which sum to 1.0000 = v(grand).

"The expected direction" can be made exact. Raising v(S) by d moves each
member of S up by d·(|S|−1)!(n−|S|)!/n!, and each non-member down by
d·|S|!(n−|S|−1)!/n!. The reference matches this for all 7 coalitions.

The second term is the one that surprises. Improving {coder, researcher} by
0.1 lowers the reviewer's credit by **0.0333**, twice what each pair member
gains, even though the reviewer did nothing differently. The demo's
200-sample estimate lands within 0.0035 of exact, after 800 value calls where
exact enumeration made 24.

**The Shapley split is outside the core.** Coder and researcher receive
0.8417 together, but v({coder, researcher}) = 0.85, so the pair does better
without the reviewer. The demo's comment calls the game superadditive, and it
is. It is not convex, though: the reviewer adds 0.20 to {coder} and only 0.15
to {coder, researcher}. That is what lets a coalition block the fair split.

Three differences between the lesson and its code:

- **Exact vs sampled.** The docstring says Shapley is "exact for N<=6 and
  sampled otherwise". Nothing chooses between the two, and `shapley_exact`
  enumerates N! orders for any N.
- **Names.** The lesson lists `shapley()`, `second_price_auction()` and a
  Reputation class "with exponential decay and slashing". The code has
  `shapley_exact` and `second_price`, and Reputation has no slashing method.
- **Reputation routing.** The demo prints **+3.6%** against a promised
  10–20%. Over 1000 seeds the mean gain is 3.5%, and only 5% of seeds fall
  in the band. Half the tasks are random warmup, and alpha = 0.95 keeps every
  score near its initial 1.0. Routing to the best agent after warmup would
  gain 23.5%.

### 2 — sampling four agents costs more than computing them exactly

**Error falls as 1/√K, but at N=4 no value of K is worth paying for.** The
test game has 4 agents with weights 4, 3, 2, 1 and v(S) = (Σw)²/100. Error is
the mean over 200 seeds of the worst agent's error:

| K | calls | mean worst-agent error |
|---:|---:|---:|
| 10 | 50 | 0.0604 |
| 24 | 120 | 0.0402 |
| 100 | 500 | 0.0197 |
| 1000 | 5000 | 0.0063 |

The ratio between K=100 and K=1000 is 3.1, against √10 = 3.16. Each sample
costs N+1 value calls, so K=24 already costs what `shapley_exact` does, 120
calls, and still misses by 0.04.

Exact computation needs even less than that. There are only 2⁴ = 16
coalitions. The subset form of Shapley reads each one once and matches the
reference to 1e-12, which beats sampling at K=4 (20 calls). Also, drawing 24
orders *with* replacement sees 15.3 distinct orders on average, while the
same 24 draws *without* replacement give the exact answer.

**The efficiency check from exercise 1 cannot see sampling error.** Every
sampled allocation sums to v(grand) to within 1e-14, because each ordering
telescopes to v(grand) − v(∅). An estimate that is 0.06 off passes it.

### 3 — every profitable coalition is a bidding ring

A team bids as a unit with its best member's value, since the slot is one
task done by one agent. Each of the 26 possible teams of two or more is run
through `second_price` against everyone outside it.

**Only teams that include both agent-c and agent-a form.** Just 7 teams
change the outcome, and all 7 contain the winner (0.95) and the runner-up
(0.82). What the winner saves is 0.82 minus the best bid left outside the
team:

| team | payment | saved |
|---|---:|---:|
| none | 0.82 | — |
| {a, c} | 0.77 | 0.05 |
| {a, c, e} | 0.60 | 0.22 |
| {a, b, c, e} | 0.45 | 0.37 |

**This is not a Pareto improvement.** Agent-c's value wins the slot in all 26
runs, so total welfare stays 0.95. What the team saves is exactly the revenue
the auctioneer loses: the team gains, the auctioneer loses, and nobody else
changes.

If all five bid as one team, the auction switches off: `second_price` returns
None when there are fewer than two bids. The mechanism has no reserve price,
so its lowest possible revenue is whatever bid the team leaves outside.

The lesson's own fair-credit tool then pays the losers. `shapley_exact`
splits the four-member team's 0.37 as follows: agent-c and agent-a 0.1192
each, agent-e 0.0942, agent-b 0.0375. The runner-up, who could never win,
gets the same share as the winner. The only way merging could be Pareto-better
is if it created value, and bidding as a unit for a single slot cannot.

### 4 — the guarantee covers the bid, and the LLM controls the distribution

*Draws on "Second-price auction for aggregation", checked against the Google Research post it cites.*

The post's mechanism differs from what the lesson describes. Each agent
submits a *distribution over the next token* plus a bid. The aggregator
combines the distributions token by token, for example with a bid-weighted
average. Payments are an analogue of second-price payments, which exist for
any monotone aggregation function. The post implements them through "stable
sampling": for each seed, the output is one of two tokens depending on
whether the bid is below or above a threshold.

Monotone means: "if an agent weakly increases their bid, the aggregated
distribution function would only change to a distribution weakly preferred by
the agent." The lesson's version, "value depends on which proposal is chosen,
not how many were bid", is a different condition. Neither the lesson nor its
code (a plain Vickrey auction over scalars) aggregates anything.

**The assumption that breaks truthfulness: that agents report their
distributions honestly.** The post says so directly: agents "truthfully report
their distributions, but may be strategic about their bids". Incentive
compatibility is proven over bids, given "known partial preference orders over
distributions".

In an LLM setting, the distribution is exactly what the agent controls.
Picture an advertiser's agent whose model would honestly give its brand 30%
of the next-token mass. It can report 100% instead: sharpen the distribution
with temperature 0 or a system prompt that repeats the brand name. At the
same bid, the averaged distribution then moves further toward it. Bidding
truthfully is still optimal *given* the report, but the report is now a lever
the mechanism never priced. You would see it as creatives that read like one
sponsor's copy while that sponsor paid for a modest share.

The lesson's collusion failure is separate. Exercise 3's bidding ring leaves
every bid truthful and still costs the auctioneer 0.37 of 0.82.

### 5 — a hundred samples cost more calls than exact Shapley over ten agents

The paper has no Shapley step to implement. It is Yang et al., "Unlocking the
Potential of Decentralized LLM-based MAS", a 5-page paper in the AAMAS 2025
Blue Sky Ideas track. Shapley appears once: "Attribution methods, such as the
Shapley Value, ensure profits are allocated based on each agent's
contribution", with a citation to Shapley 1953. The paper also mentions
blockchain smart contracts for payouts. It gives no algorithm or value
function and contains no auction, and the text has no DIDs, which the lesson
says it combines.

So the step implemented here is the textbook one, on a task whose exact
answer has a closed form. Ten agents cover 20 skills, and v(S) is the
fraction of skills covered. Agent i's Shapley value is then the sum, over the
skills it holds, of 1/(20·k), where k is how many agents hold that skill.

| method | value calls at N=10 | time | error |
|---|---:|---:|---:|
| `shapley_exact` (all orders) | 39,916,800 | ~20 s (from 0.02 s at N=7) | 1e-12 at N=7 |
| subset form | 1,024 | 2 ms | 1e-12 |
| `shapley_sampled`, 100 draws | 1,100 | — | 0.0133 on values averaging 0.08 |

100 draws cost more calls than the exact answer. They pick the right top
agent on all 50 seeds, so the ranking survives even though the amounts are
17% off. The lesson's advice to "Shapley-sample, not Shapley-exact" only
holds once 2^N outgrows the sample budget: around N=14 for 1,000 draws.

**The coverage game rewards a Sybil.** Clone the agent with the most shared
skills (agent-5) as an eleventh agent run by the same operator. The
operator's combined Shapley value rises by **42%** and v does not change. The
extra credit is taken from the honest agents who shared those skills.
Symmetry, the axiom that makes Shapley fair, pays out for duplicates. The
lesson names Sybil attacks as a failure mode, and its own credit rule is how
they succeed.
