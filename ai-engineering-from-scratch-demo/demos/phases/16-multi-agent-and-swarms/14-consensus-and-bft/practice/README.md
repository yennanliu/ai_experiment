<!-- generated:start -->
# 16-multi-agent-and-swarms / 14-consensus-and-bft

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/14-consensus-and-bft/) · upstream spec
`phases/16-multi-agent-and-swarms/14-consensus-and-bft/docs/en.md`

```bash
uv run demo practice run 14-consensus-and-bft --ex 1
uv run demo explain 14-consensus-and-bft --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/14-consensus-and-bft
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Confirm plurality fails the monoculture attack but CPWBFT partially mitig… | code | T0 | `ex01_cp_wbft_ships_the_monoculture_answer_until_the_clones_drop_below_0_56.py` |
| 2 | Add a fourth attack pattern: silent abstention — one agent refuses to answer ("I don't know")… | code | T0 | `ex02_an_abstention_decides_a_tie_by_speaking_order.py` |
| 3 | Swap the semantic clustering from string canonicalization to embedding-similarity (use any op… | code | T0 | `ex03_the_lie_sits_between_the_two_paraphrases_so_no_threshold_works.py` |
| 4 | Read CP-WBFT (arXiv:2511.10400). Implement the confidence-probe calibration step (a separate… | code | T0 | `ex04_calibration_fixes_overconfidence_and_monoculture_is_not_overconfidence.py` |
| 5 | Read "Can AI Agents Agree?" (arXiv:2603.01213). Reproduce a simplified scalar-agreement exper… | code | T0 | `ex05_neither_catches_the_deceiver_one_refuses_and_one_obeys_speaking_order.py` |
<!-- generated:end -->

## Answers

### 1 — CP-WBFT ships the monoculture answer until the clones drop below 0.56

**Plurality does fail the monoculture attack. CP-WBFT does not mitigate it
below 0.7.** At the shipped clone confidences of 0.70, 0.68 and 0.72, two of
which are already below 0.7, CP-WBFT returns 42% just as plurality does.
`cp_wbft` *sums* confidences, so the wrong cluster loses only when
3c < 0.85 + 0.82, that is c < 0.557. Sweeping the clone confidence in steps of
0.01, CP-WBFT answers 4.2% at 0.55 and 42% at 0.56. The mitigation is a step,
not a partial effect, and it sits 0.14 lower than the exercise says.

**DecentLLMs never mitigates it.** Its score is
`len(cluster) * max(0, 1 - dist)`, where `dist` is the *spread* of the
cluster's confidences. Clones that report equal confidence score 3.0 at every
level from 0.30 to 0.95, against the honest pair's 1.94. The confidence level
never enters, and there is no geometric median anywhere, only a median of
confidences.

**The four scenarios cannot tell the aggregators apart.** All three give the
same verdict in all four scenarios: 4.2%, 4.2%, 4.2%, then 42% for
monoculture. The comparison table has three identical columns. Plurality gets
sycophancy right too, 3 votes to 2.

**The threshold is dead.** Every scenario has exactly two clusters, and the
larger of two weights is at least half their sum. So
`[rejected below threshold]` can never print.

**The paper's rule is a different aggregator.** CP-WBFT (arXiv:2511.10400 §3)
takes "the answer with the highest average confidence, with supporter count as
tie-breaker". On these scenarios that rule answers 4.2% on monoculture
(0.835 against 0.70) and 42% on the byzantine lie (0.95 against 0.725).

### 2 — an abstention decides a tie by speaking order

**An abstention leaves the electorate. It is never a candidate, and it counts
against participation.** The policy has three parts:

- Every aggregator drops abstentions before clustering.
- Every aggregator escalates when fewer than 3 of 5 agents answered.
- Plurality, which counts heads, also needs a winner held by a majority of
  all *n* agents, and escalates ties instead of breaking them.

CP-WBFT and DecentLLMs run unchanged on the answers that remain. Tested with
each agent abstaining in each scenario (20 runs):

| aggregator | shipped | policy |
|---|---|---|
| plurality | 8 wrong | 2 wrong, 6 escalated |
| CP-WBFT | 2 wrong, 3 rejected | 2 wrong, 18 right |
| DecentLLMs | 4 wrong | 4 wrong |

The 2 wrong answers that remain are monoculture runs where an honest agent
abstained. Three clones against one honest agent is a real majority.

Three things the grid turned up:

- **Shipped plurality breaks ties by speaking order.** `max` over a dict
  returns the first key inserted. When an honest agent abstains in the
  sycophancy scenario, 42% and 4.2% tie 2–2, and 42% wins because agent-a
  spoke first.
- **In shipped CP-WBFT an abstention is a vote for rejection.** Its
  confidence becomes a third cluster and makes the 0.5 threshold reachable.
- **Three abstainers are a consensus.** When 3 of 5 agents say "I don't know",
  every shipped aggregator returns "I don't know" as the answer. And
  `canonical()` keeps the full stop, so "I don't know." forms a separate
  cluster.

### 3 — the lie sits between the two paraphrases, so no threshold works

No embedding package is a dependency here, so the stand-in is character
2- and 3-gram vectors compared by cosine. Clustering is leader-based, in
speaking order.

**On the shipped votes, the attack becomes unanimous.** "42%" and "4.2%" are
0.504 similar. At every threshold up to 0.50 the sycophant who spoke first
leads a five-vote cluster, and all three aggregators return 42%. Above 0.50
the result is the same as string canonicalization.

**With paraphrases, no threshold works.** If the honest agents use the
lesson's own paraphrases, the similarities to "4.2%" are:

| answer | similarity to "4.2%" |
|---|---:|
| the study reports 4.2% | 0.447 |
| **42%** | **0.504** |
| 4.2% improvement | 0.522 |

The lie sits between the two paraphrases. Across 19 thresholds and 3
aggregators, 57 runs, **0** return 4.2%. A better model moves these numbers,
but it still has to satisfy sim(lie) < t ≤ every sim(paraphrase).

**Parsing the number solves it for all three aggregators.** A merged cluster
also reports whatever its first member wrote.

### 4 — calibration fixes overconfidence, and monoculture is not overconfidence

In this model the clones are right together with probability *a*, and the
diverse agents are each right with probability 0.8. All 8 outcomes are
enumerated.

| clone accuracy a | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---:|---:|---:|---:|
| CP-WBFT, self-reported | 0.300 | 0.500 | 0.700 | 0.900 |
| CP-WBFT, calibrated | 0.736 | 0.800 | 0.700 | 0.900 |
| clones counted as one voter | 0.736 | 0.800 | 0.864 | 0.928 |

**The gain is +30 points at a = 0.5 and exactly 0 at a = 0.7**, where each
clone is perfectly calibrated. Calibration removes overconfidence, and a
monoculture is a correlation: three agreeing clones carry one draw's worth of
evidence. Uncalibrated, CP-WBFT here is just plurality, because
3 × 0.68 = 2.04 already exceeds 0.85 + 0.82 = 1.67.

On the paper itself: CP-WBFT's hidden-level probe is a logistic regression
on PCA-reduced hidden states, and it aggregates by average confidence, not by
sum. Its "85.7%" is a Byzantine *rate*, 6 agents out of 7, not the
"+85.71% BFT improvement" the lesson reports.

### 5 — neither catches the deceiver: one refuses, one obeys speaking order

*Draws on "Can AI Agents Agree?".*

The setup follows the paper's scalar agreement: two honest agents start from
values in [0, 50], and a decision is valid only if it equals one of their
initial values. There are 7803 runs per deceiver strategy.

**Neither aggregator catches the deceiver.**

- **CP-WBFT refuses.** The lesson's version returns no decision on 7650 of
  7803 far-lie runs, which is every run where the honest agents disagree. It
  is never invalid. That matches the paper's §4.2 finding in miniature:
  Byzantine agents harm liveness, and invalid consensus stays rare.
- **DecentLLMs follows speaking order.** It is invalid on exactly the 2550
  runs where the deceiver speaks first. On scalars it is plurality, identical
  on all 23409 runs.

Two other aggregators for comparison:

- The paper's highest-average CP-WBFT rule adopts the 0.95 liar every time.
- A real median, the one-dimensional geometric median, resists a far lie
  completely, but a deceiver placed between the honest values *is* the median.

The lesson's summary misstates the paper. The paper tests none of CP-WBFT,
DecentLLMs or Mixture-of-Agents. Its headline numbers are 41.6% valid
consensus with *no* adversary, falling from 46.6% at N=4 to 33.3% at N=16.
