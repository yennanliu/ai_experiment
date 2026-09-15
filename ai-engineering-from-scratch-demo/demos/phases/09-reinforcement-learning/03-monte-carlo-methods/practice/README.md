<!-- generated:start -->
# 09-reinforcement-learning / 03-monte-carlo-methods

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/09-reinforcement-learning/03-monte-carlo-methods/) · upstream spec
`phases/09-reinforcement-learning/03-monte-carlo-methods/docs/en.md`

```bash
uv run demo practice run 03-monte-carlo-methods --ex 1
uv run demo explain 03-monte-carlo-methods --ex 1
uv run pytest demos/phases/09-reinforcement-learning/03-monte-carlo-methods
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Implement first-visit MC evaluation of the uniform-random policy on 4×4 GridWorld. Run… | code | T0 | `ex01_the_plot_converges_to_the_wrong_line.py` |
| 2 | Medium. Implement ε-greedy MC control with `ε ∈ {0.01, 0.1, 0.3}`. Compare mean return after… | code | T0 | `ex02_the_metric_prices_exploring_not_learning.py` |
| 3 | Hard. Implement *off-policy* MC with importance sampling: collect data under uniform-random p… | code | T0 | `ex03_weighted_is_has_no_variance_and_no_answer.py` |
<!-- generated:end -->

## Answers

Three exercises, and each one specifies a measurement that cannot see the thing it
asks about. Exercise 1 plots a curve converging to a line it does not draw.
Exercise 2 compares a number fixed before any learning happens. Exercise 3 ranks
estimators by variance, and the winner's variance is zero for a reason that has
nothing to do with the estimator.

The board here is the **deterministic** 4×4 GridWorld — this lesson's `step` has
no slip, unlike lesson 02's. Where a DP answer is needed it is re-derived by
sweeping this lesson's own `step`, never carried over from another lesson's
printed number.

All three are **T0**: the lesson imports `random` and `collections`, and so does
this pack.

### 1 — the plot converges to a line the exercise does not draw

`V(0,0)` accumulated exactly as `mc_policy_evaluation` accumulates it, from the
lesson's own `rollout` and `returns_from`, seed 20260914:

| episodes | 100 | 500 | 1,000 | 2,000 | 5,000 | 10,000 |
|---|---:|---:|---:|---:|---:|---:|
| `V(0,0)` | −39.23 | −41.05 | −40.47 | −39.98 | −39.37 | **−39.56** |

**ANSWER: −39.556 at 10,000 episodes against a DP −39.4116** — 0.7 standard
errors, at `se = 0.216`.

**FINDING: the line it converges to is not the DP answer.** `rollout` carries
`max_steps=200`, so the estimator's limit is the 200-step censored value:

| | value |
|---|---:|
| the MDP's `V^uniform(0,0)` | **−39.4116** |
| the 200-step censored value, which MC estimates | **−39.3123** |
| bias | 0.0993 |
| episodes that reach the cap | 2.3% |

**FINDING: at 10,000 episodes that bias is invisible, and it is meant to be
seen.** 0.0993 is 0.46 standard errors at the sample size the exercise specifies.
Noise falls as `1/√n`; bias does not. They are equal at ~47,000 episodes — five
times the run — so the plot cannot show the one thing that is wrong with it.

**FINDING: what the plot does show is its own noise.** `σ = 21.61` on a mean of
−39.4. Reading `V(0,0)` to ±0.1 needs 46,682 episodes; the 10,000 asked for buy
±0.216. The −39.23 at n=100 and the −41.05 at n=500 are that σ, not learning.

**FINDING: the exact answer is 21× cheaper and is not a sample.** MC spends
589,413 environment steps to land inside a standard error. Sweeping the lesson's
own `step` to `1e-6` costs 467 sweeps — 28,020 backups — and is exact. MC is the
method for when `step` is not available; here it is, three lines above.

### 2 — the metric prices the exploring, not the learning

The lesson's own `mc_control`, three `ε`, one seed, 20,000 episodes each:

| ε | mean return, all 20,000 | mean return, last 5,000 | exact ceiling for that ε |
|---|---:|---:|---:|
| 0.01 | −6.038 | −5.930 | **−5.9041** |
| 0.1 | −6.585 | −6.561 | **−6.4208** |
| 0.3 | −8.060 | −7.990 | **−7.9906** |

**ANSWER: −6.04, −6.59, −8.06, monotone in `ε`.** The whole-run and last-5,000
means agree to within 0.108, so the curve the exercise asks about is flat after
the first few thousand episodes.

**FINDING: that ordering is arithmetic on `ε`, available before the first
episode.** The last column is the exact on-policy value of an `ε`-soft policy over
the optimal action, computed by sweeping the lesson's own `step`. Each run lands
within 0.14 of its own ceiling. "Compare mean return" compares the price of
exploring.

**FINDING: all three learn the same thing.** The greedy policy each run recovers
is worth **exactly −5.8520** from the start state — the optimum, to 0.0e+00, at
every `ε`. The number the exercise asks to compare is the one that does not
separate the three runs.

**FINDING: the tradeoff lives off the path the metric walks.**

| ε | states worth < `V*` under the recovered policy | of those, states whose own action is suboptimal | worst gap | (s,a) pairs ever tried |
|---|---:|---:|---:|---:|
| 0.01 | 4 | 3 | **97.03** | 59 / 60 |
| 0.1 | 4 | 3 | 3.86 | 60 / 60 |
| 0.3 | **0** | 0 | 0.00 | 60 / 60 |

The two counts are different questions and they give different answers. A state can
choose an optimal action and still be worth less than `V*`, because the loss is
inherited from a successor that chooses badly — which is what the fourth state in
each of the first two rows is doing.

At `ε = 0.01` the top-right corner `(0,3)` is assigned `right` — into the wall,
forever. That state is worth **−100.0** under the learned policy against an
optimal −2.97. Every episode starts at (0,0) and the recovered path never visits
the corner, so the stated metric reports −5.93 for a policy with a non-terminating
state in it.

**CONTROL: exploration is a threshold here, not a dial.** Run the same code at
`ε = 0`:

| | pairs tried | `V(0,0)` of the recovered policy | mean return |
|---|---:|---:|---:|
| ε = 0 | **6 / 60** | **−100.0** | −86.6 |
| ε = 0.01 | 59 / 60 | −5.8520 | −6.04 |

From 0 to 0.01 the outcome moves 94 points. From 0.01 to 0.3 it moves 2.1, and all
of that second move is cost rather than capability. The interval the exercise
sweeps is entirely on the flat side of the threshold.

### 3 — weighted IS has no variance, and 45% of the time no answer

Off-policy from `μ` = uniform to `π` = the deterministic optimal policy. All three
estimators are accumulated from the *same* trajectories, so the comparison
isolates the estimator: 20 independent batches of 4,000 episodes, target
`V^π(0,0) = −5.851985` known exactly.

| estimator | std across batches | mean | range |
|---|---:|---:|---|
| plain IS | 5.224 | −4.794 | [−16.8, **0.0000**] |
| per-decision IS | 1.220 | −5.812 | [−8.05, −3.96] |
| weighted IS | **0.0000** | **−5.851985** | one value, 9/20 undefined |

**ANSWER: weighted IS — and not by a margin, by a kind.** Its standard deviation
is not small, it is zero.

**FINDING: that zero is degeneracy, not quality.** `π` is deterministic and so is
the board, so exactly one trajectory has nonzero weight: the 6-step optimal path,
`ρ = 4⁶ = 4096`. Every batch that sees it sees the *same* return. Weighted IS is
averaging one distinct number, and all 11 defined estimates agree to 1e-12 and
equal the truth to 0.0e+00. Nothing here transfers to a stochastic `π` or a
stochastic board.

**FINDING: the price is having no answer at all.** 9 of 20 batches contained no
matching trajectory — the denominator is 0 and the estimator is undefined —
against a predicted `(1 − 4⁻⁶)^4000 = 0.377`. Only 0.020% of trajectories match
`π` end to end. "Which has lowest variance" is ranking an estimator that declines
to answer nearly half the time, and the ranking covers only the surviving batches.

**FINDING: plain IS's worst failure is not noise, it is a confident 0.** Its
maximum across the 20 batches is exactly `0.0000`, on the same 9 batches where
weighted IS abstains — an estimate outside `[−100, −5.85]`, the range of every
return this MDP can produce. Its spread is 89% of the quantity being estimated,
and those empty batches are most of it. Weighted IS abstains; plain IS reports.

**MECHANISM: per-decision IS's 4.3× reduction comes from weighting rewards, not
trajectories.**

```text
plain:         ρ_{0:T} · Σ_t γ^t r_t          one weight, applied to everything
per-decision:  Σ_t γ^t · ρ_{0:t} · r_t        a weight per reward
```

`ρ_{0:t}` is `4^(t+1)` only while the prefix still matches, so a trajectory that
follows `π` for two steps and then leaves still contributes its first two rewards.

| | trajectories that contribute something |
|---|---:|
| plain IS | 0.020% |
| per-decision IS | **25.2%** |

The full 4096× weight is only ever applied to the last reward of a complete match.
That buys 4.3× less spread and a range that never includes 0.
