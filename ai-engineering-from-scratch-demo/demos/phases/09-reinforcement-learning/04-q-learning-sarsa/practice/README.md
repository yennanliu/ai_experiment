<!-- generated:start -->
# 09-reinforcement-learning / 04-q-learning-sarsa

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/09-reinforcement-learning/04-q-learning-sarsa/) · upstream spec
`phases/09-reinforcement-learning/04-q-learning-sarsa/docs/en.md`

```bash
uv run demo practice run 04-q-learning-sarsa --ex 1
uv run demo explain 04-q-learning-sarsa --ex 1
uv run pytest demos/phases/09-reinforcement-learning/04-q-learning-sarsa
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Implement Q-learning and SARSA on the 4×4 GridWorld. Plot learning curves (mean return… | code | T0 | `ex01_neither_and_the_curve_is_not_theirs.py` |
| 2 | Medium. Build a cliff-walking environment (4×12, last row is the cliff with reward -100 and r… | code | T0 | `ex02_q_learning_walks_the_edge_and_pays_for_it.py` |
| 3 | Hard. Implement Double Q-learning. On a noisy-reward GridWorld (Gaussian noise σ=5 added to p… | code | T0 | `ex03_the_two_biases_cancel_at_the_state_named.py` |
<!-- generated:end -->

## Answers

Three exercises about the difference between on-policy and off-policy control, and
the first of them is on a board where that difference does not exist. Exercise 1
asks which of two algorithms converges faster and the honest answer is that they
are indistinguishable — to 0.7 standard errors, on twelve paired seeds. Exercise 2
adds a cliff, and the same difference becomes 29 points. Exercise 3 asks for a
demonstration that does not reproduce.

A thread runs through all three: **the quantity each exercise asks you to measure
is not the quantity it is asking about.** The learning curve measures the
behaviour policy. The cliff comparison's obvious metric ties. `max_a Q(0,0,a)`
sits on top of two large biases that cancel.

All three are **T0**: the lesson imports `random` and `collections`, and so does
this pack.

### 1 — neither, and the curve does not belong to either of them

Twelve paired seeds rather than the single one `main()` uses, because "who
converges faster" is a claim about a difference:

| | SARSA | Q-learning |
|---|---:|---:|
| episodes to reach the limit and stay | **300** (all 12 seeds) | **300** (all 12 seeds) |
| mean return, last 500 | −6.614 | −6.610 |
| paired difference | \-0.0045 ± 0.0067 | (0.7 σ from zero) |

**ANSWER: neither, measurably.**

**FINDING: the limit is `ε`'s, not either algorithm's.** The exact undiscounted
return of an `ε`-soft policy over the optimal action is **−6.6087** at `ε = 0.1`,
swept from the lesson's own `step`. Both curves settle within 0.005 of it. The
plot measures the behaviour policy the two share — and that policy is not learned,
it is specified.

**FINDING: both learn the optimal policy exactly, which the curve also cannot
show.** The greedy policy recovered on all 24 runs is worth **−5.8520** from the
start, the optimum to 0.0e+00. The curve's plateau is 0.76 worse; the difference
is exploration, which neither algorithm is being asked to stop doing.

**FINDING: the plotted quantity is not the optimized one.** `total += r` sums raw
rewards while every update discounts at `γ = 0.99`. Undiscounted `ε`-soft return
is −6.6087; discounted is −6.4208. Survivable here, and not on a board where the
discount changes the ranking.

**FINDING: the comparison needs a board this one is not.** Q-learning bootstraps
from `max_a Q(s',a)` and SARSA from the `a'` it will actually take. Here that
difference is invisible because no state punishes the `ε`-random action — the
worst an exploratory step costs is one wasted move.

### 2 — Q-learning walks the edge, and pays 29 points for it

Only the environment is built. `step` and `reset` are module-level names that the
lesson's own `sarsa` and `q_learning` look up at call time, so replacing those two
runs both learners unmodified on the new board. Seed 42, 3,000 episodes:

```text
Q-learning                       SARSA
><>>>>>v>>vv                     >>>>>>>>>>>V
>vv>>>>>>vvv                     ^>^>>>>>>>>V
>>>>>>>>>>>V                     ^^>^^^^^<^>V
SXXXXXXXXXXG                     SXXXXXXXXXXG
```

(upper-case = the greedy path, `X` = cliff.)

**ANSWER: Q-learning, and not marginally.** Scored by how many of the ten cells
directly above the cliff each greedy path walks through — which is the thing that
costs:

| | cliff-edge cells walked | path length |
|---|---:|---:|
| Q-learning | **10, 10, 10** (seeds 7, 42, 101) | 13, 13, 13 |
| SARSA | **0, 0, 0** | 15, 17, 17 |

**FINDING: the policy Q-learning learns is better and the policy it runs is much
worse.**

| | followed greedily | run at `ε = 0.1` | measured online |
|---|---:|---:|---:|
| Q-learning | **−13.00** (optimal) | −50.52 | −49.95 |
| SARSA | −16.33 | **−21.44** | −21.59 |

The first two columns are exact, swept over the new board; the third is the
lesson's own logged return over the last 500 episodes. They agree to within 0.57,
so the 29-point gap belongs to the policies rather than to three seeds.

**FINDING: the gap between the policy learned and the policy run is 7.4× wider
for Q-learning.** 37.52 against 5.10. The cliff-edge path is optimal only for an
agent that never explores, and the agent that learned it explores 10% of the time.
Three or four extra steps is the premium SARSA pays to close that gap.

**MECHANISM: SARSA's target contains the exploration and Q-learning's does not.**
`r + γQ(s',a')` averages over the `ε`-random `a'` that may step off the cliff;
`r + γ max_a Q(s',a)` prices the edge as if the agent never slipped. Neither is
wrong — they answer different questions, and only one of them is the question the
learning curve scores.

**FINDING: the obvious reading of the metric cannot separate them.** The deepest
row either path touches is 2, for both: SARSA's first move out of the start corner
is upward into row 2 at column 0, which is not above a cliff cell. Read as a
minimum distance the metric ties; read as how much of the edge is walked, it is 10
against 0.

### 3 — two biases of the same size cancel at the state the exercise names

"Show" is read as measuring over enough runs to support a claim — 20 paired seeds,
because one run of this cannot resolve a bias of any size. `V*(0,0)` is value
iteration over the clean `step`; the noise is zero-mean and does not move it.

| | bias in `max_a Q(0,0,a)` | per-run spread | significance |
|---|---:|---:|---:|
| Q-learning | **−0.162** | ±1.13 | 0.6 σ — indistinguishable from zero |
| Double Q-learning | **−2.038** | ±1.75 | 3.7 σ, and an *under*estimate |

**ANSWER: the claim does not reproduce, and its second half is backwards.** The
measurable bias belongs to the estimator the exercise says has none, and it points
the other way.

**MECHANISM: the maximization bias is real, and it is being cancelled.** The
measured −0.162 splits into two halves that are read off the learned estimates
themselves, and the split is an exact identity rather than an argument:

| | value |
|---|---:|
| `E[max_a Q(0,0,a)]` | −6.0141 |
| `max_a E[Q(0,0,a)]` | −7.4099 |
| **maximization premium** `E[max Q] − max E[Q]` | **+1.396** |
| `max_a Q*(0,0,a)` | −5.8520 |
| **mean estimation error** `max E[Q] − max Q*` | **−1.558** |
| sum — the measured bias | **−0.162** |

`Q*(0,0,·)` is `up −6.79, down −5.85, left −6.79, right −5.85` — two tied optima,
so the max really is taken over near-ties and really is biased upward. But every
entry underneath it is depressed by −1.56 to −2.24 (mean −1.94, spread 1.41). The
two halves nearly cancel, and `max_a Q` — the only quantity the exercise asks you
to look at — shows neither.

Reading the premium off the estimates matters. Drawing Gaussians at the measured
spread of 1.41 instead — the max operator isolated from all learning — puts it at
+1.09, understating the real thing by 28%, because the learned estimates are
neither independent nor equally dispersed across the four actions.

**FINDING: the depression is the noise, not the code and not the budget.** The
same code on the same 2,000 episodes at σ=0 gives per-action errors of `+0.01,
+0.00, +0.01, +0.00` — exact, so the implementation is right. The budget is then
checked where it matters rather than inferred from the clean run, since noise can
change the convergence rate: quadrupling it to 8,000 episodes at σ=5 moves
per-action error by at most 0.11 (`−2.14, −1.80, −2.17, −1.54`). So −1.94 is the
noise, not a finite-budget artifact.

**FINDING: one run cannot show any of this.** The per-run spread of
`max_a Q(0,0,a)` is 1.13 — wider than Q-learning's measured bias, comparable to
the maximization premium the mechanism supplies, and to Double Q's. "Show … by a
meaningful amount" is asked of a measurement whose own error bar is larger than
every effect in the experiment.

**FINDING: Double Q's underestimate is the price it was designed to charge.** Each
table sees half the updates and `QB[argmax QA]` is deliberately not a maximum, so
the estimator is pessimistic by construction. Here that costs 2.04 against the
1.40 premium it exists to remove — an over-correction of 1.5×. Trading a +1.40
bias for a −2.04 one is a trade; the exercise frames it as a fix.
