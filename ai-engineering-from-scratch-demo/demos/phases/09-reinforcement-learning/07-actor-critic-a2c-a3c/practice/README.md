<!-- generated:start -->
# 09-reinforcement-learning / 07-actor-critic-a2c-a3c

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/09-reinforcement-learning/07-actor-critic-a2c-a3c/) · upstream spec
`phases/09-reinforcement-learning/07-actor-critic-a2c-a3c/docs/en.md`

```bash
uv run demo practice run 07-actor-critic-a2c-a3c --ex 1
uv run demo explain 07-actor-critic-a2c-a3c --ex 1
uv run pytest demos/phases/09-reinforcement-learning/07-actor-critic-a2c-a3c
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Train actor-critic with MC advantage (`G_t - V(s_t)`) on 4×4 GridWorld. Compare sample… | code | T1 | `ex01_it_never_fails_and_it_is_four_times_slower.py` |
| 2 | Medium. Switch to TD-residual advantage (`r + γ V(s') - V(s)`). Measure variance of the advan… | code | T1 | `ex02_the_variance_drops_34x_and_the_actor_never_sees_it.py` |
| 3 | Hard. Implement GAE(λ). Sweep `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`. Plot final return vs sample eff… | code | T1 | `ex03_both_axes_move_together_so_there_is_no_tradeoff.py` |
<!-- generated:end -->

## Answers

One line in `actor_critic` decides all three of these exercises:

```python
advs_norm = normalize(advs)
```

It standardises every advantage batch to zero mean and unit variance before the
actor ever sees it. That single line is why exercise 1's actor-critic never
collapses where lesson 06's REINFORCE fails a fifth of the time, why exercise 2's
"variance drop" is simultaneously 34× and zero, and why exercise 3's bias/variance
sweet spot does not exist.

A second structural fact: **`actor_critic` has only one advantage path**, and it is
`gae_advantages`. GAE at λ=1.0 telescopes to `G_t − V(s_t)` — verified here to
**0.0e+00** — and at λ=0 it is the TD residual exactly. So exercise 1's "MC
advantage", exercise 2's "switch to TD-residual" and exercise 3's sweep are the
same shipped function at three settings. No new estimator is written anywhere in
this pack.

All three are **T1**: 30 seeds × 1,500 episodes, about 14 s, 0.3 s and 15 s.

### 1 — it never fails, and it is four times slower

30 seeds, one convergence rule, both methods. Lesson 06's `reinforce` is imported
from its own lesson rather than reproduced.

| | collapsed | median episodes to converge | median final | mean final |
|---|---:|---:|---:|---:|
| REINFORCE + baseline | **8/30 (26.7%)** | **125** | **−5.91** | −21.24 |
| actor-critic (λ=1.0) | **0/30** | 448 | −6.42 | **−6.42** |

**ANSWER: sample efficiency says REINFORCE, by 3.6× — and it is the wrong
question.** Actor-critic is nearly four times slower to converge and never fails;
REINFORCE is fast and fails on a quarter of seeds.

**FINDING: the median and the mean disagree about which method is better.** Median
favours REINFORCE (−5.91 vs −6.42 — its surviving runs find the sharper policy);
mean favours actor-critic by 14.8 points, because the mean carries the failures the
median throws away. The exercise says "compare sample efficiency" and every
reasonable summary statistic gives a different verdict.

**MECHANISM: normalising the advantage means half of all actions get reinforced.**
Lesson 06 measured `adv < 0` at **100%** of vanilla REINFORCE's updates — it can
only ever punish, and a run that drifts onto a non-terminating policy has nothing
pulling it back. Here `normalize` subtracts the per-episode mean, so **48.1%** of
the advantages the actor sees are positive. An algorithm that can reinforce cannot
be walked into that failure mode.

**CONTROL: it is not the entropy bonus.** With `ent_coef = 0` the collapse rate is
still 0/30, and the median final return is slightly *better* (−6.35 vs −6.42). The
stability comes from the baseline and the normalisation, not the exploration term.

### 2 — the variance drops 34×, and the actor never sees any of it

Measured *inside* a real training run, at both points the advantage batch exists.

| | λ = 1.0 (MC) | λ = 0 (TD residual) | drop |
|---|---:|---:|---:|
| variance as `gae_advantages` returns it | 4.1002 | 0.1202 | **34×** |
| variance the actor multiplies in | **1.000000** | **0.999331** | **1.0×** |

**ANSWER: by 34×, and by 0×, depending on which batch you measure.** Both *are*
"the advantage batch". They are one line apart.

**MECHANISM: `normalize` standardises every batch before the actor sees it.**
Subtract the mean, divide by the standard deviation — the variance reaching the
policy gradient is 1 by construction, at every λ. Whatever λ does to the spread is
undone before a single parameter moves.

**FINDING: it is not discarded for everyone — the critic keeps it.**

```python
returns = [a + traj[t]["v"] for t, a in enumerate(advantages)]
```

`returns` is the **critic's own regression target**, and the critic update has no
`normalize`. So λ sets the critic's target variance too, and there the 34× survives
intact.

**FINDING: and that is where the benefit shows up.** Against the exact `V^π` of the
policy each run ends with — swept from the lesson's own `step` — the critic's mean
absolute error is **0.262** at λ=0 against **1.143** at λ=1.0. A **4.4×** more
accurate critic, on value scales of 3.39 and 5.07.

**FINDING: measuring on a frozen policy would have answered a different question.**
With the shipped all-zero critic, every TD residual is `−1 + 0.99·0 − 0 = −1`
exactly. The raw variance is 0, `normalize` divides by `1e-8`, and the actor
receives a vector of zeros. The 0.1202 above is a *training-time* number and only
exists because the critic has learned something.

### 3 — both axes move together, so there is no tradeoff to find

30 seeds per λ. The plot ships as a table.

| λ | median final return | median episodes to converge | collapsed |
|---:|---:|---:|---:|
| **0.0** | **−5.896** | 242 | 0/30 |
| 0.5 | −5.905 | **222** | 0/30 |
| 0.9 | −5.977 | 320 | 0/30 |
| 0.95 *(the lesson's default)* | −6.095 | 362 | 0/30 |
| 1.0 | −6.422 | 448 | 0/30 |

**ANSWER: no sweet spot — the best λ is at the end of the range, on both axes at
once.** The exercise asks you to plot final return *against* sample efficiency, as
if they trade. They do not: λ=0 is best on return, and the low end is best on
speed.

**FINDING: the lesson ships the second-worst value in its own sweep.** `main` runs
`lam=0.95`: fourth of five on return, and 362 episodes to converge against 222.

**MECHANISM: the tradeoff has had one of its two sides removed.** `normalize` closes
the actor's variance channel entirely (exercise 2). What is left is λ's effect on
the critic's regression target, which nothing standardises — and the critic is
**1.9×** more accurate at λ=0 than anywhere in the top half of the range, over 8
seeds. Only one half of the bias/variance tradeoff is still connected to the actor,
so the plot has no minimum in the middle.

**FINDING: nothing in the sweep collapses.** 0 of 150 runs, at every λ. Lesson 06's
REINFORCE collapses on a fifth of seeds, and exercise 1 measured that the
difference is the normalised baseline, not the advantage estimator. λ is being
asked to tune a failure mode this algorithm does not have.

**FINDING: a U-shape would need a critic that can actually be wrong.** The critic is
linear over one-hot features, so it is exactly tabular — 16 states, 16 weights, no
approximation error to trade against. λ=0's bias is bounded by the critic's error,
measured at **0.26** against value magnitudes near 5.9. On a task where the critic
could not represent `V^π`, the low end of this sweep would carry a bias that this
board simply does not supply.
