<!-- generated:start -->
# 09-reinforcement-learning / 06-policy-gradients-reinforce

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/09-reinforcement-learning/06-policy-gradients-reinforce/) · upstream spec
`phases/09-reinforcement-learning/06-policy-gradients-reinforce/docs/en.md`

```bash
uv run demo practice run 06-policy-gradients-reinforce --ex 1
uv run demo explain 06-policy-gradients-reinforce --ex 1
uv run pytest demos/phases/09-reinforcement-learning/06-policy-gradients-reinforce
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Implement REINFORCE on 4×4 GridWorld with a linear softmax policy. Train for 1,000 epis… | code | T1 | `ex01_a_third_of_runs_never_reach_the_terminal.py` |
| 2 | Medium. Add a running-mean baseline. Train again. Compare sample efficiency and variance to t… | code | T1 | `ex02_the_baseline_does_not_reduce_steps_it_changes_who_survives.py` |
| 3 | Hard. Add an entropy bonus `β · H(π)`. Sweep `β ∈ {0, 0.01, 0.1, 1.0}`. Plot final return and… | code | T1 | `ex03_the_mean_and_the_median_move_opposite_ways.py` |
<!-- generated:end -->

## Answers

This is the lesson where the algorithm itself fails, not just the measurement.
**A third of REINFORCE runs on this board never reach the terminal at all** — they
end frozen on a policy that walks in circles until `rollout`'s 100-step cap ends
the episode. All three exercises are written around that fact, because none of
their questions has a stable answer without it.

All three are **T1** — 60 seeds of 1,000 episodes each, about 13 s, 21 s and 39 s.
Everything runs on the lesson's own `reinforce`, `rollout`, `returns_to_go`,
`init_theta` and `softmax`; exercise 3 adds the one piece of mathematics the
exercise asks for and nothing else.

### 1 — a third of the runs never reach the terminal

60 seeds of the exercise's own setting: 1,000 episodes, no baseline.

| outcome | seeds | final return | std within a run |
|---|---:|---:|---:|
| converged | 40 | median **−5.92** | 0.51 |
| collapsed | 20 (**33.3%**) | **−63.3968** | **0.0** (18 of 20) |

**ANSWER: about −5.92 on the runs that converge — and a third of them do not.**

**FINDING: the collapsed runs sit on a closed-form value, not near one.**
−63.3968 is exactly `−(1 − 0.99¹⁰⁰)/(1 − 0.99)`, the discounted return of a
100-step episode that never terminates. 18 of the 20 match it **to one ULP on
every one of their last 200 episodes**; the other 2 still finish occasionally and
land within 0.099. The policy has not become poor — it has stopped finishing.

**FINDING: the "std of returns" the exercise asks for is a mixture.** Exactly
**0.0** inside a frozen run, **0.51** inside a converged one, **27.0** across
seeds. The third number is the distance between the two outcomes, not the variance
of either. One number cannot describe a bimodal result, and the exercise asks for
one number.

**MECHANISM: with no baseline and every reward negative, REINFORCE only punishes.**
**100%** of returns-to-go are strictly negative, so `adv < 0` at every update. The
gradient is

```text
∂/∂θ_i  =  (1[i = a] − p_i) · adv
```

so for the action actually taken, `(1 − p_a) > 0` times `adv < 0` **lowers** its
probability, and for the other three `(0 − p_i) < 0` times `adv < 0` **raises**
theirs. At every step of every episode. Nothing is ever reinforced — only made
less likely than its alternatives, and a run that drifts onto a non-terminating
policy has no force pulling it back.

**FINDING: the curve is discounted and the target it is read against is not.**
`returns_log` stores `returns[0]`, a γ = 0.99 return, while the lesson's `main`
prints "optimal return on this 4x4 GridWorld = −6.0". The discounted optimum is
**−5.8520**; −6.0 is the undiscounted one. A perfectly converged run is *expected*
to sit slightly above the number it is being compared against.

### 2 — the baseline does not reduce steps; it changes which runs survive

60 **paired** seeds, one definition of convergence applied to both arms: the first
episode after which 50 consecutive returns stay above −10. Runs that never
converge are counted, not dropped — dropping them is what makes the baseline look
like a speed-up.

| | collapsed | converged | median episodes to converge | mean final |
|---|---:|---:|---:|---:|
| vanilla | 20/60 (33.3%) | 40 | **137.5** | −25.14 |
| + baseline | **13/60 (21.7%)** | 47 | **141.0** | −18.38 |

**ANSWER: it does not reduce them — it adds 3.5.** The exercise asks by how much
the baseline reduces steps to convergence. On the runs that converge at all, it
does not.

**FINDING: what it changes is who survives.** Collapse rate 33.3% → 21.7%, and the
mean final return improves from −25.14 to −18.38. That improvement is *entirely* a
change in the mix of outcomes; no run got faster.

**FINDING: and it is not free.** The outcome flips on **13 of 60** seeds: 10
rescued, and **3 that converged without the baseline collapse with it**. A net 7
seeds better, at the cost of 3 the vanilla arm had already solved.

**MECHANISM: on a collapsed run the baseline annihilates its own signal.** The
running mean tracks `returns[0]`, so once a run is capped every return is
−63.3968 and the baseline converges to that same value. Replaying the lesson's own
`b ← 0.95b + 0.05·returns[0]` recursion over a collapsed run's returns, the final
`|G − baseline|` is **1.07e-13**. The gradient carries `adv`, so it vanishes and
the policy is frozen in place. Vanilla keeps `adv = −63.4` at every update.

So the baseline **never escapes a collapse** — the 10 seeds it rescues are ones
where it stopped the collapse happening in the first place, and the 3 it breaks are
ones where it caused one.

**FINDING: the baseline is the start-state return, subtracted at every timestep.**
It is fitted to `G` at `s₀` — about −5.9 on a converged run — and the same number
is subtracted from the return-to-go one step before the terminal, which is −1.0. A
factor of 6 apart. A state-independent baseline stays unbiased, so this is not
*wrong*; it simply reduces no variance where the two disagree. Mean within-run std
on converged runs is 0.51 vanilla against 0.49 with it.

### 3 — the mean and the median move opposite ways

The entropy gradient is derived, not approximated. For a softmax:

```text
∂H/∂z_i  =  −p_i (log p_i + H)
```

added to the lesson's own policy-gradient term; the update is otherwise
`reinforce` line for line. The inner feature loop is skipped where `x[j] == 0`,
which is **exact** because the features are one-hot — verified against the
unskipped loop at **0.0e+00**.

| β | mean final | median final | collapsed | start-state entropy |
|---:|---:|---:|---:|---:|
| 0 | −25.18 | **−5.95** | 20/60 (33.3%) | 0.019 |
| 0.01 | −28.02 | −5.99 | 23/60 (38.3%) | 0.008 |
| 0.1 | −23.69 | −6.01 | 18/60 (30.0%) | 0.023 |
| 1.0 | **−21.87** | **−6.89** | 16/60 (26.7%) | 0.124 |

**ANSWER: there is no sweet spot — there is a choice of statistic.** The mean
improves by 3.31 from β=0 to β=1.0 while the median falls by 0.94, monotonically.
Which of the two you plot decides where the "sweet spot" appears to be.

**FINDING: the mean is tracking the collapse rate, and that is inside its error
bar.** The collapse column is **not monotone** — β=0.01 is the worst of the four —
and at 60 seeds one arm's standard error is about ±6.1 points, so every step in
that row is inside it. The mean moves because it is a mixture whose weights are
noise.

**FINDING: the median is the one thing in the sweep that moves cleanly.** −5.95 →
−6.89, monotonically, on the runs that actually converge. That is the entropy
bonus doing exactly what it advertises: holding the policy away from the
deterministic optimum, and being paid for in return.

**FINDING: the entropy axis never leaves the floor.** 0.019, 0.008, 0.023, 0.124
against a uniform `ln 4 = 1.3863`. Even at β=1.0 the policy sits at **9%** of
uniform; for the first three β it is under 2%. The sweep does not span a
meaningful range of the quantity the exercise puts on one of its two axes.

**FINDING: the bonus is orders of magnitude under what it opposes.** The
policy-gradient term carries `adv`, between −1 and −63 here. The entropy term
carries `β·(−p_i)(log p_i + H)`, bounded near `0.37β` — about **0.004** at
β=0.01. That is why that column is indistinguishable from β=0, and why β would
have to be far above 1.0 before the entropy axis moved at all.
