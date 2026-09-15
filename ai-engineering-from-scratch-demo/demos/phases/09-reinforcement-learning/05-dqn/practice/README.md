<!-- generated:start -->
# 09-reinforcement-learning / 05-dqn

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/09-reinforcement-learning/05-dqn/) · upstream spec
`phases/09-reinforcement-learning/05-dqn/docs/en.md`

```bash
uv run demo practice run 05-dqn --ex 1
uv run demo explain 05-dqn --ex 1
uv run pytest demos/phases/09-reinforcement-learning/05-dqn
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. Plot the per-episode return curve. How many episodes until the runn… | code | T1 | `ex01_the_running_mean_never_crosses_it.py` |
| 2 | Medium. Disable the target network (use the online net for both sides of the Bellman target).… | code | T1 | `ex02_neither_and_the_target_net_was_never_stale.py` |
| 3 | Hard. Add Double DQN: use the online net to pick `argmax a'`, target net to evaluate. Compare… | code | T1 | `ex03_double_dqn_halves_the_bias_it_does_not_remove_it.py` |
<!-- generated:end -->

## Answers

This is the first lesson in the phase whose exercises cost real time: `code/main.py`
trains a 16-32-4 network by hand-rolled backprop for 400 episodes, and the pack
runs that loop nineteen times. All three exercises are **T1** rather than T0 for
that reason — ex1 takes about 9 seconds, ex2 about 35, and ex3 a little over four
minutes. Nothing here needs a GPU or a key; it is simply pure-Python arithmetic at
volume.

Two of the three exercises expect an instability that does not happen, and the
third expects a fix that only half works. All the numbers below come from the
lesson's own primitives — `init_net`, `clone`, `epsilon_greedy`, `train_step`,
`forward`, `step` — with `main`'s loop rebuilt around them, because `main` returns
nothing a test can read.

### 1 — the running mean never crosses it, and four other readings do

"Running mean" is not defined anywhere in the lesson. On one run of the shipped
configuration at its own seed:

| reading of "running mean" | first episode above −10 |
|---|---:|
| **cumulative from episode 0** | **never** — ends at −10.64 |
| trailing 10 | 115 |
| trailing 50 | 154 |
| trailing 100 | 179 |
| the lesson's own 50-episode blocks | 200 |

**ANSWER: five readings, four different numbers, and one that does not exist.**

**FINDING: the first 50 episodes sink the cumulative mean for the whole run.** They
average −28.06 against a −6 optimum that no later episode can beat. At the
converged rate of −6.34 it would take about 70 *more* episodes — a 470-episode run
— for the cumulative mean to climb over the line.

**MECHANISM: the crossing episode measures the `ε` schedule, not the learning.**
`ε = max(0.05, 1 − ep/200)`. Sweeping the lesson's own `step` for the exact
50-step-capped return of an `ε`-soft policy over the **optimal** policy — a curve
with no learning in it at all — that ideal crosses −10 on a trailing-50 window at
**episode 142**, against the measured **154**. An agent that was already optimal at
episode 0 answers this exercise with nearly the same number.

**FINDING: the network is finished long before the curve is.**
`max_a Q(0,0,a) = −5.8731` against `V*(0,0) = −5.8520` — a gap of 0.021. The blocks
from episode 150 onward still read −6.84, −6.28, −6.32, −6.20, −6.34, and that
movement is `ε` being switched off, not the value function still arriving.

**FINDING: the 50-step cap is what makes the early episodes that heavy.** The inner
loop stops at 50 steps, so the worst episode is exactly −50. Most of the curve's
visible −6-to−50 range is a truncation constant.

### 2 — neither, and the target net was never stale enough to matter

Two arms, two seeds each. Disabling the target network means passing `online` where
`train_step` expects `target`; nothing else changes.

| arm | last-100 mean / sd | `max Q(0,0)` vs `V* = −5.8520` |
|---|---:|---:|
| with target, seed 0 | −6.27 / 0.69 | −5.8731 |
| with target, seed 1 | −6.34 / 0.78 | −5.7475 |
| **without**, seed 0 | −6.32 / 0.89 | **−5.8504** |
| **without**, seed 1 | −6.30 / 0.77 | **−5.8506** |

**ANSWER: neither oscillation nor divergence.** The two arms differ by **0.005**
while the two seeds *inside* one arm differ by 0.070. The worst tail episode
anywhere is −11, against a −50 floor.

**FINDING: by value accuracy the no-target arm is the better of the two.** Worst
error **0.0016** without the target network against **0.1044** with it — a factor
of 65. The component the exercise asks you to remove was not buying accuracy
either.

**MECHANISM: the target network was never stale.** It is refreshed every 200 steps,
21 times over the run. The sup-norm gap between online and target *just before* a
refresh averages **0.639** and peaks at 1.215, on a `Q` scale of 5.9 — about 11%.
Deleting a perturbation that size cannot destabilise a bootstrap it was only
shifting by that much.

**FINDING: the deadly triad has no shared approximator to interfere through.**
`state_features` is one-hot over 16 states, so each state owns its own column of
`W1` and the first layer generalises between states not at all. The net carries
**676 parameters for the 64 `Q` values** it fits — 10.6× more. A target network
damps off-policy bootstrapping through a *shared* approximator; this one barely
shares.

**FINDING: what looks like oscillation is `ε = 0.05` and nothing else.** A 6-step
optimal episode takes at least one random action `1 − 0.95⁶ = 26.5%` of the time,
which is the whole of the −6-to−11 band both arms occupy after convergence.

### 3 — Double DQN halves the bias; it does not remove it

Three paired seeds and a `σ = 0` control, at the lesson's own 32-hidden, batch-32
configuration. Only the target *computation* is new — the rewritten minibatch goes
through the lesson's own `train_step`.

| | seed 0 | seed 1 | seed 2 | mean | last-50 return |
|---|---:|---:|---:|---:|---:|
| DQN | +1.61 | +0.80 | +3.17 | **+1.86** | −8.1 |
| Double DQN | +0.46 | +0.84 | +1.82 | **+1.04** | −6.3 |
| difference | +1.15 | **−0.03** | +1.35 | +0.82 ± 0.35 | |

**ANSWER: the exercise's first half holds and its second half does not.** DQN
overestimates `V*(0,0) = −5.8520` on every seed. But Double DQN overestimates on
every seed too — it is **1.8× smaller, not absent**. Decoupling selection from
evaluation shrinks the maximization bias; it does not cancel it.

**FINDING: one seed of three reverses the comparison.** Seed 1's paired difference
is −0.03. The mean is +0.82 ± 0.35, 2.3 standard errors from zero. "Compare bias
with vs without" run once, as the phrasing invites, has a one-in-three chance of
showing nothing.

**FINDING: the better estimate buys a better policy, which the exercise does not
ask about.** Last-50 return is −6.3 under Double DQN against −8.1 under DQN, on a
−6 optimum. That 1.7-point gap is **twice** the 0.82 value gap Double DQN is
usually justified by — the number the agent *earns* moves further than the number
it reports.

**CONTROL: at `σ = 0` the difference disappears.** With the noise switched off and
everything else identical, the two biases are **−0.021 and −0.030 — 0.009 apart**,
against +0.82 with noise. Both arms carry a target network throughout, and
exercise 2 measured that deleting it on the clean board costs nothing. So what the
reward noise defeats is neither the architecture nor the target net: it is the
`max` over four noisy estimates.

#### Why this one runs at full scale

`DESIGN D11` prefers a scaled-down runnable for expensive exercises. Two were tried
and both **inverted the sign of the effect**:

| configuration | DQN mean bias | Double DQN mean bias |
|---|---:|---:|
| 32 hidden / batch 32 (shipped) | **+1.86** | **+1.04** |
| 16 hidden / batch 32 | −1.06 | +0.15 |
| 16 hidden / batch 8 | −0.13 | −18.54 |

Batch size is the term that averages the reward noise, so shrinking it does not
scale the phenomenon down — it replaces the phenomenon with gradient noise. At
batch 8 both arms are destroyed and the comparison is meaningless. The effect
exists only at the shipped size, so the exercise runs there, and the four minutes
is the honest price.
