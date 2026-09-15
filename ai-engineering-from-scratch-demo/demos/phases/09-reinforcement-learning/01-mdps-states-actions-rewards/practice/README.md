<!-- generated:start -->
# 09-reinforcement-learning / 01-mdps-states-actions-rewards

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/09-reinforcement-learning/01-mdps-states-actions-rewards/) · upstream spec
`phases/09-reinforcement-learning/01-mdps-states-actions-rewards/docs/en.md`

```bash
uv run demo practice run 01-mdps-states-actions-rewards --ex 1
uv run demo explain 01-mdps-states-actions-rewards --ex 1
uv run pytest demos/phases/09-reinforcement-learning/01-mdps-states-actions-rewards
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Implement the 4×4 GridWorld and random-policy rollout in `code/main.py`. Run 10,000 epi… | code | T0 | `ex01_the_mean_stops_at_two_hundred_steps.py` |
| 2 | Medium. Run `policy_evaluation` with `γ ∈ {0.5, 0.9, 0.99}` for the uniform-random policy. Pr… | code | T0 | `ex02_growth_is_slowest_beside_the_terminal.py` |
| 3 | Hard. Turn the GridWorld stochastic: each action slips to an adjacent direction with probabil… | code | T0 | `ex03_a_uniform_policy_cannot_feel_the_slip.py` |
<!-- generated:end -->

## Answers

The lesson is one 4×4 GridWorld and five functions of pure stdlib, and all three
exercises turn out to ask about the same thing from three directions: what the
discount, the episode cap and the action distribution each do to a value that
the MDP already determines exactly. The **Easy** exercise asks for a mean that
converges to the wrong number. The **Medium** one asks you to explain an effect
that runs the other way. The **Hard** one asks "better or worse" about the one
policy on the board for which the answer is neither.

All three are **T0** — the lesson imports nothing but `random`, and so does this
pack.

### 1 — the average is of a censored episode, not of the MDP

Environment re-derived from the five lines the lesson page prints, checked
against the lesson's own `step` on all 64 (state, action) pairs, then 10,000
episodes through the reference's own `rollout`:

| | value |
|---|---:|
| mean return, 10,000 episodes | **−58.16** |
| standard deviation | 46.0 |
| median | −44 |
| optimal, by value iteration | **−6** |

**ANSWER: −58.2 ± 46.0, which is 9.7× the optimal −6.** The −6 is not taken from
the exercise on trust: value iteration over the lesson's own `step` returns it.

**FINDING: that mean is not converging to `E[G]`.** `rollout` carries
`max_steps=200`, which censors the episode, so the estimator has two candidate
limits and it converges to the wrong one:

| | exact | distance from the measured mean |
|---|---:|---:|
| censored at 200 steps | **−58.3204** | 0.4 standard errors |
| the MDP's own `V(0,0)`, γ→1 | **−59.4286** = −416/7 | 2.8 standard errors |

The 1.11 between them is bias, not noise. Ten thousand episodes buy a standard
error of 0.45; the gap is 2.4 of them, and a larger sample shrinks the noise and
leaves the bias exactly where it is.

**FINDING: 2.5% of episodes hit the cap and carry all of it.** A censored
episode is recorded as −200 when its true return is unbounded below — which is
also why the sign of the bias is *up*, toward the optimum, in an experiment whose
stated purpose is to show how far the random policy sits from it.

**FINDING: Step 2's stated range holds neither value.** The lesson says the
random policy averages "around −60 to −80 for this 4×4 board". The two numbers
this MDP can produce are −59.43 and −58.32; both are above the top of that
window.

**FINDING: mean and std are a symmetric summary of an asymmetric shape.** Median
−44 against a mean of −58, hard-clipped at −200 by the cap and at −6 by the
geometry. The ±1σ band `[−104, −12]` spans 47% of the entire achievable range.

### 2 — growth is slowest beside the terminal

`V` for the uniform-random policy at the three discounts, from the lesson's own
evaluation loop:

```text
γ = 0.5                              γ = 0.9                         γ = 0.99
 -2.000 -1.999 -1.996 -1.992    -9.361 -9.219 -8.975 -8.750   -39.412 -38.188 -36.249 -34.645
 -1.999 -1.995 -1.983 -1.953    -9.219 -8.974 -8.500 -7.970   -38.188 -36.405 -33.338 -30.400
 -1.996 -1.983 -1.920 -1.696    -8.975 -8.500 -7.416 -5.757   -36.249 -33.338 -27.603 -20.407
 -1.992 -1.953 -1.696  0.000    -8.750 -7.970 -5.757  0.000   -34.645 -30.400 -20.407   0.000
```

**ANSWER: every value grows toward a ceiling of `1/(1-γ)` — 2, 10, 100.** The
start state reaches 100%, 94% and 39% of that ceiling at the three discounts: the
ceiling rises 50×, and the share of it actually reached falls.

**FINDING: the premise is backwards, and monotonically.** Grouping states by
Manhattan distance to the terminal and taking the magnitude ratio between γ=0.5
and γ=0.99:

| ring (distance to terminal) | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| growth, γ 0.5 → 0.99 | **12.0×** | 15.2× | 17.1× | 18.2× | 19.1× | **19.7×** |

It increases with distance at every step. The two states adjacent to the terminal
are the *slowest*-growing on the board and the start corner the fastest, a factor
of 1.64 the other way from the one the exercise asks you to explain.

**MECHANISM: raising γ can only restore return that a shorter horizon cut off,
and a state beside the terminal has the least of it.** In closed form, with a
constant −1 per step and hitting time `T`:

```text
V(s) = -(1 - E[γ^T]) / (1 - γ)
```

so `(1-γ)|V(s)|` is the fraction of the ceiling a state reaches — 20% beside the
terminal against 39% at the start, at γ=0.99. Growth is then the 50× ceiling rise
scaled by the ratio of those fractions: `50 × 0.204/0.848 = 12.0` and
`50 × 0.394/1.000 = 19.7`, which is the measured table to within 0.2.

**FINDING: γ=0.5 collapses the grid instead of ranking it.** An effective horizon
of 2 steps on a board 6 deep leaves all 15 non-terminal states within 0.30 of one
another on a ceiling of 2, and 13 of them within 0.08. The grid the exercise asks
you to print cannot order most of the states it shows.

**CONTROL: the shipped evaluator is not the one the lesson prints.** Step 3's
snippet writes `V[s] = v` and reads it back inside the same sweep — Gauss-Seidel.
`code/main.py` builds `new_values` and updates synchronously — Jacobi. Same fixed
point to 1.3e-11, different cost:

| γ | shipped (synchronous) | printed (in place) |
|---|---:|---:|
| 0.5 | 21 sweeps | 17 |
| 0.9 | 113 | 76 |
| 0.99 | 467 | **305** |

In-place versus synchronous is exactly the distinction the *next* lesson's Key
Terms table teaches, with "converges faster in practice" beside it.

### 3 — a uniform policy cannot feel the slip

**ANSWER: neither. `V[start]` does not move.** −39.4116480543 deterministic,
−39.4116480543 with `p = 0.1` slip — bit for bit, and no state on the board
moves at all. This is not a small change; it is no change.

**MECHANISM: the slip matrix is doubly stochastic and the uniform policy is its
fixed row.** A slip that does not depend on the state is a stochastic matrix `K`
on the four directions, applied to the action before the environment sees it — so
an agent running `π` under slip is an agent running `π·K` without it, and the
lesson's own `policy_evaluation` evaluates it unchanged. Every direction receives
`1-p` from itself and `p/2` from each of the two it is perpendicular to, so every
column of `K` sums to 1, and

```text
(π·K)(a') = Σ_a (1/4) · K(a'|a) = 1/4
```

exactly. The slipped uniform policy *is* the uniform policy. Same chain, same
Bellman system, same fixed point.

**ROBUSTNESS: nothing depends on `p`, or on how "adjacent" is read.** Unchanged
at `p` = 0.1, 0.3, 0.5, 0.9 and 1.0 — an actuator that *always* slips included.
Unchanged under the looser reading where the action slips to any of the three
other directions. Any symmetric slip is doubly stochastic, so the ambiguity the
exercise leaves open cannot change the answer.

**CONTROL: the board really did become stochastic.** The same slip, applied to
policies that have a preference to be pushed off:

| policy | deterministic | with slip |
|---|---:|---:|
| uniform random | −39.4116 | **−39.4116** |
| the lesson's greedy down+right | −7.589 | −8.362 (−10.2%) |
| optimal (value iteration, γ=0.99) | −5.852 | −6.428 |

**FINDING: every policy on this board answers the question except the one the
exercise names.** A policy that prefers a direction can be pushed off it; a
policy with no preference has nothing to be pushed off. "Does `V[start]` get
better or worse?" is a good question asked of the single policy in its own
lesson for which it has no answer.
