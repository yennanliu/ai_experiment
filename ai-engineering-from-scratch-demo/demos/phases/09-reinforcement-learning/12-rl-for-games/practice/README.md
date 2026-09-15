<!-- generated:start -->
# 09-reinforcement-learning / 12-rl-for-games

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/09-reinforcement-learning/12-rl-for-games/) · upstream spec
`phases/09-reinforcement-learning/12-rl-for-games/docs/en.md`

```bash
uv run demo practice run 12-rl-for-games --ex 1
uv run demo explain 12-rl-for-games --ex 1
uv run pytest demos/phases/09-reinforcement-learning/12-rl-for-games
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Implement the GRPO bandit in `code/main.py`. Train on 2 prompts × 4 answer tokens each.… | code | T0 | `ex01_three_quarters_of_the_updates_do_nothing.py` |
| 2 | Medium. Plug in PPO (clipped) and vanilla REINFORCE. Compare sample efficiency and reward var… | code | T0 | `ex02_grpo_loses_at_every_sample_budget.py` |
| 3 | Hard. Extend to a length-2 "reasoning chain": the agent emits two tokens and the verifier rew… | code | T0 | `ex03_the_harder_task_is_the_one_grpo_learns_faster.py` |
<!-- generated:end -->

## Answers

All three exercises are about GRPO's one moving part: the group. It has no learned
critic, so its baseline is the mean reward of the `G=8` samples it just drew, and its
advantage is their standardised spread. When the eight samples agree the spread is
zero, every advantage is `0 / 1e-8 = 0`, and the update does nothing. Exercise 1
measures how often that happens, exercise 2 shows what GRPO pays for the eight samples
it needs to have an opinion at all, and exercise 3 finds the group-agreement rate
explaining a result that looks backwards — the harder task is the one GRPO learns
faster.

All three are **T0**: the whole lesson runs in under five seconds.

Accuracy throughout is read out of the policy — the softmax probability of the correct
answer, averaged over prompts — rather than from the lesson's 200-episode `evaluate`.
The policy is a softmax over four answers, so the probability of being right is
available exactly, and sampling it 200 times only adds noise to a number that is
already known.

### 1 — it converges, and three quarters of the updates do nothing

The exercise asks for convergence in under 1,000 updates with `G=8`. It converges, on
every seed:

| updates | mean P(correct), 10 seeds | worst seed |
|---:|---:|---:|
| 100 | 0.6825 | |
| 250 | 0.9075 | |
| 500 | 0.9536 | |
| 1,000 | **0.9692** | 0.9660 |

The cost is in the second table. Over 2,000 updates, **67.2%** of them produce exactly
zero gradient, and the rate climbs as the policy improves:

| updates | degenerate groups |
|---|---:|
| 1–500 | 39.8% |
| 501–1,000 | 75.2% |
| 1,001–1,500 | 75.6% |
| 1,501–2,000 | 78.2% |

GRPO's signal is group *disagreement*, and success is what destroys it. At the start
the degenerate groups are all-wrong; by the end they are all-right. That is also why
the curve flattens rather than finishes — accuracy moves 0.9075 → 0.9536 over 250
updates and 0.9536 → 0.9692 over the next 500, half the rate for twice the budget. The
last 0.031 is bought by the quarter of updates that still see a mixed group.

One discrepancy: the exercise says "2 prompts × 4 answer tokens", but `QUESTIONS`
carries three entries — `what is 1+2`, `what is 3*3` and `capital of France` — so the
shipped bandit is 3 × 4. Every number here is for the bandit the lesson actually ships.

### 2 — GRPO loses at every sample budget, once you count samples

"Sample efficiency" has two readings and they disagree. GRPO draws 8 samples per
update and REINFORCE draws one, so comparing them per *update* hands GRPO eight times
the data and calls the result efficiency. Per *sample drawn from the policy*:

| samples | REINFORCE | GRPO |
|---:|---:|---:|
| 400 | **0.893** | 0.460 |
| 1,600 | **0.983** | 0.875 |
| 4,000 | **0.994** | 0.954 |
| 16,000 | **0.999** | 0.970 |

Per update — the axis the lesson's own `main` prints — the ordering flips, twice:

| updates | GRPO | REINFORCE |
|---:|---:|---:|
| 500 | **0.954** | 0.923 |
| 2,000 | 0.970 | **0.987** |

One axis has GRPO ahead early and behind late; the other has it behind throughout.

PPO is added as the clipped ratio on the same group and the same advantage, so the
three differ only in how the advantage is formed. **Its clip never fires.** On one pass
over a fresh group the ratio is exactly 1 — the group was drawn from the policy being
updated — and the only thing that can move it is a step of `lr/G = 0.0125`, nowhere
near the `[0.8, 1.2]` band. The clip rate is 0.00% at one epoch and 0.08% at four. PPO
on this bandit is GRPO with extra arithmetic.

Reward variance is both what GRPO removes and what it needs. Its per-update variance
falls from 0.174 over the first 200 updates to 0.022 over the last 200, which is
exercise 1's 67.2% seen from the other side. REINFORCE's holds at 0.069 — but `verify`
returns 0 or 1 and `reinforce_step` uses the raw reward as the advantage, so a wrong
answer gives a gradient of exactly zero, on **10.1%** of updates. REINFORCE here can
only reward, never punish. That is the mirror of lesson 06, where every return was
negative and the same update could only punish.

### 3 — the harder task is the one GRPO learns faster

Following the hint exactly — one advantage per sequence, applied unchanged at both
token positions — GRPO handles the length-2 chain, and handles it better than the
single-token bandit it was compared against:

| updates | length-2 chain (1/16 random) | length-1 bandit (1/4 random) |
|---:|---:|---:|
| 500 | 0.9121 | |
| 2,000 | **0.9889** | 0.9701 |
| 5,000 | 0.9961 | |

The harder task ends closer to solved. To check whether sequence-level credit is the
bottleneck the exercise implies, the run is repeated with a verifier that scores each
position separately — dense per-token credit reaches 0.9895 against 0.9889 at the same
budget, **a difference of 0.0005**. The credit assignment is not where the difficulty
is.

The mechanism is exercise 1's. At a 1/16 random success rate the group keeps
disagreeing far longer: degenerate updates run **23.5%** over the first 500 against
39.8% for the 1/4 task, so more of the budget carries signal. The thing that makes the
task hard is the same thing that keeps GRPO's gradient alive.

And the advantage does reach the right token without being told which one it is. The
two positions are independent in the policy and a sequence is correct only when both
are, so a single sequence-level advantage pushes both positions of a correct pair up
and both of a wrong pair down — the right direction for each, even though neither
position is scored on its own. Accuracy is already 0.9121 at 500 updates, from a start
of one correct pair in sixteen.
