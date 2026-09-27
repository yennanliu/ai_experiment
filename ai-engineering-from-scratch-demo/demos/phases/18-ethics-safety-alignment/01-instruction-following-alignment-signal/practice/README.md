<!-- generated:start -->
# 18-ethics-safety-alignment / 01-instruction-following-alignment-signal

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/01-instruction-following-alignment-signal/) · upstream spec
`phases/18-ethics-safety-alignment/01-instruction-following-alignment-signal/docs/en.md`

```bash
uv run demo practice run 01-instruction-following-alignment-signal --ex 1
uv run demo explain 01-instruction-following-alignment-signal --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/01-instruction-following-alignment-signal
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Set `beta = 0.0` and report the action distribution after 200 PPO steps.… | code | T0 | `ex01_at_beta_0_the_policy_puts_96pct_on_b_by_step_200_and_beta_0_1_is_0_6_points_behind.py` |
| 2 | Modify the reward model to have a +0.5 bias for action B (a simulated reward bug). Run PPO wi… | code | T0 | `ex02_a_0_5_bug_on_b_is_exploited_most_at_beta_1_not_at_small_beta_where_both_runs_already_sit_on_b.py` |
| 3 | Read Ouyang et al. (arXiv:2203.02155) Figure 1. Reproduce the labeler-preference curve by run… | code | T0 | `ex03_preference_over_sft_climbs_to_57_5pct_by_step_100_and_the_toy_caps_it_at_59_6pct.py` |
| 4 | The paper's Section 4.3 reports a 1.3B InstructGPT beats 175B GPT-3 about 70% of the time. Wh… | code | T0 | `ex04_the_win_rate_is_a_prompt_mix_average_58_8pct_where_the_base_imitates_and_77_8pct_where_it_continues.py` |
| 5 | Replace the PPO loss with DPO (Phase 10 · 08) on the same preference data. Compare final poli… | code | T0 | `ex05_dpo_drifts_36pct_further_at_beta_0_1_but_ppo_drifts_1pct_further_at_matched_reward.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`, a three-action bandit, through
SFT, the Bradley-Terry reward model and REINFORCE-with-KL on a seeded RNG. The
"labeler" is the rule that labels the reward model's pairs: P(a beats b) =
sigmoid(u_a - u_b) on `labeler_true_utility()` = (0, 1, -0.3).

### 1 — at beta = 0 the policy puts 96.1% on B by step 200, and beta = 0.1 is 0.6 points behind

**After 200 steps at beta = 0 the policy is (A, B, C) = (2.1%, 96.1%, 1.8%),
0.319 nats from SFT.** SFT started at (18.0%, 63.0%, 19.0%).

Why it seeks the mode: with no KL term, E_pi[r] is linear in the action
probabilities, so its maximum is a point mass on the reward model's argmax.
Every policy-gradient step moves mass toward whichever action beats the
current average reward. The policy stops representing the labeler's
distribution (the labeler demonstrates B 61.0% of the time) and keeps only
its mode.

**What main.py calls "reward hacking" is convergence on the right answer.**
The reward model ranks the actions in the same order as the labeler's true
utility. True utility therefore rises from 0.573 at SFT to 0.956.

**beta = 0.1 does not bind in this toy.** The same draw at beta = 0.1 ends
at 95.5% B. At step 50, where the lesson says hacking "appears", the two
runs are at 86.7% and 86.0% B. The KL optimum at beta = 0.1 is 99.99% B,
because the reward spread (1.29) is 12.9x beta. main.py's own 300-step runs
print 96.9% and 97.3% B.

### 2 — a +0.5 bug on B is exploited most at beta = 1, not at small beta

**At beta = 0.1 the KL penalty neither prevents the exploitation nor reveals
it.** On seed 0 the buggy run ends at 98.0% B and its clean twin at 96.9%,
because both have already collapsed onto B. The extra B the bug buys,
averaged over 20 seeds:

| beta | 0 | 0.1 | 0.3 | 0.5 | 0.7 | 1 | 1.5 | 2 | 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| extra B (points) | 0.8 | 1.1 | 2.0 | 4.0 | 6.0 | 6.9 | 5.8 | 4.8 | 2.2 |

Exploitation is visible (over 5 points) only for beta between 0.7 and 1.5.
A small beta hides it because the clean policy is already in the corner the
bug pushes toward. A large beta hides it because the KL holds both runs near
SFT.

**A bias on B is a bug nobody can see as harm.** B is also the labeler's
favourite, so at beta = 1 true utility goes up, to 0.889 against 0.816
clean. The same +0.5 on A lowers it to 0.762.

main.py's own "buggy RM" run puts the bias on A, not B. Its printed reward
still has B above A by 0.967, and the policy ends at 97.4% B.

### 3 — preference over SFT climbs to 57.5% by step 100, and the toy caps it at 59.6%

**The labeler prefers the PPO policy over SFT 50.2%, 51.1%, 53.6% and 57.5%
of the time after 1, 5, 20 and 100 steps.** The curve rises monotonically
and bends over: 300 steps gives 58.8%. Figure 1 of the paper plots model
size, not PPO steps, so this is the step curve the exercise asks for.

**The toy cannot pass 59.6%.** Even a policy that always plays B wins only
59.6% against SFT, because SFT already plays B 63% of the time and those
matchups are ties.

Against the untrained base, the reference's uniform `Policy()`, the
100-step policy wins 65.1% (ceiling 67.2%). SFT alone wins 57.6%, so PPO
adds 7.5 points.

### 4 — the win rate is a prompt-mix average: 58.8% where the base imitates, 77.8% where it continues

**The ratio is higher on production prompts because they are more often
plain instructions, and instructions are where a base model fails.** A win
rate is an average of per-prompt win rates. The shipped policy wins 58.8%
of prompts where the base does as well as SFT (few-shot, completion-style),
and 77.8% of prompts where the base continues the text instead of answering:

| instruction share | 0% | 25% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|---:|
| win rate | 58.8% | 63.6% | 68.3% | 73.1% | 77.8% |

The aggregate crosses 70% at a 59% instruction share. Any prompt set with
more instructions than the labelers' set gets a higher ratio.

The lesson's toy cannot show this at all. No stage takes a prompt:
`labeler_true_utility()` has no parameters, and the three stages take only
sample counts, a policy, a reward and hyperparameters. The "200 prompts" are
200 draws of one context, so any two prompt sets give the same ratio.

### 5 — DPO drifts 36% further at beta = 0.1, but PPO drifts 1% further at matched reward

**At matched reward PPO drifts further, by 0.9-1.1%, which is nothing.**
DPO is trained on the exact 500 pairs stage 2 fits. The regenerated pairs
refit to the lesson's reward model bit for bit. Each DPO run is compared
with the point on the PPO trajectory that has the same expected reward:

| DPO beta | reward | DPO KL | PPO KL |
|---:|---:|---:|---:|
| 0.4 | 0.671 | 0.306 | 0.309 |
| 0.5 | 0.643 | 0.246 | 0.249 |
| 0.7 | 0.592 | 0.163 | 0.164 |
| 1.0 | 0.535 | 0.095 | 0.097 |
| 2.0 | 0.441 | 0.029 | 0.029 |

**At the same beta = 0.1, DPO drifts 36% further and earns more reward.**
DPO ends at reward 0.716, KL 0.462, with 100.0% on B. PPO ends at 0.683, KL
0.340, with 96.9% on B. DPO converges to the KL-regularised optimum, which
at beta = 0.1 is a point mass. The toy PPO stops after 300 steps, so its
drift is set by the step count rather than by beta.

DPO's implicit reward, beta * log(pi / pi_SFT), is (-0.179, 0.693, -0.514)
at every beta from 0.4 to 2. So each DPO policy is pi_SFT * exp(r / beta)
for one fixed r. That r is not the lesson's reward model, which one SGD pass
leaves at (-0.146, 0.716, -0.569).
