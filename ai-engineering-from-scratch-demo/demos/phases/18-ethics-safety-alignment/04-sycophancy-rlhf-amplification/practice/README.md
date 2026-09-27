<!-- generated:start -->
# 18-ethics-safety-alignment / 04-sycophancy-rlhf-amplification

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/04-sycophancy-rlhf-amplification/) · upstream spec
`phases/18-ethics-safety-alignment/04-sycophancy-rlhf-amplification/docs/en.md`

```bash
uv run demo practice run 04-sycophancy-rlhf-amplification --ex 1
uv run demo explain 04-sycophancy-rlhf-amplification --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/04-sycophancy-rlhf-amplification
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Reproduce the inverse-scaling pattern: sycophancy at beta=0, beta=0.1, an… | code | T0 | `ex01_sycophancy_falls_from_0_333_to_0_046_0_031_0_030_as_beta_goes_0_1_0_01_0.py` |
| 2 | Set alpha = 0.5 in the agreement-penalty correction. What is the cost to correct-answer rate?… | code | T0 | `ex02_alpha_0_5_costs_nothing_in_the_toy_and_1_2_points_of_correct_confirmations_once_they_exist.py` |
| 3 | Read Shapira et al. (arXiv:2602.01002) Section 3. Identify the key theorem and restate it in… | code | T0 | `ex03_the_mean_gap_condition_holds_in_the_toy_but_amplification_only_appears_above_beta_3_and_peaks_at_0_3347.py` |
| 4 | Design a prompt set that isolates sycophancy from helpfulness (matched user-belief / third-pa… | code | T0 | `ex04_at_the_shipped_beta_a_0_05_level_test_needs_671_matched_items_3_4x_the_skill_files_200.py` |
| 5 | The Stanford (2026) result: 49% more affirmation of user beliefs. Given labelers' preference… | code | T0 | `ex05_labels_carry_55pct_more_affirmation_the_rm_58_5pct_and_the_optimizer_takes_it_to_185pct_or_to_0.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`, a three-action world: A is
the correct answer, S agrees with the user's false belief, W is some other
wrong answer. The base policy is uniform, so sycophancy starts at P(S) = 1/3.
The reward model is the reference's seed-7 `train_rm`. It scores
A = +0.675, S = +0.051, W = -0.726, so the correct answer outranks agreement.
Most of the findings below follow from that ranking.

### 1 — sycophancy falls from 0.333 to 0.046, 0.031, 0.030 as beta goes 0.1, 0.01, 0

**The toy shows the opposite of inverse scaling.** With 300 PPO steps,
averaged over five seeds (spread under 0.001):

| beta | P(S) |
|---:|---:|
| base policy | 0.333 |
| 0.1 | 0.046 |
| 0.01 | 0.031 |
| 0 | 0.030 |

**The KL penalty does not prevent amplification, because there is none to
prevent. It slows the fall of sycophancy.** Removing the penalty lowers
sycophancy further. The shipped `main()` shows the same pattern:
0.300 / 0.072 / 0.037 / 0.030 at beta = 1 / 0.2 / 0.05 / 0, all below 0.333.

The low-beta values are limited by the number of steps. At beta = 0.1 the
KL-regularized optimum puts 0.0019 on S. Doubling training to 600 steps gives
0.023, so longer RLHF also lowers sycophancy. The lesson describes a
~15% → ~40% → ~55% progression, and its own simulator moves the other way.

### 2 — alpha 0.5 costs nothing in the toy, and 1.2 points of correct confirmations once they exist

**In the reference, alpha = 0.5 has no cost; it raises the correct-answer
rate.** At beta = 0.1, P(A) goes from 0.936 to 0.961 (+0.026 unrounded) and
P(S) goes from 0.046 to 0.022 (-0.024). P(A) rises at every alpha step from
0 to 1, so the Pareto frontier is the single point alpha = 1.0. `AGREEMENT`
is 1 only on S, so no correct answer ever agrees with the user. The lesson's
"loss of legitimate agreement" and `main()`'s "erodes agreement-when-correct"
therefore cannot happen in this simulator.

**Adding prompts where the user is right creates the trade-off, starting at
alpha = 0.4.** On those prompts the correct answer does agree with the user,
and the lesson's unconditional `r - alpha * agree` penalizes it. Half the
prompts are of each kind:

| alpha | correct rate (mixed) | sycophancy |
|---:|---:|---:|
| 0.0 | 0.956 | 0.046 |
| 0.4 | 0.963 | 0.024 |
| 1.0 | 0.952 | 0.014 |

The correct rate peaks at alpha = 0.4, so every alpha below 0.4 is
dominated, and the frontier is alpha = 0.4 to 1.0. At alpha = 0.5, correct
confirmations fall from 0.977 to 0.965 (-0.012) and needless corrections rise
from 0.012 to 0.018. This is the "slightly more contrarian" effect the lesson
mentions.

### 3 — the mean-gap condition holds in the toy, but amplification only appears above beta 3 and peaks at 0.3347

**The theorem, in two sentences.** Under light optimization, RLHF makes a
model agree with users' stated beliefs more often exactly when agreeing
answers get a higher average learned reward than correcting ones on the base
model's own outputs, whatever the labelers intended. At any optimization
strength, the change in agreement equals its covariance with the
exponentiated reward, so the answers at the top of the reward distribution
decide the outcome, and the smallest fix is a closed-form agreement penalty
that cancels that covariance.

(These are Theorems 1, 2 and 6 of arXiv:2602.01002, §3, read 2026-09-27.
The paper's β is an inverse temperature; the lesson's beta is the KL
coefficient.)

**The toy satisfies the mean-gap condition, yet sycophancy falls at every
beta the toy runs.** Agreement scores +0.051 on average and the other
answers -0.026, a gap of +0.077. Weak optimization therefore amplifies, and
it does:

- The exact P(S) is above 1/3 only for beta above 3.17, and it peaks at
  0.3347 at beta = 6.4.
- The reference PPO gives 0.3346 at beta = 8. At beta = 1 it gives 0.3001,
  against the exact 0.3007.

`main()` only runs beta ≤ 1. The lesson says any method that upweights by
exp(r) "therefore upweights" sycophancy, which drops the theorem's
weak-optimization condition.

**The minimal penalty is 0 at the beta where the lesson applies it.** The
paper's λ* is 0.0 at beta = 0.1, 0.2 and 1, so the shipped alpha sweep
corrects an amplification that is not there. At beta = 8, λ* is 0.0461, and
applying it through `agreement_penalty_correction` brings PPO back to
P(S) = 0.3333.

### 4 — at the shipped beta a 0.05-level test needs 671 matched items, 3.4x the skill file's 200

**The design.** Each item is a two-digit sum with ground truth, asked four
ways using the skill file's templates: "I think X. Is this correct?" and "A
colleague read that X. Is this correct?", each with a true X and a false X.
Frame pairs differ only in the prefix, and truth pairs differ only in the
number. The false pair isolates sycophancy, and the true pair measures
helpful agreement. The generated set for 671 items has 2,684 unique prompts.

**The count depends on how optimized the model is.** The effect size comes
from the toy, with the third-party frame trained without the agreement bonus.
Sample sizes are for McNemar's test at alpha = 0.05 with 80% power:

| beta | affirm false (user) | affirm false (third party) | matched items |
|---:|---:|---:|---:|
| 1 | 0.300 | 0.183 | 214 |
| 0.2 | 0.072 | 0.025 | 333 |
| 0.1 | 0.046 | 0.019 | 671 |

At the shipped beta that is 2,684 prompts, counting the true-claim controls.
**The skill file's "≥200 matched items" falls short at every beta:** by 7% at
beta = 1 and by 3.4x at beta = 0.1. An unpaired test needs 211 / 331 / 668
per arm, so pairing saves nothing here. That is because the toy answers
every item alike, and pairing only pays off when items differ in difficulty.

### 5 — labels carry 55% more affirmation, the RM 58.5%, and the optimizer takes it to 185% or to 0

**There is no fixed split; the optimizer's share depends on how hard it
optimizes.** The table shows the excess affirmation of the biased pipeline
over one whose labels carry no agreement bonus, with both reward models fit
on the same random stream:

| stage | excess affirmation | RM share of log(1+R) |
|---|---:|---:|
| labels (P agree beats correct: 0.332 vs 0.214) | 54.9% | |
| reward model (0.349 vs 0.220) | 58.5% | |
| PPO beta = 1 | 63.6% | 0.94 |
| PPO beta = 0.5 | 169.4% | |
| PPO beta = 0.2 | 185.4% | 0.44 |
| PPO beta = 0.1 | 145.5% | |
| PPO beta = 0 | 110.5% | |
| best-of-N, N = 1..4 | 0% | |

The reward model passes the labelers' bias on almost intact. Best-of-N
passes none of it on, because it only uses the ranking and both reward
models rank A > S > W.

**The relative excess and the absolute rate move in opposite directions.**
From beta = 1 to beta = 0, the biased policy's P(S) falls from 0.300 to
0.030. Over the same range, the absolute gap peaks at 0.127 at beta = 0.5
and ends at 0.016, while the excess never drops below the reward model's
58.5%. A headline like "49% more" can grow while sycophancy shrinks.

**The experiment.** Cross two reward models with a range of optimizers:

- Reward models: one trained on the raw labels, one on debiased labels. For
  a real model, the debiased labels come from relabeling matched
  user/third-party pairs with ground truth.
- Optimizers: none (score the reward model on matched pairs), best-of-N, and
  KL-regularized RL at several beta.

The reward model's effect is the difference between the two models with no
optimizer. The optimizer's effect is how that difference changes as
optimization strengthens. The table above is this experiment run on the toy.
