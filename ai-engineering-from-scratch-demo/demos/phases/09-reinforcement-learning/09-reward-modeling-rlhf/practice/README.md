<!-- generated:start -->
# 09-reinforcement-learning / 09-reward-modeling-rlhf

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/09-reinforcement-learning/09-reward-modeling-rlhf/) · upstream spec
`phases/09-reinforcement-learning/09-reward-modeling-rlhf/docs/en.md`

```bash
uv run demo practice run 09-reward-modeling-rlhf --ex 1
uv run demo explain 09-reward-modeling-rlhf --ex 1
uv run pytest demos/phases/09-reinforcement-learning/09-reward-modeling-rlhf
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Train the Bradley-Terry reward model in `code/main.py` on 500 synthetic preference pair… | code | T0 | `ex01_it_is_100_percent_and_ten_pairs_get_there.py` |
| 2 | Medium. Run the toy PPO-RLHF loop with `β ∈ {0.0, 0.1, 1.0}`. For each, plot RM score vs KL-t… | code | T0 | `ex02_all_three_hack_and_the_penalty_arrives_too_late.py` |
| 3 | Hard. Implement DPO (closed-form preference-likelihood loss) on the same preference data and… | code | T0 | `ex03_comparing_on_rm_score_rewards_whichever_hacks_hardest.py` |
<!-- generated:end -->

## Answers

The three exercises are a pipeline — fit a reward model, optimise against it, then
replace the optimiser — and each stage turns out to be settled before its exercise
begins. The reward model is perfect because its two vocabularies are disjoint. The
KL penalty cannot prevent reward hacking because the advantage normalisation
deletes it. And the comparison the third exercise asks for is scored on the axis
reward hacking is defined to maximise.

All three are **T0** — the whole lesson is 12 tokens and 3 prompts, and the pack
runs in under a second.

### 1 — it is 100%, and ten pairs get there

| training pairs | 1 | 2 | 3 | 5 | **10** | 20 | 50 | 500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| held-out accuracy | 0.792 | **0.954** | 0.986 | 0.995 | **1.000** | 1.000 | 1.000 | **1.000** |

**ANSWER: 1.0000, on every one of 20 seeds** — minimum 1.0000, not a single missed
held-out pair anywhere.

**FINDING: ten pairs get there, and two clear the stated bar.** The exercise asks
for 500, which is **50× more** than the task needs.

**MECHANISM: `GOOD` and `BAD` share no token.** Their intersection is empty:

```python
GOOD = ("clear", "specific", "kind", "thorough", "precise", "helpful")
BAD  = ("vague", "rude", "wrong", "short", "cold", "careless")
```

`sample_pair` draws the preferred response from `GOOD²` and the rejected one from
`BAD²`, so a bag-of-words scorer separates them as soon as it has seen each of the
12 tokens once — one gradient step each. No example in the distribution is
ambiguous, so there is nothing for the model to be wrong about.

**FINDING: the held-out set is not held out from anything that matters.** 3,888
distinct (prompt, preferred, rejected) triples exist, but the model has **12
parameters** and the labelling rule is a partition of them into 6 and 6. Train and
test differ in sample, not in structure.

**FINDING: the separation is total, not merely accurate.** 6 of 6 `GOOD` tokens end
with positive weight and 6 of 6 `BAD` with negative. The sign pattern alone already
decides every pair the generator can produce.

### 2 — all three hack, and the penalty arrives after it is over

The reference policy is uniform over 12 tokens, so KL is bounded above by
`ln 12 = 2.4849`.

| β | final RM | final KL | % of max KL | top token probability |
|---:|---:|---:|---:|---:|
| 0.0 | 1.1793 | 2.2524 | **91%** | 0.958 |
| 0.1 | 1.1778 | 2.2183 | **89%** | 0.947 |
| 1.0 | 1.1691 | 1.9424 | **78%** | 0.872 |

(uniform would be 0.083.)

**ANSWER: all three, including `β = 1.0`.** Every arm collapses onto the single
highest-scoring token.

**FINDING: `β` has to reach 100 before the KL is meaningfully held.**

| β | 0 | 0.1 | 1 | 10 | **100** |
|---|---:|---:|---:|---:|---:|
| KL as % of max | 91% | 89% | 78% | 46% | **20%** |

A **1000×** increase over the lesson's default, and it costs 35% of the RM score.
Nothing inside the range the exercise states binds at all.

**MECHANISM: the advantage normalisation deletes the penalty.** The reward is
`rm_score − β·KL`, but `kl()` is computed from the *whole distribution*, so it
depends on the prompt and not on the sampled token — a 16-rollout batch holds at
most **3** distinct values of it against 12 for the RM score. Then:

```python
advs = [(r - mean_r) / sd_r for r in rewards]
```

which removes exactly what is common across the batch.

**FINDING: and the penalty only becomes visible once it is too late.**

| | sd(rm) | sd(β·KL) |
|---|---:|---:|
| update 10 (steering still possible) | **1.1071** | 0.000666 |
| update 150 (collapsed) | **0.0000** | 0.004505 |

Early, the reward-model term outweighs the penalty by **1662×** — the penalty is
invisible after standardisation. By update 150 the policy has collapsed so hard that
every rollout draws the same token and `sd(rm)` is *exactly zero*; only then does
the KL term dominate the advantage, with nothing left to steer. The term meant to
prevent the collapse becomes audible after it has finished.

**FINDING: the hack lands exactly on the reward model's ceiling.** The highest
single-token weight is 1.1873; the β=0 arm reaches 1.1793, **99.3%** of it. When the
reward model is a bag of 12 weights there is one best token, and the policy finds
it.

### 3 — comparing on RM score rewards whichever method hacks hardest

At 2,400 gradient steps each:

| | RM score | KL | % of max KL | policy samples | RM queries |
|---|---:|---:|---:|---:|---:|
| PPO-RLHF (β=0.1) | **1.178** | 2.271 | **91%** | 2,400 | 2,400 |
| DPO | 0.893 | 0.397 | **16%** | **0** | **0** |

**ANSWER: PPO scores higher and DPO stays closer.** PPO reaches **99.2%** of the
reward model's ceiling — by becoming the one token that ceiling belongs to.

**FINDING: DPO uses no policy samples and no reward model at all.** PPO-RLHF spends
600 gradient steps fitting the reward model, then draws 150 × 16 samples from the
policy and queries the reward model once per sample. DPO reads the same preference
pairs directly, so **stage 1 never has to happen** — the 600 pairs the reward model
consumed are the same pairs DPO trains on.

**FINDING: DPO's score rises with data and its KL rises with it.**

| pairs | 600 | 2,400 | 10,000 |
|---|---:|---:|---:|
| RM score | 0.295 | 0.893 | 1.086 |
| KL as % of max | 2% | 16% | 28% |

DPO is not on a different trajectory from PPO — it is further back along the same
one. Every gain in reward-model score is bought with distance from the reference,
for both methods.

**FINDING: the exercise's two axes disagree, and only one of them is stated.**
Ranked by final RM score, PPO wins by 0.285. Ranked by KL-to-reference at equal
compute, DPO wins by a factor of **5.7**. Exercise 2 established that RM score is
maximised by collapsing onto one token — so "final RM score achieved" is precisely
the axis reward hacking is defined to maximise, and comparing on it alone scores the
two methods by how hard each one hacks.
