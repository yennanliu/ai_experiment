<!-- generated:start -->
# 10-llms-from-scratch / 07-rlhf

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/07-rlhf/) · upstream spec
`phases/10-llms-from-scratch/07-rlhf/docs/en.md`

```bash
uv run demo practice run 07-rlhf --ex 1
uv run demo explain 07-rlhf --ex 1
uv run pytest demos/phases/10-llms-from-scratch/07-rlhf
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Modify the reward model to use the mean of all hidden states instead of just the last positio… | code | T1 | `ex01_mean_pooling_wins_because_nothing_aggregates.py` |
| 2 | Implement reward model calibration. After training, run all preference pairs through the rewa… | code | T1 | `ex02_the_margin_is_negative_and_gets_worse.py` |
| 3 | Simulate reward hacking. Create a reward model that gives high scores to long responses (rewa… | code | T1 | `ex03_the_length_reward_is_a_constant.py` |
| 4 | Implement a multi-objective reward. Train two reward models -- one for helpfulness and one fo… | code | T1 | `ex04_the_conciseness_term_does_all_the_work.py` |
| 5 | Compare different KL coefficients. Run PPO with beta=0.001 (too low, reward hacking), beta=0.… | code | T1 | `ex05_all_three_betas_give_the_same_numbers.py` |
<!-- generated:end -->

## Answers

`code/main.py` is the full RLHF loop in numpy: a `RewardModel` on top of lesson
04's transformer, Bradley-Terry preference training, KL against a frozen
reference, and a PPO episode loop. Every piece is present and correctly named,
and two of them do not do what their names say — which is what all five
exercises end up measuring.

All five are **T1** on the `math` group (`uv sync --extra math`).

### The two facts the exercises rest on

**`train_reward_model` fits the head to a feature the model never computes.**
The update multiplies the Bradley-Terry gradient by

```python
rm.ln_f.forward(rm.embedding.forward(preferred_ids))[:, -1, :]
```

— the embedding passed straight into the final LayerNorm, **skipping every
transformer block** — while `forward` scores `ln_f(blocks(embedding(x)))[-1]`.
Cosine between the two: **0.50**. And the gradient uses only `h_preferred` where
the derivative is `(σ(d) − 1)·(h_preferred − h_rejected)`, so each step raises
the rejected response's score too. Training ends at **33% accuracy on its own
six pairs**.

**`ppo_training` updates in a random direction.**

```python
block.ffn.W1 += lr * total_reward * np.random.randn(*block.ffn.W1.shape) * 0.01
```

The reward sets the step *size*; the direction is a fresh Gaussian draw every
episode, so the expected update is zero whatever the reward says.

### 1 — mean pooling wins because nothing aggregates

| pooling | correct of 6 | mean margin | margin spread |
|---|---:|---:|---:|
| last position | **2** | **−0.0383** | 0.0503 |
| mean | **4** | +0.0041 | **0.0081** |

**ANSWER: mean pooling, 4–2.** Last-position scores *below* chance — its margin
is negative, so it prefers the rejected response on average rather than merely
guessing.

**MECHANISM: the exercise names the reason itself.** Last-position "relies on
the causal attention to aggregate information", and `train_reward_model` updates
`reward_head` and nothing else — every attention matrix is bit-for-bit at
initialisation. Mean pooling wins because averaging 128 random projections is a
lower-variance summary than reading one: 6× tighter margins.

### 2 — the margin is negative, and worse on held-out data

| | (a) preferred | (b) rejected | (c) margin | correct |
|---|---:|---:|---:|---:|
| the 6 training pairs | +0.1752 | +0.2135 | **−0.0383** | 2 / 6 |
| 4 new pairs | +0.0526 | +0.1632 | **−0.1106** | 1 / 4 |

**ANSWER: the margin is negative**, and on unseen data it holds its sign and
triples. The exercise asks whether a clear margin survives; what survives is the
wrong sign.

**MECHANISM: reward correlates +0.268 with response length**, and the lesson's
dataset makes the *rejected* response the padded one — preferred is shorter in 4
of 6 pairs. A model that has learned only "longer scores higher" gets this
dataset backwards by construction.

**FINDING: a margin is a discrimination statistic, not a calibration one.**
Calibration is whether a score maps to a probability. The exercise's metric
would pass a model that ranked every pair correctly by 1e-9.

### 3 — the length reward is a function of the prompt

**ANSWER: reward hacking cannot occur.** `generate_response` appends exactly
`max_new_tokens` tokens, and there is no EOS token, stopping criterion or
early-exit branch anywhere in the file. The policy emits **20 tokens every
time**, so `len(response)/100` takes six values across twenty episodes — one per
prompt, each `(len(prompt) + 20)/100` — and the policy's contribution to all six
is the same constant 20.

**MECHANISM: length is a loop bound, not something the model can emit.** The one
quantity this reward measures is the one quantity the policy cannot influence.

**FINDING: the KL penalty changes nothing.** At `kl_coeff=0.1` the reward trace
is identical to `kl_coeff=0.0`, and the KL peaks at **6e-12**. The exercise asks
you to show a penalty preventing a behaviour the code cannot produce.

### 4 — the conciseness term does all the work

| objective | correct of 6 |
|---|---:|
| `1.0 × helpful` | **2** |
| `0.7 × helpful + 0.3 × concise` *(the exercise's)* | 4 |
| `0.0 × helpful + 1.0 × concise` | **4** |
| `0.9 × helpful + 0.1 × concise` | **5** |

**ANSWER: the combination gets 4 of 6, up from 2.** The improvement the exercise
predicts is real.

**FINDING: conciseness alone also gets 4 of 6.** The 0.7 weight on the trained
model contributes nothing the length term was not already supplying, and the two
pairs it still misses are the two where the preferred response is longer.

**FINDING: the verbosity trap is inside the helpfulness model, not opposite
it.** That model correlates +0.268 with length and scores below chance. The
conciseness term is overriding a signal that is verbose *and* unhelpful, not
balancing a helpful-but-verbose one.

**FINDING: 0.7 is on a plateau.** The whole range 0.0–0.7 ties at 4. The only
weight that does better is **0.9**, a 9:1 mix in favour of the component that
scores 2 of 6 alone — the opposite direction from the exercise's argument.

### 5 — all three betas give the same numbers

| beta | reward slope/episode | KL start | KL peak |
|---|---:|---:|---:|
| 0.001 | −1.35e-03 | 0.0 | 5.0e-13 |
| 0.02 | −1.35e-03 | 0.0 | 5.0e-13 |
| 0.5 | −1.35e-03 | 0.0 | 5.0e-13 |

**ANSWER: one curve, not three.** Rewards agree to 1e-12 at every episode; so do
the KLs. None of the exercise's labels — "too low, reward hacking", "standard",
"too high, no learning" — distinguishes anything.

**MECHANISM: `beta` multiplies a number that never exceeds 1e-12.**
`copy_model_weights` makes the policy and the reference the same model, so KL
starts at exactly 0. Even at beta=0.5 the penalty is 2.5e-13 against a reward of
order 0.1.

**MECHANISM: the policy never moves.** At `lr=1.5e-5` and reward ≈ 0.2 the step
is ~3e-8 per weight against weights of scale 0.02. After 20 episodes the largest
single weight has moved **3.8e-07**, and KL is quadratic in that displacement.

**FINDING: the reward trends the wrong way, and the trend is the prompt order.**
The exercise predicts "steady reward improvement" at beta=0.02; the fit is
**−1.35e-03 per episode**, a decline, and the same decline in all three runs.

| arm | reward slope/episode |
|---|---:|
| trained, beta = 0.001 / 0.02 / 0.5 | −1.35e-03 |
| policy frozen, same prompts and sampling | **−1.35e-03** (bit for bit) |
| policy frozen, prompt order reversed | −2.25e-04 |

Freezing the policy entirely — same prompts, same sampling stream, the update
computed and thrown away — reproduces the reward sequence exactly, so none of
the decline comes from learning. Reversing the prompt order on that same frozen
policy shrinks the fit sixfold. The per-prompt mean rewards run −0.12 to +0.18,
and 20 episodes over 6 prompts ends on the two lowest: the "reward curve" is the
prompt schedule sampled 20 times.
