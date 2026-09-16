<!-- generated:start -->
# 10-llms-from-scratch / 08-dpo

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/08-dpo/) · upstream spec
`phases/10-llms-from-scratch/08-dpo/docs/en.md`

```bash
uv run demo practice run 08-dpo --ex 1
uv run demo explain 08-dpo --ex 1
uv run pytest demos/phases/10-llms-from-scratch/08-dpo
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement KTO (Kahneman-Tversky Optimization). KTO doesn't need pairs -- just label each resp… | code | T1 | `ex01_each_method_prefers_the_others_metric.py` |
| 2 | Implement length-normalized DPO. Instead of raw log-probabilities, divide by the number of re… | code | T1 | `ex02_normalising_removes_the_only_signal.py` |
| 3 | Build an ORPO-style combined loss. Add a standard next-token prediction loss on the preferred… | code | T1 | `ex03_the_dpo_term_is_a_constant_eleven_percent.py` |
| 4 | Implement iterative DPO. Run DPO for 3 epochs, then generate new responses from the trained m… | code | T1 | `ex04_the_unit_decides_whether_anything_inverts.py` |
| 5 | Compare DPO with different reference models. Instead of using the SFT checkpoint as the refer… | code | T1 | `ex05_the_winner_wins_both_questions_on_noise.py` |
<!-- generated:end -->

## Answers

`code/main.py` is DPO in numpy: sequence log-probabilities, the Bradley-Terry
loss with its implicit rewards, a frozen reference, a training loop, and a
preference-accuracy metric. Every exercise here is a variant — KTO, length
normalisation, ORPO, self-play, alternative references — and every one of them
runs into the same two facts.

All five are **T1** on the `math` group (`uv sync --extra math`).

### The two facts

**The reference is a copy of the policy, so every log-ratio is exactly zero.**
`copy_model_weights` duplicates the policy into the reference, and DPO's whole
signal is `pi_logprob − ref_logprob`. At the state training starts from, that is
`0.0` for all 12 responses — not approximately, exactly. So `L_dpo = 0.6931` and
every implicit reward margin is `0.0000`, for every pair, at every beta and
every alpha.

**`dpo_train` updates in a random direction**, with `update_direction = 1.0 if
logit < 0 else -0.1` scaling a fresh `np.random.randn(...)` each step. Across
three epochs the largest single weight moves **9.9e-07**.

Together these make `evaluate_preference_accuracy` return **0.000**, not 0.5:
`preferred_reward > rejected_reward` is `0 > 0`, false for every pair. The floor
of the lesson's own metric is the state its own trainer starts in.

### 1 — each method prefers the other's metric

Both arms are trained, as the exercise says to: DPO through the lesson's own
`dpo_train`, KTO through the same update rule with the KTO loss deciding the
direction, from one initialisation and one RNG stream. Both are then scored
under *both* metrics, because "compare accuracy against DPO" does not say which
metric the comparison uses.

**FINDING: both methods start from an exact tie and break it differently.**
`copy_model_weights` makes the reference the policy, so `pi_logprob -
ref_logprob` is 0.0 for all twelve responses. KTO's loss is fixed at `0.6931 +
1.5 × 0.6931 = 1.7329` per pair and DPO's at `0.6931`, whatever the data says.
Untrained, KTO scores **0.500** — the base rate of a constant "bad" classifier,
since `0 > 0` is false — and DPO **0.000**, the floor of a strict inequality on
two equal numbers.

**ANSWER: after training, each scores better under the other's metric.**

| trained with | KTO metric | DPO metric |
|---|---:|---:|
| DPO | 0.417 | **0.333** |
| KTO | **0.333** | 0.667 |

KTO training produces the best DPO accuracy in the table and the worst KTO
accuracy. "Compare accuracy against DPO" has four answers and they do not agree
on an ordering.

**MECHANISM: neither trainer uses a gradient.**

```python
update_direction = 1.0 if metrics["logit"] < 0 else -0.1     # dpo_train
block.ffn.W1 += lr * update_direction * np.random.randn(*shape) * 0.01
```

A random direction whose only tie to the loss is a sign. After 5 epochs at
`lr=5e-6` the largest weight has moved 8.3e-07 under DPO and 1.4e-06 under KTO,
against weights of scale 0.02. Every accuracy above is the sign of noise, read
on 6 pairs and quantised to sixths.

**FINDING: `evaluate_preference_accuracy`'s floor is 0, not 0.5.** A model that
is exactly indifferent — the state DPO is initialised into, by construction —
scores 0% where a preference metric should give it 50%. The first number the
lesson's own training loop can print is the worst one the metric has.

### 2 — normalising removes the only signal

| ranking by | correct of 6 |
|---|---:|
| raw total log-probability | **4** |
| total ÷ response tokens | **1** |

**ANSWER: the premise is right and the fix costs accuracy.** Total log-prob
correlates **−0.987** with response length, so raw ranking picks the shorter
response — which is correct in 4 of 6 pairs, because preferred is shorter in 4
of 6 pairs.

**MECHANISM: per-token log-probability barely varies** — the twelve responses
span **−5.5443 to −4.3577** nats/token, a 1.19-nat window against a uniform
baseline of 5.5452. Dividing by length removes the quantity that varies a lot
and leaves the one that hardly varies.

**FINDING: on this dataset the length bias *is* the signal** — 67% accurate,
against 17% for the corrected version. The bias is wrong in general and right on
a dataset built by padding the rejected response.

### 3 — the DPO term is a constant 11%

| alpha | L_sft | L_dpo | combined | DPO share |
|---:|---:|---:|---:|---:|
| 0.1 | 5.3734 | 0.6931 | 5.4427 | **1.27%** |
| 0.5 | 5.3734 | 0.6931 | 5.7199 | 6.06% |
| 1.0 | 5.3734 | 0.6931 | 6.0665 | **11.43%** |

**ANSWER: even at alpha = 1.0 the objective is 88.6% SFT.**

**MECHANISM: the terms are 7.8× apart before alpha is applied.** `L_sft` starts
near its ceiling (5.3734 against `ln(256) = 5.5452`) with the whole range below
it; `L_dpo` is pinned at 0.6931 by a margin that is structurally zero.

**FINDING: the DPO term's *value* is constant — its gradient is not zero.**
`L_dpo` is 0.6931 for all six pairs because every implicit margin is exactly 0,
but that is one value repeated, not a flat function: differentiating
`-log(sigmoid(beta·m))` at `m = 0` through the lesson's own `dpo_loss` gives
**−0.0500 = −beta/2**. The term supplies a preference gradient and alpha scales
it. What makes it inert here is `dpo_train`, which never computes that gradient
— its update is `lr * (1.0 if logit < 0 else -0.1) * randn(...)`. Reaching an
even split needs **alpha = 7.8**, and the term it would weight is still 0.6931
everywhere.

### 4 — the unit decides whether anything inverts

| | preferred | rejected | preferred shorter |
|---|---:|---:|---:|
| original pairs | 50.3 tok | **105.0 tok** | 4 / 6 |
| self-play pairs, in tokens | 50.3 tok | **56.3 tok** | **4 / 6** |
| self-play pairs, in characters | 50.3 ch | 28.5 ch | 2 / 6 |

**ANSWER: 0.000 after round 1, 0.000 after round 2.** Not one pair changes sign.

**FINDING: self-play does not invert the length relation.** In the units DPO
scores — `tokenize_sequence` — the round-1 samples are 56.3 tokens, still longer
than preferred's 50.3, and preferred is the shorter response in 4 of 6 pairs
before and after. The property Exercise 2 shows these log-probabilities track is
halved by round 2 and left pointing the same way.

**FINDING: counted in characters the same data says 2 of 6.** `sample` decodes
the drawn bytes with `errors="replace"`, and 11 to 15 of each sample's ~29
characters are U+FFFD. `len(str)` counts each once; `tokenize_sequence`
re-encodes each as three bytes. The unit, not the self-play, produces the
inversion.

**FINDING: the generated responses are not responses.** 30 tokens sampled from a
model that moved **1.7e-06** across two whole DPO runs — `(human-written answer,
undecodable bytes)`.

**FINDING: 0.000 is the metric's floor.** Every implicit margin is exactly 0, so
the strict inequality is false for all six pairs before either round and after
both.

### 5 — the winner wins both questions on noise

Each strategy gets its own DPO run, from identical initial weights and the same
RNG stream, driven one epoch at a time so that (b) can snapshot the policy after
epoch 1 and (c) can keep a genuine EMA.

| reference | curve range | margin spread | accuracy |
|---|---:|---:|---:|
| SFT checkpoint | 3.8e-06 | 2.7e-06 | 0.000 |
| base model | **0.3164** | **0.2027** | 0.167 |
| epoch-1 checkpoint | **3.1e-06** | 9.2e-07 | **0.833** |
| EMA of the policy | 3.4e-06 | 2.2e-06 | 0.000 |

**ANSWER: the epoch-1 checkpoint wins both questions at once** — highest
accuracy and flattest curve, which is exactly the combination the exercise asks
for. It wins them on margins of order 1e-06 produced by a policy that moved
9e-07.

**FINDING: two references that agree to 1e-06 score 0.000 and 0.833.** The
epoch-1 snapshot and the SFT checkpoint are the same weights to six decimal
places, so `preferred_reward > rejected_reward` is a sign test on noise. Five of
six land one way against one reference and none against the other.

**FINDING: only the base model produces a curve at all.** The other three
references are copies of the policy, so `pi_logprob − ref_logprob` is 0 for
every response and the loss is `log(2) = 0.6931` at every step. Three of the four
candidates for "most stable curve" are flat lines.

**FINDING: most stable and most informative point in opposite directions.** The
only arm whose margins distinguish the six pairs is the least stable one — and
it is least stable because its reference is a *different random
initialisation*, so the spread measures the gap between two random models rather
than anything DPO learned.
