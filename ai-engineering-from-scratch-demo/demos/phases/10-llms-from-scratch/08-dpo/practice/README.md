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
| 1 | Implement KTO (Kahneman-Tversky Optimization). KTO doesn't need pairs -- just label each resp… | code | T1 | `ex01_kto_wins_on_a_tie_break.py` |
| 2 | Implement length-normalized DPO. Instead of raw log-probabilities, divide by the number of re… | code | T1 | `ex02_normalising_removes_the_only_signal.py` |
| 3 | Build an ORPO-style combined loss. Add a standard next-token prediction loss on the preferred… | code | T1 | `ex03_the_dpo_term_is_a_constant_eleven_percent.py` |
| 4 | Implement iterative DPO. Run DPO for 3 epochs, then generate new responses from the trained m… | code | T1 | `ex04_self_play_inverts_the_length_relation.py` |
| 5 | Compare DPO with different reference models. Instead of using the SFT checkpoint as the refer… | code | T1 | `ex05_the_most_stable_curve_is_the_one_with_no_signal.py` |
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

### 1 — KTO wins on a tie-break

| | loss | accuracy |
|---|---:|---:|
| KTO (good + 1.5 × bad) | 0.6931 + 1.0397 = **1.7329** | **0.500** |
| DPO | **0.6931** | **0.000** |

**ANSWER: KTO 0.500, DPO 0.000** — and neither has seen a gradient. KTO calls a
response good when `beta * ratio > 0`; `0 > 0` is false, so it labels all twelve
bad, getting every rejected response right and every preferred one wrong.
Exactly half. DPO's strict inequality on two equal numbers gives zero.

**FINDING: the comparison is decided by tie-breaking conventions.** KTO's 0.500
is the base rate of a constant classifier on a balanced set; DPO's 0.000 is a
metric floor. Both losses are constants, because both log-ratios are zero.

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

**FINDING: the DPO term is constant, so it contributes no gradient.** Whatever
alpha is set to, the combined loss has one active term. Reaching an even split
needs **alpha = 7.8** — and the term it would weight is still 0.6931 everywhere.

### 4 — self-play inverts the length relation

| | mean rejected length | preferred shorter |
|---|---:|---:|
| original pairs | **105 bytes** | 4 / 6 |
| self-play pairs | **28 bytes** | 2 / 6 |

**ANSWER: 0.000 after round 1, 0.000 after round 2.** Not one pair changes sign.

**FINDING: self-play reverses the only relation the model can see.** The
original rejected responses are the *padded* ones; the round-1 policy samples 30
tokens, so the new "rejected" responses are the *short* ones. Exercise 2 shows
length is the only property these log-probabilities track.

**FINDING: the generated responses are 30 bytes from a model that moved
1.7e-06.** The new pairs are `(human-written answer, random bytes)`.

### 5 — the most stable curve is the one with no signal

| reference | accuracy | mean margin | margin spread |
|---|---:|---:|---:|
| (a) base model, different init | **0.167** | −0.1197 | **0.1343** |
| SFT checkpoint (a copy) | 0.000 | −0.0000 | **< 1e-5** |
| (c) EMA of the policy | 0.000 | +0.0000 | **< 1e-5** |

**ANSWER: the SFT copy and the EMA tie for most stable — their margins are all
essentially zero.** A reference equal to the policy gives `pi − ref = 0` for
every response, so the curve is a flat line at zero.

**FINDING: "most stable" and "most informative" point in opposite directions.**
The base model is the least stable arm by four orders of magnitude and the only
one whose margins carry information about the pairs.

**FINDING: option (b) is option (a) to six decimal places.** "A checkpoint from
epoch 1 of DPO" is the policy after one epoch, and the policy moves 9.9e-07
across all three. Two of the exercise's three options name one model.
