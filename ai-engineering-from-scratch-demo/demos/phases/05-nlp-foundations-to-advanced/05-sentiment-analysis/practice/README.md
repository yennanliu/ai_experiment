<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 05-sentiment-analysis

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/05-sentiment-analysis/) · upstream spec
`phases/05-nlp-foundations-to-advanced/05-sentiment-analysis/docs/en.md`

```bash
uv run demo practice run 05-sentiment-analysis --ex 1
uv run demo explain 05-sentiment-analysis --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/05-sentiment-analysis
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Add `apply_negation` as a preprocessing step in the scikit-learn pipeline and measure t… | code | T1 | `ex01_the_dead_entry_in_negation_words.py` |
| 2 | Medium. Implement class-weighted logistic regression (pass `class_weight="balanced"` to sciki… | code | T1 | `ex02_balanced_loses_on_the_metric_that_names_it.py` |
| 3 | Hard. Build a sarcasm detector by training a second classifier on the residuals of the sentim… | code | T1 | `ex03_the_residual_set_is_not_a_sample_of_sarcasm.py` |
<!-- generated:end -->

## Answers

Three exercises about measurement. Exercise 1 asks for one F1 number that turns
out to average a gain and a collapse. Exercise 2 says "measure the effect"
without saying on what, and the three obvious answers disagree on the sign.
Exercise 3 warns you that your accuracy will land at chance and does not say
why — it is because a model's residuals are not a sample of the phenomenon you
wanted them to be.

All three are **T1** (numpy and scikit-learn, no GPU, no key).

### 1 — The dead entry in `NEGATION_WORDS`

**ANSWER: the F1 delta is +0.6667**, from 0.0000 to 0.6667, on 128 training and
96 test rows generated from 4 frames and a 16-word polarity lexicon.

| | plain | spelled-out | contracted | F1 | accuracy | vocab |
|---|---:|---:|---:|---:|---:|---:|
| without `apply_negation` | 0.500 | 0.500 | 0.500 | 0.0000 | 0.500 | 29 |
| with `apply_negation` | **1.000** | **1.000** | **0.000** | 0.6667 | 0.667 | 45 |

**MECHANISM: without scoping, every sentiment word sits equally in both
classes.** Training carries each word plain (its true polarity) and under
spelled-out negation (the opposite), so bag-of-words sees `good` as often in one
class as the other and the model is at chance on all three forms. The step
separates the features: `the ending was not good` becomes
`['the', 'ending', 'was', 'not', 'NOT_good']`.

**MECHANISM: `NEGATION_WORDS` lists `n't`, and `TOKEN_RE` can never produce
it.** The pattern is `[A-Za-z]+(?:'[A-Za-z]+)?`, which matches `wasn't` whole —
tokenizing `wasn't`, `didn't`, `isn't` and `won't` yields `n't` **nowhere**. The
entry is unreachable, so the lesson's clear intent to scope contractions never
fires.

**FINDING: the step turns a coin flip into a confident wrong answer.**
`the ending wasn't good` scopes to itself, `good` intact. *Before* the step
`good` was ambiguous and the item was 50/50; *after* it, `good` is an
unambiguous positive feature, so every contracted negation is confidently
misread — 0.500 → **0.000**. Contraction is how most English negation is
actually written, and the single F1 number reports the average of a doubling and
a collapse.

### 2 — `balanced` loses on the metric that names it

**ANSWER: two of three headline metrics get worse.** 2000 rows at 90-10, split
`[1254, 146]` train and `[538, 62]` test:

| | accuracy | F1 | recall | precision | predicted positive |
|---|---:|---:|---:|---:|---:|
| plain | **0.9633** | **0.7963** | 0.6935 | 0.9348 | 46 |
| `class_weight="balanced"` | 0.8783 | 0.5876 | **0.8387** | 0.4522 | 115 |
| majority-class predictor | 0.8967 | 0.0000 | 0.0000 | 0.0000 | 0 |

**FINDING: the do-nothing predictor beats the balanced model on accuracy.**
0.8967 against 0.8783 — while scoring F1 0.0 and never naming the minority class
once. On the metric the exercise's phrasing most naturally invites, predicting
nothing outranks the repair.

**MECHANISM: the weight is `n / (2 × count)` = 1400 / (2 × 146) = 4.79**, and
the model spends it on more positive guesses: 115 where the plain model said 46,
against **62** that are real. Those 69 extra answers recover **9** more true
positives and drop precision 0.9348 → 0.4522.

**CONTROL: the imbalance is not pathological.** The unweighted model already
beats the baseline at 0.9633 accuracy and 0.7963 F1 — 90-10 has not collapsed it
into the majority class. Weighting is a choice of operating point on the
precision/recall curve, not a rescue, and it changes the threshold rather than
the ranking.

### 3 — The residual set is not a sample of sarcasm

**Setup**, since the exercise asks for it: 96 rows from 4 frames and a 16-word
lexicon. 64 sincere, label following the word. 32 sarcastic — positive words,
negative sentiment — reusing the same frames; 16 carry the cue `oh sure` and
**16 are byte-identical to a sincere positive row**. 70/30 split on a fixed
permutation, the lesson's own `tokenize` + `apply_negation` as the analyzer,
logistic regression.

**ANSWER: the residual-trained detector lands exactly at the majority rate.**
0.6207 on test, against a majority-class rate of **0.6207**. It learned nothing —
at chance, which is where the exercise says most first attempts land.

**MECHANISM: residuals are not a sample of sarcasm.**

| | |
|---|---|
| sentiment accuracy | 0.8507 train / 0.6207 test |
| training residuals | **10 of 67** |
| sarcastic *among* those residuals | **3 of 10** |
| sarcastic in training overall | 21 of 67 |
| "is an error" scored against the sarcasm label | precision **0.300**, recall **0.143** |

Errors have many causes and sarcasm is one of them. The second classifier is
trained to predict the *first model's mistakes*, a target that agrees with
sarcasm on 30% of what it flags and misses six of every seven sarcastic rows.

**MECHANISM: the residual rows alone have no contrast.** Only 10 of 67 rows are
residuals and every one of them is an error, so "train a classifier on the
residuals" read literally is a one-class fit. The target used here — "is an
error", over the whole training set — is the nearest well-posed problem.

**FINDING: the error floor is irreducible.** 16 of the 32 sarcastic rows repeat
a sincere positive row exactly, so no feature of the text separates them.
Sweeping the sentiment model's `C` over `[1, 1e3, 1e6]` leaves the training
residual count at `[10, 11, 11]` — it never approaches zero, because it cannot.
"Most first attempts land there" is not a failure of effort.

### A note on file lengths

The three files run 139 / 111 / 149 lines of code, over D14's 120-line target on
two of the three and under its 150-line ceiling. The overrun is fixture: the
exercises name "a small sentiment dataset", "a synthetic 90-10 class imbalance"
and a sarcasm corpus, and none of the three ships, so each file generates and
labels its own — exercise 3's, in particular, has to control the *overlap*
between the sarcastic and sincere rows for the measurement to mean anything.
