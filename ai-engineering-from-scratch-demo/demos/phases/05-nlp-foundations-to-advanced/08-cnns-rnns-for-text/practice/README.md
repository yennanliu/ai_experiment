<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 08-cnns-rnns-for-text

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/08-cnns-rnns-for-text/) · upstream spec
`phases/05-nlp-foundations-to-advanced/08-cnns-rnns-for-text/docs/en.md`

```bash
uv run demo practice run 08-cnns-rnns-for-text --ex 1
uv run demo explain 08-cnns-rnns-for-text --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/08-cnns-rnns-for-text
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Train a TextCNN on a 3-class toy dataset (you invent the data). Verify that filter widt… | code | T1 | `ex01_most_of_the_gain_is_filter_count.py` |
| 2 | Medium. Implement max-pool, mean-pool, and last-state pooling for the LSTM classifier. Compar… | code | T1 | `ex02_last_state_forgets_after_two_steps.py` |
| 3 | Hard. Build a BiLSTM-CRF NER tagger (combine lesson 06 and this one). Train on CoNLL-2003. Co… | code | T0 | `ex03_the_three_numbers_it_asks_for_are_not_defined.py` |
<!-- generated:end -->

## Answers

The lesson's `code/main.py` is three functions — a gradient-attenuation formula,
a 1-D convolution and a max-pool — and no framework is installed, so all three
exercises are answered with frozen encoders and a trained linear readout. That
turns out to be the right instrument for what they actually ask: each exercise
compares two architectural choices, and freezing everything else is what makes
the comparison readable.

Exercises 1 and 2 are **T1** (numpy, scikit-learn); exercise 3 is **T0** — it is
arithmetic.

### 1 — Most of the gain is filter count

The corpus plants `very good` in class 0, `not very good` in class 1 and neither
in class 2, so only a width-3 window can separate the first two.

**ANSWER: widths (2,3,4) do beat width 3 — and the comparison changes two things.**

| arm | filters | macro-F1 (12 seeds) |
|---|---:|---:|
| width 3 | 8 | 0.4777 |
| widths 2,3,4 | **24** | **0.5910** |
| width 3, matched | **24** | 0.5705 |
| widths 5,6,7 | 24 | 0.4755 |

Three widths at eight filters each is 24 filters against 8. Held to 24 on both
sides the gain falls from **+0.1133 to +0.0205** — **82% of the effect was
capacity, not width**.

**CONTROL: width still matters, and in the right direction.** Widths (5,6,7),
also 24 filters, score 0.4755 — level with width 3 at a *third* the count. A
six-wide window averages away a two-token and a three-token pattern, so the
extra capacity buys nothing. The width effect is small and it is not zero.

**FINDING: "on average" is doing work.** The multi-width set wins on **9 of 12**
seeds and the per-seed gap runs **−0.1655 to +0.3897** around a mean of +0.1133 —
a spread three times the effect. One training run answers this exercise either
way.

### 2 — Last-state forgets after two steps

Each sequence is 20 filler tokens with one cue planted at a chosen position. The
encoder is a frozen tanh recurrence with its spectral radius scaled to the
lesson's own **0.9**, so the three poolings read *identical* hidden states and
any difference between them is the pooling.

**ANSWER: which pooling wins is decided by where the cue is.**

| cue position | 0 | 10 | 16 | 17 | 18 | 19 |
|---|---:|---:|---:|---:|---:|---:|
| distance from end | 19 | 9 | 3 | 2 | 1 | 0 |
| max-pool | 0.7167 | 0.7333 | 0.7167 | 0.7333 | 0.7333 | 0.7056 |
| mean-pool | 0.7389 | 0.6778 | 0.6667 | 0.7056 | 0.6833 | 0.7000 |
| **last-state** | 0.4611 | 0.4611 | 0.4833 | **0.7389** | **0.9444** | **1.0000** |

**MECHANISM: max and mean have no notion of *when* a step happened**, so their
scores move by 0.0277 and 0.0722 across the whole sweep. Last-state has nothing
else.

**FINDING: the split is exact, with the threshold at three.** Max-pool wins at
every position ≥3 steps from the end and loses at every position nearer — 3 to
3, no exception. A corpus whose verdict falls in the last two tokens ranks
last-state first; one whose evidence is anywhere else ranks it last.

**MECHANISM: the lesson's own `vanishing_gradient_sim` predicts this, and is
optimistic.** It returns 0.9^d, so it predicts **0.729** retained at three steps
— where the classifier is already at **0.4833**, chance on a two-class problem.
The linear model does not see tanh saturating, nor the 17 later tokens
overwriting a 24-dimensional state.

### 3 — The three numbers it asks for are not defined

`torch`, `tensorflow`, `jax` and `transformers` are all absent and CoNLL-2003 is
not downloadable, so no training time and no measured F1 are reported. Two of
the three quantities do not need a GPU.

**ANSWER: memory, as stored activations per sequence.**

| n | BiLSTM (2nH) | attention (L·heads·n²) | CRF lattice (n·\|tags\|²) |
|---:|---:|---:|---:|
| 8 | 4,096 | 4,608 | 648 |
| 32 | 16,384 | 73,728 | 2,592 |
| 128 | 65,536 | 1,179,648 | 10,368 |
| 512 | 262,144 | **18,874,368** | 41,472 |

**The ranking inverts at n = 8** — shorter than any CoNLL sentence — and by
n = 512 the attention side is **72×** the recurrent side. "Report memory" is
answered by choosing n. The CRF lattice never competes for the answer: it is the
part the exercise treats as the baseline and the part that costs nothing.

**FINDING: 92% of the "BiLSTM-CRF" is a vocabulary.**

| | parameters |
|---|---:|
| embedding table | **23,040,000** |
| BiLSTM weights | 2,099,200 |
| CRF transitions | 81 |
| **BiLSTM-CRF total** | **25,139,281** |
| 6-layer transformer + same table | 65,507,328 (**2.61×**) |

Halving the vocabulary moves the reported memory more than deleting the encoder
does, and the two totals differ by 2.61× rather than an order of magnitude
because both carry the same table underneath.

**CONTROL: no architecture here moves F1 as much as lesson 06's split did.**
Lesson 06 measured the same model on the same corpus scoring 1.0000 entity F1
under a by-sentence split and 0.0952 under a by-entity one — **0.9048 of F1 from
an evaluation choice**, with model, features and data held fixed.

### A note on file lengths

The three files run 139 / 149 / 110 lines of code; the first two are over D14's
120-line target and under its 150-line ceiling. The overrun is the substitute
model in both cases: no framework is installed, so each file carries its own
corpus generator, frozen encoder and readout in order to ask the architectural
question the exercise poses. Exercise 3 needs none of that and stays at 110.
