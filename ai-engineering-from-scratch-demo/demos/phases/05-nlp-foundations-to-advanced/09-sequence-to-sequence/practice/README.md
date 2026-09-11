<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 09-sequence-to-sequence

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/09-sequence-to-sequence/) · upstream spec
`phases/05-nlp-foundations-to-advanced/09-sequence-to-sequence/docs/en.md`

```bash
uv run demo practice run 09-sequence-to-sequence --ex 1
uv run demo explain 09-sequence-to-sequence --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/09-sequence-to-sequence
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Implement the toy copy task. Train a GRU seq2seq on input-output pairs where the target… | code | T1 | `ex01_the_simulation_never_reads_epochs.py` |
| 2 | Medium. Add beam search decoding with beam width 3. Measure BLEU on a small parallel corpus a… | code | T0 | `ex02_unnormalised_beam_returns_the_shortest_hypothesis.py` |
| 3 | Hard. Fine-tune `facebook/bart-base` on a 10k-pair paraphrase dataset. Compare the fine-tuned… | code | T0 | `ex03_ten_examples_is_not_a_measurement.py` |
<!-- generated:end -->

## Answers

Three exercises whose measurements are not what their names say. Exercise 1's
"accuracy" is a coin flip and its "training" parameters are never read. Exercise
2's beam search loses to greedy everywhere until the score is divided by length,
and then wins exactly where the exercise predicts. Exercise 3's evaluation could
not resolve the difference it is asked to report.

Exercise 1 is **T1** (numpy for the rank and the least-squares decoder); 2 and 3
are **T0**.

### 1 — The simulation never reads `epochs`

**ANSWER: `epochs` and `n_train` appear zero times in the function body.**

| `(epochs, n_train)` | lengths 5 / 10 / 20 |
|---|---|
| (200, 300) | 0.89 / 0.83 / 0.69 |
| (1, 1) | **0.89 / 0.83 / 0.69** |
| (100000, 1000000) | **0.89 / 0.83 / 0.69** |

Nothing is fitted — the embeddings are random and stay random.

**MECHANISM: the metric is a two-alternative comparison, so its floor is 0.50.**
The function scores the true sequence against *one* random sequence and counts
wins. The lesson's table bottoms out at **0.51** by length 80, which is chance
rather than collapse. Exact-match copy accuracy at length 20 over ten symbols has
a chance level of **1e-20** — the two numbers are not the same quantity.

**MECHANISM: a fittable model puts the bottleneck at an exact rank.** The one-hot
encoding of a length-L sequence over V symbols has rank **L(V−1)+1**, because
each position's block sums to one:

| L | predicted rank | measured rank |
|---:|---:|---:|
| 5 | 46 | **46** |
| 10 | 91 | **91** |
| 20 | 181 | **181** |

**ANSWER: exact-match reaches 1.000 once the context clears that rank.** Fixed
random projection into d dimensions, least-squares decoder on top:

| L \ d | 8 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|
| 5 (rank 46) | 0.000 | 0.920 | **1.000** | 1.000 |
| 10 (rank 91) | 0.000 | 0.010 | 0.850 | **1.000** |
| 20 (rank 181) | 0.000 | 0.000 | 0.000 | 0.535 |

The failure below the rank is **sharp** — 0.010 at d=32 against 0.850 at d=64 for
L=10 — not the smooth slide the lesson's simulation shows, because that slide is
a coin flip getting harder rather than a code running out of room.

**CONTROL: in the lesson's own simulation more context is not reliably better.**
At length 20, `context_dim` 4/8/16 gives 0.68 / 0.69 / **0.63**. With nothing
trained, extra dimensions add noise to a dot product rather than capacity to a
code — so the demonstration does not show the property it names.

### 2 — Unnormalised beam returns the shortest hypothesis

**ANSWER: beam-3 loses to greedy on every sentence.**

| decoder | BLEU | output lengths (references 3–7) |
|---|---:|---|
| greedy | 0.2507 | 3, 4, 5, 6, 7 |
| beam-3 | **0.0253** | **1, 1, 1, 1, 1** |
| beam-3, length-normalised | **0.2982** | 3, 4, 5, 6, 7 |
| beam-1 | 0.0213 | 1, 1, 1, 1, 1 |

**MECHANISM: every extra token subtracts.** The score is a sum of
log-probabilities, each negative, so a longer hypothesis cannot outscore its own
prefix unless a step is free. Beam-3 returns the shortest hypothesis the model
will end — that is not a search failure, it is the search succeeding at the
stated objective. Greedy never sees that hypothesis because greedy never compares
a finished sequence with an unfinished one.

**FINDING: divide by length and the exercise's claim is exactly right.**
Length-normalised beam-3 beats greedy and differs from it at exactly one position
on every sentence — the last:

| reference length | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|
| positions where they differ | [2] | [3] | [4] | [5] | [6] |
| final index | 2 | 3 | 4 | 5 | 6 |

At length 5: `['x','x','x','x','x']` versus `['x','x','x','x','y']`. The decoder
is built around a trap — `x` has the higher emission (0.45 vs 0.40) and the worse
continuation — so greedy takes `x` throughout, and beam takes it until the step
where nothing follows and the emission is the whole score. "Usually last tokens"
is right, about the normalised variant.

**FINDING: beam width 1 is not greedy decoding.** Beam-1 returns 1 token where
greedy returns n. Both keep one hypothesis; beam-1 compares completed hypotheses
against live ones at every step and greedy only asks which token is best next.
They differ in the **stopping rule**, not the width.

### 3 — Ten examples is not a measurement

`transformers`, `torch`, `datasets` and `sacrebleu` are all absent and there is
no paraphrase corpus, so no fine-tune is run and no BLEU from one is reported.
What is checkable is whether the comparison could resolve anything.

**ANSWER: it could not.** Percentile bootstrap over 2000 resamples of exercise
2's outputs:

| | greedy | beam | effect |
|---|---|---|---:|
| corpus BLEU (40 sentences) | 0.2507 | 0.2982 | **+0.0475** |
| 95% interval at n=10 | [0.2018, 0.3003] | — | width **0.0985** = 2.1× the effect |
| 95% interval at n=40 | [0.2263, 0.2750] | [0.2691, 0.3271] | width **0.0488** ≈ the effect |

At n=40 the two intervals **overlap**, so the point estimates ranking beam above
greedy are consistent with no difference at all. Quadrupling the set buys the
root-n factor of two, so resolving this effect would take hundreds of sentences —
not ten.

**FINDING: "BLEU" is a family of numbers.** The same outputs against the same
references, changing only the n-gram order:

| n-gram order | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| greedy | 0.1769 | 0.1966 | 0.2234 | 0.2507 |
| beam | 0.3538 | 0.2780 | 0.2815 | 0.2982 |

A spread of **0.0738** against a between-system difference of 0.0475. Reporting
"BLEU" without the order, the smoothing and the tokenisation names one member of
a family.

**CONTROL: only the sign survives.** Beam is at or above greedy at every order,
so the direction of the comparison is stable even though its magnitude is not —
which is an argument for the ten qualitative examples and against the BLEU
printed beside them.

### A note on file lengths

The three files run 122 / 144 / 111 lines of code; the first two are over D14's
120-line target and under its 150-line ceiling. The larger overrun is the decoders: the
exercise asks for beam search against greedy and neither ships, so the file
carries both, a fourth arm for length normalisation, and a smoothed BLEU with a
brevity penalty because sacrebleu is not installed. Exercise 3 imports all of
that from exercise 2 rather than rebuilding it.
