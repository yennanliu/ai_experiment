<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 03-word-embeddings-word2vec

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec/) · upstream spec
`phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec/docs/en.md`

```bash
uv run demo practice run 03-word-embeddings-word2vec --ex 1
uv run demo explain 03-word-embeddings-word2vec --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run the training loop on a tiny corpus (20 sentences about cats and dogs). After 200 ep… | code | T1 | `ex01_a_nine_in_twelve_check.py` |
| 2 | Medium. Add subsampling of frequent words. Words with frequency above `10^-5` are dropped fro… | code | T1 | `ex02_the_threshold_leaves_nothing_to_measure.py` |
| 3 | Hard. Train a model on the 20 Newsgroups corpus. Compute two bias axes: `he - she` and `docto… | code | T1 | `ex03_a_probe_without_a_null.py` |
<!-- generated:end -->

## Answers

Three exercises about a real skip-gram trainer, and all three turn on the same
missing quantity: a run. Exercise 1 states a result that holds on 9 of 12 seeds.
Exercise 2 names a threshold under which nothing survives to be measured.
Exercise 3 asks for a per-word bias ranking that changes on every seed, on a
corpus built to have no bias at all.

All three are **T1** — numpy, no GPU, no key. They train the lesson's own
`train()` 61 times between them, which is where the ~60s runtime goes; the
sweeps are the measurement, not overhead.

### 1 — A nine-in-twelve check

**ANSWER: the check passes on 9 of 12 seeds at the epoch count the exercise
names.** 20 sentences, 43 word types, 264 skip-gram pairs, the lesson's own
hyperparameters.

| | 200 epochs | 400 epochs |
|---|---:|---:|
| `dog` in top 3, call **as the exercise writes it** | **9 / 12** | 11 / 12 |
| `dog` in top 3, call **as `main()` writes it** | **11 / 12** | 11 / 12 |
| nearest word to `cat` | `mouse` 7, `dog` 4, `purred` 1 | `mouse` 6, `dog` 6 |

The exercise states the result as a fact; its "if not, increase epochs or
vocabulary" clause is the only hint that it might not hold.

**MECHANISM: the exercise's own call wastes a slot the lesson's does not.**
`nearest(vocab, W, W[vocab["cat"]])` omits `exclude`, which the lesson's
`main()` passes on the identical call one file away. So `cat` itself takes slot
1 at similarity **1.0** on all 12 runs and only 2 of the 3 are left to win.
Scoring the same runs with the word excluded recovers 2 of the 3 failures.

**FINDING: more epochs buys back the wasted slot and does not move the
neighbourhood.** Doubling to 400 lifts the as-written score 9 → 11 and leaves
the self-excluded score at 11. The run that fails on its merits at 200 still
fails at 400 — the exercise's remedy fixes the call, not the embedding.

**FINDING: `mouse` wins the slot more often than `dog`, and that is the
objective working.** Skip-gram raises the probability of words that *co-occur*
with `cat`, and `mouse` does — `a cat chased a mouse`. `dog` shares the slots
`cat` fills instead. The exercise asks a co-occurrence model for a
substitutability judgement, and gets a co-occurrence answer.

**CONTROL: the negative sampler never runs short here, and is the wrong
distribution.** `rng.integers(0, vocab_size, size=k_neg * 2)` filtered to 5 came
up short **0 times in 20 000** draws, so the `[:k_neg]` truncation is safe at
this vocabulary size. But it is uniform over *types*: `the`, at 14 occurrences,
is drawn as a negative exactly as often as each of the 21 words that occur once,
where word2vec specifies a unigram^0.75 distribution.

### 2 — The threshold leaves nothing to measure

**ANSWER: at 10⁻⁵, no training pair survives.**

| | |
|---|---|
| corpus | **96 tokens**, 43 types |
| types above 10⁻⁵ | **43 / 43** — the rarest by a factor of **1042** |
| max keep probability `sqrt(t/f)` | **0.0310** |
| tokens surviving | **2 / 96** |
| training pairs surviving | **0 / 264** |

10⁻⁵ is word2vec's published threshold, chosen against a corpus of a billion
tokens. A threshold on a *frequency* is a statement about corpus size, and this
one selects the entire vocabulary while discarding almost every token of it.
"Measure the effect on rare-word similarity" has nothing left to measure.

**FINDING: raised three orders of magnitude, the effect is non-monotone.**

| | pairs kept | share of pairs touching a rare word | rare word's nearest is a true neighbour |
|---|---:|---:|---:|
| unsubsampled | 264 | 0.311 | 0.738 |
| t = 0.03 | 204 | 0.382 | **0.798** |
| t = 0.01 | 94 | **0.447** | **0.638** |

Subsampling helps, then hurts. And the obvious proxy — how much of the training
signal now touches rare words — **rises the whole way**, including past the
point where the outcome turns over. `t = 0.01` has the best rare-word coverage,
the worst rare-word embeddings, and drops one rare word out of the vocabulary
entirely (20 of 21 survive).

**FINDING: dropping tokens is not dropping pairs.** word2vec removes tokens,
which pulls distant words inside the window and mints pairs the corpus never
contained — **10** of them at t=0.03, including `mat`–`sat` and `dog`–`rug`.
Dropping pairs, as the exercise words it, leaves 178 against 204 and can create
nothing.

**CONTROL: "proportional to frequency" has no floor, and word2vec's rule is all
floor.** `1 - sqrt(t/f)` is clipped at zero, so at t=0.03 the most frequent word
is dropped with probability **0.546** and the rarest with **0.000** — never. A
rule proportional to frequency drops the rarest word with probability 0.071,
which is the opposite of what subsampling is for.

### 3 — A probe without a null

Two corpora are built, because a corpus written on purpose has a *known* true
bias. In the **null** corpus (64 sentences) every occupation appears with `he`
and with `she` the same number of times in the same frames — its gender bias is
zero by construction. In the **skewed** corpus (32 sentences) four occupations
take `he` three times in four and four take `she`.

**ANSWER: on the corpus with no bias in it, the probe reads 0.539.**

| | null corpus | skewed corpus |
|---|---:|---:|
| largest \|projection\| on `he − she` | **0.539** | 0.299 |
| spread *between* occupation means | 0.036 | — |
| spread of one occupation *across seeds* | **0.283** | 0.044 |
| he-weighted vs she-weighted group means | 0.011 apart | **0.148 apart** |
| occupations that topped the ranking | dancer, doctor, lawyer, teacher | chef, doctor, engineer |

**MECHANISM: the seed moves each occupation eight times further than the corpus
separates them.** Occupation means on the null corpus are
`[0.151, 0.154, 0.146, 0.181, 0.145, 0.146, 0.153, 0.156]` — a spread of 0.036
— against a per-occupation across-seed deviation averaging 0.283. Any ranking
read off a single run is reading the initialisation.

**FINDING: the ranking the exercise asks for is unstable on both corpora.**
"Which occupations have the largest bias gap" names a different occupation on
different seeds even on the corpus where the bias is real and was put there
deliberately. The four biased occupations are tied within noise of each other.

**FINDING: one level up, the probe works — and only when it should.** Averaging
the four `he`-weighted occupations against the four `she`-weighted ones gives a
gap of **0.148** on the skewed corpus and **0.011** on the null, with per-seed
deviation falling from 0.283 to 0.044. The *direction* is measurable; the
ranking is not.

**FINDING: the two axes are not consistently oriented.**
`cos(he − she, doctor − nurse)` across 5 seeds of the *same* corpus:
`[-0.343, -0.087, 0.323, 0.153, 0.009]`. It spans 0.666 and changes sign.
Reporting a word's position "on both axes" presumes a relation between them
that the runs do not hold.

**CONTROL: the noise on the unbiased corpus is 3.6× the signal on the biased
one.** 0.539 against a whole group gap of 0.148. No single number off a single
run is readable as bias without the null run beside it — and the null run is
the one the exercise does not ask for.

### A note on file lengths

The three files run 123 / 150 / 133 lines of code, over D14's 120-line target
and under its 150-line ceiling. The overrun is sweeps: exercise 1 runs 24
trainings across two epoch counts, exercise 2 runs 12 across three subsampling
arms and carries both the token-dropping and pair-dropping variants, and
exercise 3 runs 10 across two purpose-built corpora. Exercise 2 imports
exercise 1's `SENTENCES` via `practice.load_module` rather than carrying a
second copy of the corpus.
