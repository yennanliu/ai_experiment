<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 02-bag-of-words-tfidf

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/02-bag-of-words-tfidf/) · upstream spec
`phases/05-nlp-foundations-to-advanced/02-bag-of-words-tfidf/docs/en.md`

```bash
uv run demo practice run 02-bag-of-words-tfidf --ex 1
uv run demo explain 02-bag-of-words-tfidf --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/02-bag-of-words-tfidf
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Implement `cosine_similarity(doc_vec_a, doc_vec_b)` on the L2-normalized TF-IDF output.… | code | T0 | `ex01_identical_documents_rarely_score_one.py` |
| 2 | Medium. Add `n-gram` support to `bag_of_words`. Parameter `n` produces counts over `n`-grams.… | code | T0 | `ex02_the_vocabulary_has_to_change_too.py` |
| 3 | Hard. Build the TF-IDF-weighted-embedding hybrid above using GloVe 100d vectors (download onc… | code | T1 | `ex03_the_comparison_is_a_question_about_e.py` |
<!-- generated:end -->

## Answers

Three exercises about a pipeline of eight small functions, and each one turns on
a function the exercise does not mention. Exercise 1 asks you to implement
`cosine_similarity`, which the lesson already ships — as a dot product, correct
only because `main()` normalizes first. Exercise 2 asks you to add n-grams to
`bag_of_words`, which cannot hold n-grams unless `build_vocab` produces them.
Exercise 3 asks which of three representations wins, and the answer is decided
by the embedding matrix, which is the one input that will not download.

Exercises 1 and 2 run at **T0** on the standard library; exercise 3 is **T1**
(numpy for the rank and the sweep).

### 1 — Identical documents rarely score exactly 1.0

**ANSWER: `s == 1.0` holds in 21 of 120 corpora.** Over every 3-subset of a
fixed 10-document pool, each corpus carrying a duplicated document:

| implementation | input | exactly 1.0 | worst \|s−1\| |
|---|---|---:|---:|
| the lesson's dot product | L2-normalized | **21 / 120** | 3.33e-16 |
| the lesson's dot product | raw `tfidf()` | 0 / 120 | **0.723** |
| dot / (‖a‖ ‖b‖) | L2-normalized | **88 / 120** | 2.22e-16 |
| dot / (‖a‖ ‖b‖) | raw `tfidf()` | 11 / 120 | 2.22e-16 |

The other 99 land one or two ULPs short. The exercise's first check has to be
written as a tolerance, or it fails five times in six **on correct code**.

**MECHANISM: the shipped function is a dot product, so it is not
scale-invariant.** `sum(x * y for x, y in zip(a, b))` earns the name *cosine*
only on the pre-normalized input the exercise stipulates. Handed the lesson's
own un-normalized `tfidf()` output — one call earlier in the same file — it is
off by up to **0.723** on an identical pair, while the two-norm version stays
inside 2.22e-16. Normalizing first is a precondition, not a habit.

**FINDING: neither version is exact everywhere.** The two-norm form is exact
4× as often because it divides a sum of squares by `sqrt(S)·sqrt(S)` rather
than trusting `l2_normalize` to have landed on 1 — but 88/120 is not 120/120,
and no float cosine is.

**FINDING: "disjoint scores 0.0" is exactly true, and true because of the
smoothing.** `log((n+1)/(df+1)) + 1` bottoms out at **1.0** for a term present
in every document, so no weight is ever zero and every entry is non-negative.
A zero dot product then means *no shared index and nothing else*: all **109**
zero pairs across the 120 corpora are token-disjoint.

**CONTROL: under the textbook idf the same claim is false 368 times.** Swap in
`log(n/df)`, which does zero a universal term, and **368 of 477** zero pairs
share tokens — two documents agreeing on nothing but stopwords also score 0.0.
The property the exercise says to verify belongs to the smoothing, not to
cosine similarity.

**FINDING: two silent wrong answers.** A pair of *identical empty documents*
scores **0**, not 1.0 — their rows are all-zero and `l2_normalize` maps a zero
norm to zeros. And `zip` truncates, so vectors of width **4** and **8** built
from two different corpora return **0.632456** instead of raising.

### 2 — The vocabulary has to change too

**ANSWER: the named test passes.** A length-`n` sliding window gives
`["the cat", "cat sat"]` for `n=2` on `["the", "cat", "sat"]`, and `n=1`
reproduces the lesson's own token list unchanged — the unigram path is a
special case, not a second branch. On the lesson's three documents the bigram
vocabulary is 11 entries against 9 unigrams.

**MECHANISM: changing `bag_of_words` alone returns an all-zero matrix,
silently.** `bag_of_words(docs, vocab)` takes its columns from `build_vocab`,
and its only handling of an unknown token is `if token in vocab`. Feed it
bigrams against the unigram vocabulary and **all counts land nowhere**: every
row is `[0, 0, 0, 0, 0, 0, 0, 0, 0]`, with no error raised.

**FINDING: the zeros survive the whole pipeline and break exercise 1's
property.** `tfidf` divides by a row sum of 0 and returns zeros, `l2_normalize`
maps a zero norm to zeros, and `cosine_similarity` returns **0** — a document
scoring 0 against *itself*. A caller checking exercise 1's stated property is
the only thing that would catch this.

**FINDING: counting n-grams instead of adding them deletes every single-word
match.**

| | d0·d1 | d0·d2 | d1·d2 |
|---|---:|---:|---:|
| n=1 | 0.7998 | 0.4595 | **0.3078** |
| n=2 | 0.4932 | 0.1485 | **0.0** |

d1 and d2 share `the` and `on` and no bigram at all.

**FINDING: past n=4 nothing is shared.**

| n | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| vocabulary size | 9 | **11** | 10 | 8 | 6 |
| n-grams in >1 document | 5 | 4 | 2 | 1 | **0** |

The vocabulary peaks at n=2 and the useful part falls faster. At n=5 every
off-diagonal similarity is 0 and the representation ranks nothing.

**CONTROL: a document shorter than `n` produces no n-grams and no error
either.** A one-token document windowed at n=2 gives `[]`, so its row is
all-zero — the same degenerate vector as the vocabulary mismatch, reached a
different way. `range(len(doc) - n + 1)` is where that decision is made.

### 3 — The comparison is a question about E

No GloVe loader is installed (`gensim`, `torchtext`) and
`fetch_20newsgroups(download_if_missing=False)` raises `OSError`, so the corpus
is 16 two-class documents over a 40-word vocabulary and E is drawn from a
seeded generator. Two of the three findings hold for **every** E; the third is
that the rest of the answer holds for no particular E.

**ANSWER: plain TF-IDF wins outright.** Leave-one-out nearest centroid by
cosine — deterministic, so every difference between arms is the representation:

| representation | d=8 | d=100 |
|---|---|---|
| TF-IDF | **1.000** | **1.000** |
| mean-pooled | 0.544 (0.125–0.812) | 0.731 (0.375–1.000) |
| TF-IDF-weighted hybrid | 0.572 (0.188–0.812) | 0.841 (0.562–1.000) |

**MECHANISM: mean-pooling *is* the hybrid with idf set to 1.** Pooling is
`weights @ E` rescaled by the weight sum; mean-pooling weights by raw counts
and the hybrid by `tf*idf`, so the two arms the exercise contrasts differ in
exactly one vector — they agree to **5.55e-16** when idf is flattened. The
experiment is an ablation of idf, not a comparison of two methods.

**MECHANISM: both pooled arms are rank-limited by d, exactly.** A fixed E makes
the pooled matrix a linear map of the TF-IDF matrix, so `rank(pooled) =
min(d, rank(TF-IDF))`: **8** at d=8 and **16** at d=100, against **16** for
TF-IDF itself. At d=8 the projection discards half the corpus before any
classifier sees it.

**FINDING: which pooled arm wins is decided by E.** Over 20 seeds at d=8 the
hybrid beats mean-pooling **8** times, ties **7**, loses **5**. At d=100 it is
15 / 5 / 0. "Report which wins where" is a question about the embedding matrix.

**FINDING: the spread across seeds dwarfs the gap between the arms.** At d=8
mean-pooling scores anywhere from **0.125 to 0.812** on the seed alone — a
spread of 0.688 — against a **0.028** mean gap to the hybrid. A single-seed
comparison of these two arms measures the draw, not the method.

**CONTROL: raising d closes the gap to TF-IDF and never crosses it.** d=8 to
d=100 lifts the hybrid from 0.572 to 0.841 mean, best case **1.000** — level
with TF-IDF, never above. A projection of a representation cannot carry more
than the representation.

### A note on file lengths

The three files run 140 / 101 / 150 lines of code — the first and third over D14's 120-line target
and at its 150-line ceiling. The overrun is fixture and arms: exercise 1
sweeps four implementations over 120 corpora plus an unsmoothed-idf ablation,
and exercise 3 carries the two-class corpus, the leave-one-out classifier and a
20-seed sweep at two embedding widths, none of which the lesson ships.
