<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 04-glove-fasttext-subword

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword/) · upstream spec
`phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword/docs/en.md`

```bash
uv run demo practice run 04-glove-fasttext-subword --ex 1
uv run demo explain 04-glove-fasttext-subword --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `char_ngrams("playing")` and `char_ngrams("played")`. Compute the Jaccard overlap o… | code | T0 | `ex01_six_grams_of_prefix_whatever_follows.py` |
| 2 | Medium. Extend `learn_bpe` to track vocabulary growth. Plot tokens-per-corpus-character as a… | code | T0 | `ex02_the_asymptote_is_the_mean_word_length.py` |
| 3 | Hard. Train a 1k-merge BPE on Shakespeare's complete works. Compare tokenization of common wo… | code | T0 | `ex03_seen_once_is_seen_never.py` |
<!-- generated:end -->

## Answers

Three exercises that each state a number, and none of the three numbers is what
the exercise says it is. "Substantial shared pieces" is a Jaccard of 0.1667.
"Asymptoting near ~2-3 chars per token" describes a point the curve passes
through at 8 merges on its way to 5.136. And the surprise exercise 3 asks you to
write up is that a proper noun *in* the training corpus tokenizes slightly worse
than one that never appears in it at all.

All three run at **T0** — the lesson's `code/main.py` imports only `collections`.

### 1 — Six grams of prefix, whatever follows

**ANSWER: Jaccard 0.1667 — six shared n-grams in a union of 36.**

| | |
|---|---|
| `char_ngrams("playing")` | 23 grams |
| `char_ngrams("played")` | 19 grams |
| shared | **6** — `<pl`, `<pla`, `<play`, `lay`, `pla`, `play` |
| union | 36 |

All three pieces the exercise names (`pla`, `lay`, `play`) are there, and they
are half of everything that is shared.

**CONTROL: against a null the signal is real.** `walked`, `banana` and
`xylophone` each share **0** n-grams with `playing`. `walked` is a regular past
tense exactly like `played` and overlaps it not at all — what `played` shares
with `playing` is the stem and nothing else. So 0.1667 against 0 is the whole
of the morphological signal, not a weak version of it.

**MECHANISM: the intersection is fixed by the common prefix.**

| word | shared grams | Jaccard | length |
|---|---:|---:|---:|
| `play` | **6** | 0.2222 | 4 |
| `played` | **6** | 0.1667 | 6 |
| `player` | **6** | 0.1667 | 6 |
| `playful` | **6** | 0.1500 | 7 |

An inflection, an agent noun and an adjective are scored identically, because
nothing after `<play` can enter the intersection. Only the *union* moves, and
the union is the other word's gram count — **sorting these four by Jaccard sorts
them shortest first**, which is not a morphological relation.

**FINDING: a shared stem does outscore a shared suffix.** `singing` and
`running`, which share only `-ing`, score 3 grams at Jaccard 0.0714 and 0.0698
against the stem group's 6. The boundary markers buy this: `<pl` and `<pla` are
only available to words that start the same way.

**CONTROL: the whole-word gram is a union nobody recovers.** `char_ngrams` seeds
its set with the wrapped word itself, so `<playing>` is in one set and never in
the other — for any two distinct words, +1 to the union and 0 to the
intersection.

### 2 — The asymptote is the mean word length

**ANSWER: the quantity to plot and the quantity to expect are reciprocals, and
they move in opposite directions.**

| merges | 0 | 1 | 2 | 4 | 6 | 8 | 10 | 15 | 20 | 30 | 50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tokens / corpus char | 1.195 | 1.071 | 0.956 | 0.743 | 0.584 | 0.487 | 0.407 | 0.257 | 0.195 | 0.195 | 0.195 |
| chars / token | 0.837 | 0.934 | 1.046 | 1.345 | 1.712 | **2.055** | **2.457** | 3.897 | 5.136 | 5.136 | 5.136 |

**MECHANISM: the asymptote is exact, and it is 5.136.** `learn_bpe` breaks when
no pair remains, so budgets 20, 30 and 50 all return **18** merges. At that point
all six corpus words tokenize to themselves — 22 tokens for 22 words — so chars
per token is `113 / 22`, the corpus's **mean word length**. A six-word corpus
does not learn subwords; it memorises words.

**FINDING: 2–3 chars per token is where the curve is steepest.** It is inside
`[2, 3]` only at 8 and 10 merges, which is where the per-step gains are largest.
The exercise's range is a place the curve goes through, not where it flattens.

**MECHANISM: at zero merges there are more tokens than characters.** 135 against
113, because `learn_bpe` appends `</w>` to every word — the starting count is
characters *plus* words. The first points of the "compression" curve are partly
the tokenizer undoing a marker it added.

### 3 — Seen once is seen never

Shakespeare's complete works are not in this checkout, so the corpus is this
phase's own 29 English lesson documents — **46 423 word tokens over 4879 types**,
read through `parity.doc_text` — which happens to contain `shakespeare` twice.
Three word lists are scored, not two, and the third answers the last sentence.

**ANSWER: mean tokens per word.**

| merges | common words | rare proper nouns (count 1–6) | unseen proper nouns (count 0) |
|---|---:|---:|---:|
| 0 | 6.933 | 8.000 | 8.100 |
| 100 | 3.600 | 6.000 | 5.300 |
| 300 | 1.800 | 4.800 | 4.900 |
| **1000** | **1.067** | **4.400** | **4.200** |

**FINDING — the surprise: a name seen twice costs more than a name never seen.**
`{mikolov: 1, pennington: 1, bojanowski: 1, salton: 1, firth: 1, markov: 1,
shakespeare: 2, imdb: 2, reuters: 2, penn: 6}` scores **4.400**; ten names with
count **0** score **4.200**. Being in the training corpus bought nothing, because
BPE merges by frequency and a pair seen once never outranks one seen hundreds of
times. `shakespeare`, present twice, comes out as
`['sh', 'a', 'ke', 'sp', 'e', 'are</w>']`.

**MECHANISM: the "before" number is not a fact about BPE.** At 0 merges every
word is its characters plus `</w>`, so tokens per word is mean word length **+
1** — exactly, on all three lists (6.933 / 8.000 / 8.100 against lengths 5.933 /
7.000 / 7.100). "Measure average tokens per word before and after" compares BPE
against an alphabet, and the before half is settled by how long the words are.

**FINDING: the common vocabulary saturates while the tail is still
fragmenting.** The last 700 merges move the common list by **0.733** and the rare
list by **0.400**, and common words end at 1.067 — one token each. Every merge
past ~300 is bought for a vocabulary that no longer needs it.

**CONTROL: the gap is BPE's, not the words'.** Mean word lengths span 1.167
across the three lists, and at 0 merges the token counts differ by the same
1.167. At 1000 merges they differ by **3.333**. `model` becomes `['model</w>']`
and `mikolov` becomes `['mi', 'k', 'o', 'lo', 'v</w>']` — which side of the
vocabulary a word lands on is decided by its corpus frequency, not its shape.
