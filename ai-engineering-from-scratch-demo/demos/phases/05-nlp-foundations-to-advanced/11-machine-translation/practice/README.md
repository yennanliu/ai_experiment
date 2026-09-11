<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 11-machine-translation

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/11-machine-translation/) · upstream spec
`phases/05-nlp-foundations-to-advanced/11-machine-translation/docs/en.md`

```bash
uv run demo practice run 11-machine-translation --ex 1
uv run demo explain 11-machine-translation --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/11-machine-translation
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Translate a 5-sentence English paragraph to French and back to English using `nllb-200-… | code | T0 | `ex01_bleu_scores_a_paraphrase_like_a_wrong_language.py` |
| 2 | Medium. Implement a language-ID check on translation outputs using `fasttext lid.176` or `lan… | code | T0 | `ex02_the_guard_is_a_coin_flip_on_short_output.py` |
| 3 | Hard. Fine-tune `nllb-200-distilled-600M` on a 5,000-pair domain corpus of your choice. Measu… | code | T0 | `ex03_the_per_sentence_table_is_undefined_for_short_sentences.py` |
<!-- generated:end -->

## Answers

No translation model is installed and none is downloadable, so all three
exercises are answered against the two metrics the lesson ships — which turns
out to be where the exercises actually live. Exercise 1 asks for a measurement
BLEU cannot make. Exercise 2's guard works on ordinary output and fails on
exactly the output that needs it. Exercise 3's per-sentence table is undefined
for a fifth of any realistic corpus.

All three run at **T0**: `code/main.py` imports only `math` and `collections`.

### 1 — BLEU scores a paraphrase like a wrong language

Five round trips written with the word-choice drift the exercise predicts, plus
three controls:

| kind | BLEU | chrF |
|---|---:|---:|
| drift — *accepted / additional* | **0.0** | 72.1 |
| drift — *got to / just before / left* | **0.0** | 52.1 |
| drift — *released / March the first* | **0.0** | 62.0 |
| drift — *postponed* (one word) | 66.1 | 80.5 |
| drift — *gathered reporters* (one word) | 61.0 | 63.0 |
| identical | 100.0 | 100.0 |
| **off-target (French)** | **0.0** | 40.9 |
| **unrelated (bananas)** | **0.0** | 16.9 |

**ANSWER: five of the eight pairs score exactly 0.0**, and three of them are the
round trips the exercise expects to preserve meaning. BLEU as the lesson ships it
gives one number to a correct paraphrase, a translation into the wrong language,
and a sentence with no relation to the source.

**FINDING: chrF separates the groups with no overlap** — drift 52.1–80.5 against
40.9 and 16.9. Every good round trip above every bad one, from the other function
in the same file.

**MECHANISM: the lesson says why, next door.** `simple_bleu_note()` states there
is no smoothing, so one zero-precision order takes the geometric mean — and
word-choice drift destroys 4-grams first, while unigram and bigram precision stay
high.

**FINDING: the two round trips BLEU does score changed one word each.** What
survives the metric is word order, not meaning.

### 2 — The guard is a coin flip on short output

`fasttext`, `langdetect`, `langid` and `pycld3` are all absent, so the identifier
is character-trigram profiles over four short samples.

**ANSWER: at full length it is right on all 20 held-out sentences**, mean margin
0.8619 over the runner-up. A guard on ordinary MT output is sound.

**MECHANISM: it degrades with output length, fast.**

| prefix (chars) | 4 | 8 | 16 | 32 | full |
|---|---:|---:|---:|---:|---:|
| accuracy | **0.5000** | 0.5500 | 0.8000 | 1.0000 | 1.0000 |
| mean margin | — | 0.4762 | — | — | 0.8619 |

Chance is 0.2500.

**FINDING: the shortest outputs are the ones most likely to need the guard.** A
model that has gone off target emits a copied source fragment, a repeated token,
an empty string — and at 8 characters the check is at 0.55.

**FINDING: the cost is symmetric.** At 8 characters it is wrong on 9 of 20
sentences, and **3 of those are correct English outputs** — 3 of the 5 English
sentences in the set. Integrated without a length floor, this guard rejects most
good short translations to catch the bad ones.

**MECHANISM: the margin is the signal the exercise has no slot for.** The
identifier knows when it is guessing; "caught before returning" is a two-way
decision and this is a three-way one — pass, reject, or decline to answer.

**CONTROL: an empty generation has no trigrams and no answer at all.** The
clearest off-target failure there is, and the one input a language check cannot
read.

### 3 — The per-sentence table is undefined for short sentences

**ANSWER: a perfect translation of a two-token sentence scores BLEU 0.0.**
Scored against itself, `Yes.` gets `simple_bleu` **0.0** and `chrf` **100.0**.

**MECHANISM: the threshold is exact and equals `max_n`.** Measured by search, the
shortest reference length that can score above zero is **4** — a sentence with
fewer than four tokens has no 4-grams, `ngram_precision` returns 0.0, and the
unsmoothed geometric mean collapses.

**FINDING: a fifth of a realistic corpus has no before-number and no
after-number.** Over 20 references spanning 2 to 13 tokens, **4** score 0.0 at
their own ceiling. "Which kinds of sentences improved and which regressed" has no
row for any of them.

**FINDING: above the threshold it is still unreadable in places.** Three of
exercise 1's five round trips score 0.0 while preserving meaning — their entry
would read 0.0 before and 0.0 after whatever the fine-tune did.

**MECHANISM: chrF has no floor, and it is in the same file.** The zeroed
sentences all score 100.0 at their ceiling and the drifted round trips 52.1–80.5.
Character n-grams do not run out on a two-token sentence.

**CONTROL: the split is by length and by nothing else.** Every reference of 4
tokens or more scores 100.0 at its ceiling and every shorter one scores 0.0 — so
a before-and-after comparison on this metric covers the long half of the corpus
while being reported as though it covered all of it.
