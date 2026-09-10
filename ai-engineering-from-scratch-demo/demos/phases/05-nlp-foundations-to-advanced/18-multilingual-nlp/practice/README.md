<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 18-multilingual-nlp

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/18-multilingual-nlp/) · upstream spec
`phases/05-nlp-foundations-to-advanced/18-multilingual-nlp/docs/en.md`

```bash
uv run demo practice run 18-multilingual-nlp --ex 1
uv run demo explain 18-multilingual-nlp --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/18-multilingual-nlp
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run the zero-shot classification pipeline on 10 sentences per language across English,… | code | T0 | `ex01_the_toy_cannot_tell_hindi_from_arabic.py` |
| 2 | Medium. Use `paraphrase-multilingual-MiniLM-L12-v2` to build a cross-lingual retriever over a… | code | T0 | `ex02_a_small_lexicon_does_what_the_encoder_would.py` |
| 3 | Hard. Compare English-source and Hindi-source fine-tuning for a Hindi classification task. Us… | code | T0 | `ex03_the_answer_is_a_lookup_and_the_ties_decide_it.py` |
<!-- generated:end -->

## Answers

None of the three exercises can be run as written: `transformers`, `torch`,
`sentence_transformers` and `sentencepiece` are all absent, and the lesson ships
no classifier and no retriever. What it ships is a three-feature typology table
and two functions over it. Exercises 1 and 3 are questions that table already
answers — and its answers contradict exercise 1 and trivialise exercise 3.
Exercise 2's number is reachable with no cross-lingual resource at all.

All three run at **T0** — `code/main.py` imports nothing outside the stdlib.

### 1 — The toy cannot tell Hindi from Arabic

The exercise predicts strong French, decent Hindi, variable Arabic. The lesson's
own model, for an English source:

| target | shared features | similarity | simulated accuracy |
|---|---:|---:|---:|
| french | 2 / 3 | 0.6667 | 0.7500 |
| **hindi** | 0 / 3 | **0.0000** | **0.4500** |
| **arabic** | 0 / 3 | **0.0000** | **0.4500** |

**ANSWER: Hindi and Arabic get the same number.** The strong-French half of the
prediction holds; the half that distinguishes Hindi from Arabic is the half the
model has no resolution for.

**MECHANISM: three features admit four values, and the second function is the
first rescaled.** `similarity` counts matches over 3 features, so its range is
{0, 0.3333, 0.6667, 1.0}, and `simulate_transfer_accuracy` maps those one-to-one
onto {0.45, 0.60, 0.75, 0.90}. Calling the second a simulated accuracy adds a
unit and no information.

**FINDING: eleven languages collapse into seven profiles.** Five pairs score
similarity 1.0 — english/german, french/spanish, french/italian, spanish/italian,
hindi/marathi. Under this table Hindi and Marathi are the same language.

**FINDING: similarity is symmetric and transfer is not.** `similarity(a, b) ==
similarity(b, a)` for every pair, so the model says English→Hindi transfer works
exactly as well as Hindi→English. No feature-match count can represent the
asymmetry that zero-shot transfer actually has.

**CONTROL: the lesson names this limitation and then draws a conclusion from it
anyway** — its closing note says real similarity needs qWALS or lang2vec, and its
next line concludes Hindi beats English as a source for Marathi.

### 2 — A small lexicon does what the encoder would

12 concepts written in French and Spanish — 24 documents — queried by their 12
English originals, ranked by Jaccard overlap on tokens.

| retriever | recall@5 |
|---|---:|
| random (exact, by complement) | 0.3804 |
| **Jaccard overlap, query as written** | **0.5833** |
| Jaccard overlap, query expanded through a 27-entry lexicon | **1.0000** |

**ANSWER: 0.5833 with no cross-lingual resource at all.** A retriever that has
never seen a second language scores well above chance on a cross-lingual task.

**MECHANISM: it scores that on cognates, and the split is total.** Counting only
content words, the 7 concepts sharing a token with a translation score **1.0000**
and the 5 sharing none score **0.0000** — no middle, at a mean of 0.31 shared
content types per parallel pair. A lexical retriever is a cross-lingual retriever
over the vocabulary the languages happen to share, and nothing outside it.

Filtering function words is what makes that visible. The shared token in several
of these pairs is `a` — the English article and the French verb — so an
unfiltered overlap count reports a cognate where there is a coincidence, and the
isolated group shrinks from 5 concepts to 1.

**FINDING: 27 dictionary entries take it to 1.0000**, including on every concept
with no shared content word. The missing resource costs 27 lines.

**MECHANISM: so the measurement would not distinguish the encoder from the
lexicon.** Recall@5 over 24 documents saturates before it could separate them —
worth knowing before crediting the encoder with the result.

**CONTROL: the corpus is ASCII**, so the overlap carrying the lexical arm is
vocabulary overlap and not a normalisation artefact.

### 3 — The answer is a lookup, and the ties decide it

**ANSWER: 0.4500, and it comes from a dictionary lookup rather than a fine-tune.**
For a Hindi target the table gives Marathi similarity 1.0 and accuracy 0.90
against English at 0.0 and 0.45. `simulate_transfer_accuracy(target, source)`
takes two arguments, neither of them the 500 examples, so the number does not
move when they are supplied.

**MECHANISM: that is the LANGRANK thesis implemented, not a shortcut around it.**
The thesis is that source choice can be *predicted* from typological features
instead of measured; the lesson implements the prediction. The honest answer to
"report which source and by how much" is to read the table.

**FINDING: underneath it the ranking is mostly ties.** Sources tying for first,
per target:

| target | tied for best | target | tied for best |
|---|---:|---|---:|
| english | 1 | marathi | 1 |
| german | 1 | bengali | 3 |
| french | 2 | urdu | 3 |
| spanish | 2 | arabic | 1 |
| italian | 2 | **japanese** | **4** |
| hindi | 1 | | |

**MECHANISM: and the tie-break is the order of the candidate list.**
`rank_source_languages` uses a stable sort, so equal scores keep their input
order. Reversing the candidate list changes the recommended source for six of the
eleven targets — french, spanish, italian, bengali, urdu, japanese — with no
feature changed.

**FINDING: for this target the recommendation restates the table's own blind
spot.** Marathi scores 1.0 against Hindi, meaning the feature set finds no
difference between them at all. "Marathi is the best source for Hindi" and "this
table cannot tell Hindi from Marathi" are the same statement.

**CONTROL: the English comparison is the one the table can make confidently** —
English shares none of the three features with Hindi, so it sits at the bottom of
the scale with no tie to break. The trustworthy half of the exercise's comparison
is the half where the answer was obvious.
