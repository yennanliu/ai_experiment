<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 13-question-answering

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/13-question-answering/) · upstream spec
`phases/05-nlp-foundations-to-advanced/13-question-answering/docs/en.md`

```bash
uv run demo practice run 13-question-answering --ex 1
uv run demo explain 13-question-answering --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/13-question-answering
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Set up the SQuAD extractive pipeline above on 10 Wikipedia passages. Hand-craft 10 ques… | code | T0 | `ex01_two_answers_made_of_articles_match_exactly.py` |
| 2 | Medium. Add a refusal classifier. When the top retrieval score is below a threshold (say 0.3… | code | T0 | `ex02_the_threshold_cannot_separate_overlapping_scores.py` |
| 3 | Hard. Build a RAG pipeline over a 10,000-document corpus of your choice. Implement hybrid ret… | code | T1 | `ex03_rrf_tracks_the_better_arm_and_never_beats_it.py` |
<!-- generated:end -->

## Answers

Three exercises whose stated goals are met and whose supporting machinery is not
what it looks like. Exercise 1 lands inside its predicted band while its metrics
score two different answers as an exact match. Exercise 2's threshold is a cosine
compared against a BM25 sum. Exercise 3's fusion works and cannot beat its own
better input.

Exercises 1 and 2 are **T0**; exercise 3 is **T1** (scikit-learn for the dense
arm).

### 1 — Two answers made of articles match exactly

**ANSWER: exact match 8 of 10, mean token F1 0.800** — inside the 7–9 the
exercise predicts, on ten passages and ten hand-crafted questions.

**FINDING: the two failures are the retriever and the span rule.**
`In what year did Android launch?` retrieves the Macworld passage — overlap on
`year` and a date outweighs one mention of Android — and
`Where was the World Wide Web proposed?` retrieves correctly and the span rule
returns `The World`.

**MECHANISM: `normalize` strips articles before punctuation.**

| input | normalised |
|---|---|
| `a` | `''` |
| `the` | `''` |
| `A.` | `''` |
| `The Beatles` | `beatles` |

**FINDING: so any two answers made of articles are an exact match.**
`exact_match("a", "the")` = **1.0**, and so is every pair among
`a, an, the, The, A`. `token_f1("a", "the")` agrees at 1.0, because it returns
1.0 when both token lists are empty. A system answering `the` to everything is
graded correct on any gold answer that is also an article.

**FINDING: token F1 gives full credit for a permutation.**
`29 June 2007` against `June 29, 2007` scores EM **0** and F1 **1.000** — the
same 1.000 the identical string gets. The lesson calls F1 partial credit; on a
reordering the two metrics disagree completely rather than by a margin.

### 2 — The threshold cannot separate overlapping scores

**ANSWER: 0.3 is a cosine and the score is a BM25 sum reaching 2.4.**
`toy_bm25_score` sums `count / (1 + length/10)`, so on the ten passages it runs
**0.0 to 2.4000**. At 0.3 the rule fires only where the score is exactly zero —
a zero-overlap detector described as a confidence threshold.

**FINDING: tuning it is worth one question in sixteen.**

| threshold | accuracy |
|---|---:|
| 0.3 (the lesson's) | 0.7500 |
| **0.8696 (best)** | **0.8125** |
| 1.9231 | 0.7500 |

**MECHANISM: the two distributions overlap.** Answerable questions score
0.8696–2.4000, unanswerable ones 0.0–1.6667. **2 of 6** unanswerable score above
the lowest answerable and **4 of 10** answerable score below the highest
unanswerable. Tuning has a ceiling and reaches it immediately.

**FINDING: the ones that get through were written to share vocabulary.** The
three built from corpus terms score 0.8696, 1.3043, 1.6667; the three unrelated
ones score 0.0, 0.0, 0.8333. A lexical retriever scores a fluent question about
Apple and quantum computers like a real question, because lexically it is one.

### 3 — RRF tracks the better arm and never beats it

No 10,000-document corpus and no embedding model, so the corpus is exercise 1's
ten passages and the dense arm is TF-IDF reduced by truncated SVD.

**ANSWER: top-1 retrieval over the ten questions.**

| SVD width | BM25 | dense | RRF | hybrid gain |
|---:|---:|---:|---:|---:|
| 4 | 9 | 8 | 9 | **+0** |
| 8 | 9 | **10** | **10** | **+1** |

**MECHANISM: the fused score equals the maximum of its two inputs at both
widths** and exceeds it at neither. Fusion here is insurance against picking the
wrong retriever, not a third retriever that outperforms both.

**FINDING: the whole effect is one question.** At 8 components the arms disagree
about the top passage on **1 of 10**; RRF rescues that one and loses none. Where
they agree there is nothing to fuse.

**ANSWER: the question type that benefits is the one whose keywords sit in a
distractor.** The rescued question is `In what year did Android launch?` — the
one BM25 already got wrong, for the reason a lexical retriever is defined to get
things wrong.

**FINDING: at 4 components the dense arm is worse and rescues nothing.** The
benefit is contingent on the dense arm being good, not on the fusion being
clever. RRF loses no question at either width.
