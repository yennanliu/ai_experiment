<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 12-text-summarization

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/12-text-summarization/) · upstream spec
`phases/05-nlp-foundations-to-advanced/12-text-summarization/docs/en.md`

```bash
uv run demo practice run 12-text-summarization --ex 1
uv run demo explain 12-text-summarization --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/12-text-summarization
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run TextRank on 5 news articles. Compare the top-3 sentences to a reference summary. Me… | code | T0 | `ex01_recall_only_rouge_is_maximised_by_the_article.py` |
| 2 | Medium. Implement entity-level factuality: extract named entities from source and summary (sp… | code | T0 | `ex02_extractive_precision_is_one_by_construction.py` |
| 3 | Hard. Compare BART-large-CNN against an LLM (Claude or GPT-4) on 50 CNN/DailyMail articles. R… | code | T0 | `ex03_the_three_numbers_rank_three_different_ways.py` |
<!-- generated:end -->

## Answers

Three exercises whose metrics all reduce to the same instruction for the system
the lesson builds: emit more. Exercise 1's ROUGE is recall-only and is maximised
by returning the article. Exercise 2's entity precision is pinned at 1.0 by
construction, so only its recall moves — and its recall says what the ROUGE said.
Exercise 3 asks for three numbers that give three different orderings.

All three run at **T0**: `code/main.py` imports only `math`, `re` and
`collections`.

### 1 — Recall-only ROUGE is maximised by the article

Five (article, reference) pairs, seven sentences each.

| k | 1 | 2 | 3 | 4 | 5 | 7 |
|---|---:|---:|---:|---:|---:|---:|
| `rouge_n` (recall) | 0.1322 | 0.2453 | 0.3348 | 0.4513 | 0.5272 | **0.6584** |
| ROUGE-L (LCS F) | 0.1674 | 0.2410 | 0.2941 | 0.3418 | 0.3577 | 0.3589 |

**ANSWER: `rouge_n` rises with every sentence added and peaks on the whole
article.** It is recall against the reference with no precision term. Ranking
summarisers on it ranks the one that summarises least.

**MECHANISM: ROUGE-L is not in the lesson.** `code/main.py` ships `rouge_n`
only. The LCS F-measure has to be written first — and the precision term is the
whole difference between them.

**FINDING: with a precision term the curve stops paying for length.** From k=5 to
the whole article, `rouge_n` gains **+24.9%** and ROUGE-L gains **+0.3%**.
Per-article ROUGE-L peaks at k = [4, 4, 7, 7, 7] — **2 of 5** strictly before the
end, where `rouge_n` peaks at the article on all five because it cannot do
otherwise.

**FINDING: the exercise's own number belongs to a different k.** The top-3 it
specifies scores ROUGE-L **0.2941**, just under the 30–45 band it predicts. The
band opens at k=4, **0.3418**.

### 2 — Extractive precision is one by construction

**ANSWER: precision is 1.0000 at every k, on every article** — checked as set
inclusion over the whole sweep, not as an average. `textrank` copies sentences,
so its entities are a subset of the source's. Half the metric the exercise
defines cannot move.

**MECHANISM: the exercise's reading of that number fits every extractive
summary.** "High precision and low recall mean safe but terse" — the one-sentence
summary scores precision 1.0 / recall 0.0, and the *whole article* scores
precision 1.0 / recall 1.0. The same diagnosis for the tersest output and for no
summarisation at all.

**FINDING: recall is the informative half, and it repeats exercise 1.**

| k | 1 | 2 | 3 | 4 | 5 | 7 |
|---|---:|---:|---:|---:|---:|---:|
| entity recall | 0.1067 | 0.2800 | 0.4367 | 0.4867 | 0.6767 | **1.0000** |

**FINDING: the signal the exercise describes needs an abstractive system.**
Swapping three entities — `MIT`, `2 billion`, `GitLab` for `Canadian`, `1`,
`GitHub` — takes precision from 1.0000 to **0.0000**, separating that summary
from every extractive one perfectly. The metric is *inapplicable* to the system
this lesson builds, not uninformative.

### 3 — The three numbers rank three different ways

Three buildable systems — `textrank` at k=2, at k=5, and a hand-written
abstractive stand-in — on the same five articles.

| system | ROUGE-L | entity F1 | mean output words |
|---|---:|---:|---:|
| abstractive | **0.3985** | 0.5508 | **13.2** |
| extractive k=5 | 0.3577 | **best** | 45.0 |
| extractive k=2 | 0.2410 | worst | 19.0 |

**ANSWER: three distinct orderings of the same nine summaries.**

| axis | order |
|---|---|
| ROUGE-L | abstractive, k=5, k=2 |
| entity F1 | k=5, abstractive, k=2 |
| cost | abstractive, k=2, k=5 |

**MECHANISM: entity F1 ranks the extractive systems on recall alone.** Both score
precision 1.0000, so the gap is entirely recall — 0.2800 against 0.6767. The
longer summary wins because it is longer, which is why it won the last exercise
too.

**FINDING: the two extractive systems swap places between ROUGE-L and cost.**
Between the two systems the lesson actually builds, those two axes give opposite
answers.

**MECHANISM: first places by axis are k=2: 0, k=5: 1, abstractive: 2.** No system
tops all three and one tops none. "Document where each wins" is answered by
saying the exercise's own three numbers do not agree on a winner.

**CONTROL: none of the three can see a false statement that keeps its entities.**
A summary that reversed a claim while reusing every name would score 1.0 on
factuality, exactly like the copied sentences.

### A note on file lengths

The three files run 149 / 117 / 126 lines of code, the first and third over D14's 120-line target
under its 150-line ceiling. The overrun is the five articles the exercise asks
for, with their reference summaries, plus a ROUGE-L the lesson does not ship and
an entity extractor standing in for spaCy. Exercises 2 and 3 import the corpus,
the metric and the extractor from their siblings rather than repeating them.
