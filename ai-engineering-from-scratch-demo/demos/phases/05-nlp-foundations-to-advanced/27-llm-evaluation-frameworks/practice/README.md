<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 27-llm-evaluation-frameworks

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/27-llm-evaluation-frameworks/) · upstream spec
`phases/05-nlp-foundations-to-advanced/27-llm-evaluation-frameworks/docs/en.md`

```bash
uv run demo practice run 27-llm-evaluation-frameworks --ex 1
uv run demo explain 27-llm-evaluation-frameworks --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/27-llm-evaluation-frameworks
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Use RAGAS on 10 RAG examples with known hallucinations. Verify the faithfulness metric… | code | T0 | `ex01_faithfulness_catches_none_of_the_ten.py` |
| 2 | Medium. Hand-label 50 QA answers 0-1 for correctness. Score with G-Eval. Measure Spearman rho… | code | T1 | `ex02_the_judge_measures_brevity_not_correctness.py` |
| 3 | Hard. Build a pytest CI gate with DeepEval. Intentionally regress the retriever. Verify the g… | code | T0 | `ex03_three_of_four_metrics_cannot_see_the_regression.py` |
<!-- generated:end -->

## Answers

All three exercises name a framework that is not installed — RAGAS, DeepEval,
G-Eval — so each falls back to the stdlib approximation `code/main.py` ships as
the stand-in. The findings are what those approximations do when you actually run
them: the hallucination detector flags nothing, the judge is uncorrelated with
human labels, and the CI gate returns the same verdict before and after the
regression it is supposed to catch.

Exercise 1 and 3 run at **T0** — `code/main.py` imports only `re` and
`collections`. Exercise 2 runs at **T1**: it needs `scipy.stats.spearmanr`, which
the exercise asks for by name.

### 1 — Faithfulness catches none of the ten

Ten examples: a context sentence, the faithful answer, and the same answer with
one content token replaced by a false one.

| answer set | n | faithfulness | caught at the doc's 0.85 gate |
|---|---:|---:|---:|
| hallucinated (one token corrupted) | 10 | **1.00** | **0 / 10** |
| faithful | 10 | **1.00** | — |

**ANSWER: `ragas` is absent** (`importlib.util.find_spec('ragas')` is None), so
the substitute is the lesson's own `faithfulness`. It scores every hallucinated
answer and every faithful answer identically. No threshold in [0, 1] separates the
two sets.

**MECHANISM: a hallucination is one token and the bar is half the sentence.**
`faithfulness` supports a claim when ≥ 0.5 of its non-stopword tokens appear in
the context.

| quantity | value |
|---|---:|
| mean supported-token fraction, corrupted claims | 0.8404 |
| minimum | 0.8000 |
| margin over the 0.5 threshold, worst case | 0.3000 |
| mean move per corrupted token (1 / sentence length) | 0.1596 |

The closest case is two further corrupted tokens away from being flagged.

**FINDING: `main()`'s transcript contradicts `main()`'s code.** It prints
`case 1 = hallucinated date -> g-eval drops, faithfulness partial`. Case 1's
faithfulness is **1.00**.

**FINDING: the sign is inverted.** "Marie Curie did not win the Nobel Prize in
Physics in 1903", against a context saying she won, scores **1.00** — `not` is one
token of nine and is not in `STOP`. A true statement in different words scores
**0.00**. The metric rewards copying and punishes paraphrase, which is the failure
the lesson opens by promising to fix.

**CONTROL: a constant `1.0` scorer is indistinguishable from the metric** — over
all 20 answers it takes `[1.0]` as its only value. And the doc's Step 1 signature
`faithfulness(answer, context, llm)` raises
`TypeError: faithfulness() takes 2 positional arguments but 3 were given`.

### 2 — The judge measures brevity, not correctness

50 invented answers: 10 facts × 5 answer styles, hand-labelled 1/1/1/0/0 for 30
correct against 20 incorrect.

| answer style | human label | mean judge score |
|---|---:|---:|
| bare gold string | 1 | **1.00** |
| correct full sentence | 1 | 0.30 |
| correct paraphrase | 1 | **0.00** |
| wrong value, bare | 0 | **0.80** |
| wrong value, full sentence | 0 | 0.00 |

| correlation (Spearman) | rho | p |
|---|---:|---:|
| judge vs human | **0.0331** | 0.8196 |
| `expected in actual` baseline vs human | **0.6667** | — |
| judge vs answer brevity | 0.5762 | — |

**ANSWER: rho = 0.0331 at p = 0.82.** `deepeval` is absent, so `GEval` cannot be
constructed and the substitute is `g_eval_correctness`. The lesson's own bar is
"if rho < 0.7, your judge rubric needs work"; this judge is not distinguishable
from a coin.

**MECHANISM: the denominator is the answer, not the gold.**
`g_eval_correctness` scores the fraction of the *actual* answer's tokens that
appear in the *expected* string, and the expected string is 3 tokens. Adding true,
relevant words to a correct answer lowers its score; deleting everything except a
one-digit-wrong value raises it to 1.0.

**FINDING: it tracks brevity better than correctness** — 0.5762 against 0.0331.
That is the inverse of the length bias `docs/en.md` warns about under "Judge
bias", and the thing it measures best.

**FINDING: a one-line baseline the exercise never computes beats it.**
`expected.lower() in actual.lower()` scores **0.6667** — twenty times the judge's
correlation, with no rubric, no threshold and no model.

**CONTROL: no threshold makes it calibrated.**

| threshold | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rho | −0.1117 | 0.0000 | 0.0833 | 0.0816 | 0.0331 | 0.2364 | 0.4082 | 0.4082 | 0.4082 | 0.4082 |

The best reachable value is 0.4082, still short of 0.7, and the shipped default of
0.5 is not the problem. The judge's range is the two-point grid `{0.0, 1.0}`
because every answer is one sentence. And the shipped judge never sees the
question, though Step 3's `GEval` passes `evaluation_params` including `INPUT`.

### 3 — Three of the four metrics cannot see the regression

The gate is the write-up's own Step 4: faithfulness ≥ 0.85, relevancy ≥ 0.7. The
regression injects k irrelevant chunks alongside the gold chunk, over three QA
cases.

| k irrelevant chunks | faithfulness | relevance | precision | recall | gate verdict |
|---:|---:|---:|---:|---:|---|
| 0 | 1.0000 | 0.3810 | 1.0000 | 1.0000 | `['relevancy']` |
| 1 | 1.0000 | 0.3810 | 0.5000 | 1.0000 | `['relevancy']` |
| 3 | 1.0000 | 0.3810 | 0.2500 | 1.0000 | `['relevancy']` |
| 5 | 1.0000 | 0.3810 | **0.1667** | 1.0000 | `['relevancy']` |

**ANSWER: the gate fails identically before and after the regression.**
`deepeval` is absent, so `FaithfulnessMetric` and `ContextualRelevancyMetric`
cannot be built. The gate reports `['relevancy']` on the healthy retriever and
`['relevancy']` after the regression, for the same reason both times. "Verify the
gate fails" is satisfied by a gate that was already failing and that the
regression never moves.

**MECHANISM: three of the four are blind by arithmetic.**
`answer_relevance(question, answer)` is never handed the retrieved chunks.
`faithfulness` and `context_recall` join the chunks and take a set of their
tokens, so adding chunks can only grow that set — both are monotone
non-decreasing in retrieval noise. Only `context_precision` has a denominator
that grows.

**CONTROL: the metrics do move when retrieval fails completely** — retrieving only
distractors gives faithfulness 0.0000 and recall 0.0000. It is *degradation*, the
thing a CI gate exists to catch, that is invisible.

**FINDING: a retriever that fetches one chunk three times is scored as a perfect
one.** Returning the gold chunk duplicated gives faithfulness, precision and
recall all **1.0000** and the same verdict as the healthy retriever.

**FINDING: the one responsive metric measures string identity.**
`context_precision` tests `c in relevant_chunks`. Splitting the identical gold
sentence into two halves loses no text and scores **0.0000**, while faithfulness
and recall stay 1.0000 — change the chunker and precision reports a regression
that did not happen.

**CONTROL: bottom-decile alerting inspects zero cases at this size.**

| n cases | 3 | 5 | 9 | 10 | 20 |
|---|---:|---:|---:|---:|---:|
| `int(0.1 * n)` | 0 | 0 | 0 | 1 | 2 |

Below ten cases the lowest-10% slice is empty — here and on `main()`'s three
cases. There is nothing to rank regardless: faithfulness takes `[1.0]` as its
only value across the suite. And the doc's Step 2 signature
`answer_relevance(question, answer, encoder, llm)` raises
`TypeError: answer_relevance() takes 2 positional arguments but 4 were given`.
