<!-- generated:start -->
# 10-llms-from-scratch / 10-evaluation

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/10-evaluation/) · upstream spec
`phases/10-llms-from-scratch/10-evaluation/docs/en.md`

```bash
uv run demo practice run 10-evaluation --ex 1
uv run demo explain 10-evaluation --ex 1
uv run pytest demos/phases/10-llms-from-scratch/10-evaluation
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a "consistency" scorer that runs the same input through the model 5 times and measures ho… | code | T1 | `ex01_perfectly_consistent_and_not_reproducible.py` |
| 2 | Extend the ELO tracker to support multiple judge functions (exact match, F1, LLM-as-judge) an… | code | T1 | `ex02_the_leaderboard_flips_at_0_55.py` |
| 3 | Build an eval suite for a specific task: email classification into 5 categories. Create 100 t… | code | T1 | `ex03_the_edge_cases_break_the_scorers_not_the_models.py` |
| 4 | Implement contamination detection: given a set of eval questions and a training corpus, check… | code | T1 | `ex04_the_paraphrase_detector_scores_a_shuffle_as_identical.py` |
| 5 | Build a "model diff" tool. Given eval results from two model versions, highlight which specif… | code | T1 | `ex05_three_scorers_three_verdicts_on_one_diff.py` |
<!-- generated:end -->

## Answers

`code/main.py` is an eval harness: `EvalCase`/`EvalSuite`, three scorers, an ELO
tracker, perplexity, and three demo models to run it on. Everything works. What
the five exercises surface is that the harness's own answers depend on choices
it does not surface — which scorer, which threshold, which weight, which
process.

All five are **T1** on the `math` group (`uv sync --extra math`).

### 1 — perfectly consistent, and not reproducible

**ANSWER: consistency is 1.00 for all three models on every prompt.** Five calls
each return identical strings. `demo_model_good` and `demo_model_bad` are
dictionary lookups; `demo_model_random` seeds numpy with `hash(prompt)`, so it
is a deterministic function of the prompt too. The scorer finds nothing, and it
is right.

**FINDING: the same models answer differently in different processes.**

```python
np.random.seed(hash(prompt) % 2**31)   # hash() is salted per interpreter
```

Six fresh interpreters with `PYTHONHASHSEED` unset return several different
answers to "What is 2 + 2?" — `Paris`, `error`, `no`, `maybe`, `yes`, `unknown`
are all reachable — and several different perplexities for one fixed string.

**FINDING: every perplexity the lesson prints changes between runs**, by of the
order of ten percent, because `token_log_probs_simulated` seeds on the same
salted hash. In an evaluation lesson.

**MECHANISM: the scorer looks where the inconsistency is not.** Five calls in
one interpreter share one hash seed and cannot disagree. The exercise says the
scorer reveals fragile prompts; the fragility here is at process boundaries.

### 2 — the leaderboard flips at 0.55

The weight is a distribution over all three judges, validated as one.

| model | exact | F1 | judge |
|---|---:|---:|---:|
| `demo_model_good` | 1.000 | 1.000 | 1.000 |
| `demo_model_bad` | 0.000 | 0.291 | 0.630 |
| `terse` *(added)* | **0.143** | **0.143** | **0.240** |

**FINDING: on the lesson's own two models no weighting changes anything.** Good
dominates bad on all three judges, so the comparison the exercise asks for has
no degrees of freedom on the data it ships.

**ANSWER: with a third model the leaderboard flips at weight(exact) = 0.55** —
`good, bad, terse` below it, `good, terse, bad` at and above, with the judge
weighted 0.

**FINDING: give the judge a third of the weight and the flip becomes
unreachable.**

| weight on the judge | exact-match weight that flips the board |
|---:|---|
| 0.0 | **0.55** |
| 0.1 | 0.60 |
| 1/3 | none — only 0.67 of the weight is left for exact match |

`llm_judge_simulated` agrees with F1 about these two models (bad 0.630 against
terse 0.240), so weighting it in pushes the crossover past the available range.
Which judge carries the weight decides whether the comparison has an answer.

**MECHANISM: the judges disagree about what a wrong answer is worth.**
`exact_match` gives a verbose-but-correct answer 0; `token_f1` gives it partial
credit and the judge rewards its length. Weighting them is choosing which
failure to call worse.

**FINDING: ELO contributes only path dependence.** The judges are deterministic,
so the ELO ranking *is* the ranking by mean score — and the same three matches
in two orders give the same leaderboard with different ratings, 1530.5 against
1532.0.

### 3 — the edge cases break the scorers, not the models

100 cases: 70 ordinary, 30 edge (empty, ambiguous, foreign-language).

| model | exact, ordinary | exact, edge |
|---|---:|---:|
| rule-based | 0.400 | 0.333 |
| keyword | **0.800** | 0.333 |
| simulated LLM | 0.400 | 0.333 |

**ANSWER: the models separate by 0.400 on the ordinary cases and by 0.000 on
the edge cases** — the edge cases are where the scorer decides the outcome.

**FINDING: the scorers disagree by the maximum on an empty prediction.**

```text
exact_match("", "")  ->  1.0
token_f1("", "")     ->  0.0
```

`token_f1` returns 0.0 whenever either side has no tokens, before comparing
anything. A model that correctly labels an empty email as empty is perfect to
one shipped scorer and zero to the other — and the exercise asks for empty
emails by name.

**FINDING: a single-label suite cannot express a multi-label case.** `EvalCase`
holds one `expected`, so a model picking the other defensible category scores
the same 0.0 as one answering `"purple"`. The second category lives in
`metadata`, where no scorer reads it.

**FINDING: `token_f1` treats category labels as bags of words** —
`"billing question"` vs `"billing complaint"` scores **0.500**, so two of the
five categories are half-right by spelling.

### 4 — the paraphrase detector scores a shuffle as identical

| pair | `token_f1` |
|---|---:|
| `"the cat sat on the mat"` vs itself | 1.000 |
| `"mat the on sat cat the"` vs it | **1.000** |
| `"the the the the the the"` vs it | **0.333** |
| `"a dog ran through a field"` vs it | 0.000 |

**ANSWER: 40% exact and 70% at threshold 0.6**, on a corpus with 4 of 10
questions planted verbatim and 3 reworded — exactly the plant.

**FINDING: word order and repetition are both discarded**, because `token_f1`
takes `set(...)` of both sides. A contamination audit built on it cannot tell a
quotation from an anagram, and a corpus of filler words registers as partially
contaminated with everything sharing its stop words.

**FINDING: the threshold does the work.** 0.3 → 90%, 0.5 → 80%, 0.6 → 70%,
0.9 → 40%, on one fixed corpus. The percentage the exercise asks for is a
function of a constant it never names.

### 5 — three scorers, three verdicts on one diff

`diff` returns one record per case — id, prompt, before, after, verdict — keyed
by position so a repeated prompt stays two cases, and the counts below are
derived from those records. The same change, `demo_model_bad` → a terse model:

| scorer | improved | regressed | unchanged |
|---|---:|---:|---:|
| exact match | 1 | **0** | 6 |
| token F1 | 1 | **4** | 2 |
| LLM judge | 0 | **5** | 2 |

**ANSWER: a pure win, a net loss, and a pure loss** — from identical inputs.

**FINDING: on the lesson's own two models no scorer finds a regression**, so the
shipped pair has no contested case and cannot show what the tool is for.

**MECHANISM: `exact_match` is blind to a wrong answer becoming a different wrong
answer.** Cases 0, 2, 3 and 6 went from `"Paris is the capital city of France"` to
`"no"` — both wrong, one sharing words with the expected answer. `exact_match`
calls that unchanged; `token_f1` calls it a regression; the judge penalises the
length drop on top.

**FINDING: the exercise's framing assumes what it asks you to build.**
"Essential for understanding whether a change helped or hurt" presumes a diff
has a direction. It has one per scorer, and choosing the scorer is the decision
the tool was supposed to inform.
