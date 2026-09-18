<!-- generated:start -->
# 11-llm-engineering / 10-evaluation

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/11-llm-engineering/10-evaluation/) · upstream spec
`phases/11-llm-engineering/10-evaluation/docs/en.md`

```bash
uv run demo practice run 10-evaluation --ex 1
uv run demo explain 10-evaluation --ex 1
uv run pytest demos/phases/11-llm-engineering/10-evaluation
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add BERTScore. Implement a simplified BERTScore using word embedding cosine similarity. Creat… | code | T0 | `ex01_random_vectors_score_an_unrelated_token_at_zero_point_37.py` |
| 2 | Build pairwise comparison. Modify the judge to compare two model outputs side-by-side instead… | code | T0 | `ex02_twenty_three_of_thirty_two_comparisons_are_ties.py` |
| 3 | Implement stratified analysis. Group test cases by category (factual, technical, safety, codi… | code | T0 | `ex03_the_bootstrap_draws_eight_distinct_resamples_out_of_a_thousand.py` |
| 4 | Add inter-rater reliability. Run the LLM judge 3 times on each test case (simulating differen… | code | T0 | `ex04_three_raters_give_identical_ratings_so_kappa_is_undefined.py` |
| 5 | Build a cost tracker. Track the token usage and cost of every judge call. Each input to the j… | code | T0 | `ex05_the_exercise_overstates_the_judge_prompt_five_fold.py` |
<!-- generated:end -->

## Answers

The lesson is pure stdlib, so all five exercises are **T0** and run in CI. Two
things run through all of them. `simulate_judge_score` is a pure function — a
length bucket, a reference-overlap adjustment and `md5(...) % 100` — so
anything the exercises ask to measure about *rater* variation measures the
hash. And `bootstrap_confidence_interval` asks for 1,000 resamples and draws
8, which is the number behind every interval in the lesson.

### 1 — random vectors score an unrelated token at 0.42

```text
mean cosine between two different words   -0.0021   (sd 0.1433 = 1/sqrt(50))
identical pair                             1.0000
greedy max over 20 reference tokens        0.4157   <- the floor
```

**ANSWER: on random vectors, BERTScore is exact match plus a floor.** Greedy
matching takes the *maximum*, and the maximum of twenty draws from a
mean-zero distribution is not zero.

**FINDING: 100 common words cover 66 of the suite's 200 tokens** — 33%.
"self-attention", "unsupervised", "cardiovascular" have no vector at all, and
skip, zero and random are three different metrics.

**FINDING: it restates a metric already in the file.** F1 correlates **0.787**
with the lesson's `word_overlap_score` across 8 cases × 3 models.

**CONTROL: give the vectors structure.**

| | random vectors | synonym-aware | `word_overlap_score` |
|---|---:|---:|---:|
| `"the answer is good"` vs `"the response is great"` | 0.534 | **0.983** | 0.333 |

That gap is the point of BERTScore, and random vectors cannot produce it.

### 2 — 23 of 32 comparisons are ties

| | count |
|---|---:|
| baseline-v1 wins | 4 |
| **ties** | **23** |
| baseline-v2 wins | 5 |

**ANSWER: the win rate is 0.444 with a Wilson interval of (0.189, 0.733)** —
0.545 wide, spanning 0.5, on **9** decisive comparisons. Nothing can be
concluded, and saying so is the answer.

**FINDING: the exercise's judge has no tie option** — "which output is better"
has two values and the data has three. An integer 1–5 score ties whenever both
outputs land in the same bucket.

**MECHANISM: baseline-v2's output is longer on 32 of 32**, and the base score
is a bucket on `len(output) > len(input) * 0.5`.

**FINDING: the judge is not invariant to whitespace.** Padding v1's output with
spaces — no semantic change — changes **11** verdicts and moves the record to
6-21-5, flipping the sign.

**FINDING: the interval depends on a denominator the exercise does not name.**

```text
Wilson(wins=4, n=32)  ->  (0.050, 0.281)     upper bound 0.28
Wilson(wins=4, n=9)   ->  (0.189, 0.733)     upper bound 0.73   (2.6x)
```

### 3 — the fixture does show the effect, and the bootstrap behind it has 8 samples

| category | v1 | v2 | |
|---|---:|---:|---|
| factual | 4.750 | 4.375 | **regressed** |
| summarization | 4.000 | 3.500 | **regressed** |
| technical | 3.750 | 4.000 | improved |
| safety | 3.125 | 3.625 | improved |
| coding | 3.500 | 3.500 | unchanged |
| **overall** | **3.844** | **3.875** | **improved** |

**ANSWER: the exercise's own claim is realised in its own fixture.**

**FINDING: the intervals come from 8 distinct resamples.**

```python
idx = (seed + j * 31) % n          # LCG modulus 2**31
```

When `n` divides a power of two, `x % n` depends only on `seed % n` — the LCG's
low bits are its whole period.

| n | distinct resamples of 1,000 |
|---:|---:|
| 5 | **993** |
| 8 | **8** |
| 16 | **16** |
| 32 | **32** |

Every group size in this lesson is a power of two. The 16 scores of the two
largest categories collapse to **4** distinct means.

**FINDING: the reported interval is 1.7× too wide** — `(3.500, 4.250, 4.750)`
against a real bootstrap's `(3.875, 4.250, 4.625)`. The analysis errs toward "no
significant change" everywhere it is used.

**FINDING: `bootstrap_confidence_interval([4])` returns `(0.0, 0.0, 0.0)`** —
which prints as a score of zero, not as a refusal.

### 4 — three raters give identical ratings

**ANSWER: observed agreement 1.000 on all 32 cells.** `simulate_judge_score` is
a pure function of `(input, output, criterion)`; there is no sampling in it.
Cohen's kappa is 1.0 on all four criteria, which is a statement about the
function.

**FINDING: the rubric is barely exercised.**

| criterion | levels it ever emits |
|---|---|
| relevance | 4, 5 |
| correctness | 2, 3, 4 |
| helpfulness | 3, 4, 5 |
| safety | 1, 4, 5 |

Two or three of five, each.

**FINDING: the rule can never fire.** "If agreement is below 0.7, rewrite the
rubric" requires disagreement.

**FINDING: three raters made by appending 1, 2 and 3 spaces** — semantically
identical documents — disagree on **17 of 32** cells, agreement **0.469**. The
md5 seed is over the raw string, so the exercise's rule would read whitespace as
an ambiguous rubric.

**CONTROL: paraphrasing agrees on 29 of 32 (0.906).** The judge is more stable
under a rewrite than under a space. And **16 of the 17** whitespace
disagreements are one point apart — which the *nominal* kappa the exercise names
charges exactly as much as a 1-versus-5.

### 5 — the exercise overstates the judge prompt five-fold

```text
32 calls   3,216 input tokens   3,200 output tokens   $0.04 / run   $1.74 / month
```

**FINDING: 500 input tokens per call is 5× the measured 100.**

| part of the prompt | share |
|---|---:|
| the rubric | **62%** |
| the model output | 28% |
| the case's input text | 10% |

The thing the judge is judging is the smallest part of what it reads.

**FINDING: 28% of the input is the same text four times.** The four criteria of
a case differ only in their rubric block, so the 304 tokens of case content are
sent four times — 912 of 3,216. Batching gives 2,304 input tokens, a **28%**
cut for the same 32 scores.

**FINDING: the projection is per-suite, and the suite is 8 cases.** Scaled to
1,000 cases: **$5.00 a run, $216.88 a month**. The per-suite framing hides that
behind a suite that costs four cents.
