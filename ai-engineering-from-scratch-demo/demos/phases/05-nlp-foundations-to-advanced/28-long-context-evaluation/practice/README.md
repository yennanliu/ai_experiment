<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 28-long-context-evaluation

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/28-long-context-evaluation/) · upstream spec
`phases/05-nlp-foundations-to-advanced/28-long-context-evaluation/docs/en.md`

```bash
uv run demo practice run 28-long-context-evaluation --ex 1
uv run demo explain 28-long-context-evaluation --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/28-long-context-evaluation
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Build a NIAH with 3 depths (0.25, 0.5, 0.75) × 3 lengths (1k, 4k, 16k). Run on any mode… | code | T0 | `ex01_the_requested_grid_cannot_vary.py` |
| 2 | Medium. Add a 3-needle variant. Measure retrieval of all 3 at each length. Compare to single-… | code | T0 | `ex02_the_multi_needle_score_cannot_fail.py` |
| 3 | Hard. Construct a variable-tracing task (X1 → X2 → X3, with 3 hops) embedded in 64k of filler… | code | T0 | `ex03_every_model_answers_the_first_assignment.py` |
<!-- generated:end -->

## Answers

The lesson ships a mock retrieval model with one knob — a hard word-offset
cutoff — and three exercises that ask you to sweep it. Every finding here comes
from asking what the sweep can actually show: the grid the exercise specifies
cannot vary, the multi-needle scorer cannot fail, and the three-hop task cannot
be answered by any capacity setting at all.

All three run at **T0** — `code/main.py` imports only `random` and `re`, and the
solutions add nothing but `importlib.util` and `re`.

### 1 — The requested grid cannot vary

The grid the exercise asks for, on `mock_retrieval_model` at the effective
capacity `run_niah_grid` hard-codes (20,000 words):

| depth \ length | 1,000 | 4,000 | 16,000 |
|---|---:|---:|---:|
| 0.25 | 1 | 1 | 1 |
| 0.50 | 1 | 1 | 1 |
| 0.75 | 1 | 1 | 1 |

**ANSWER: 9 of 9.** Pass rate 1.0 in every cell, zero variance across both axes.
There is no heatmap to plot.

**MECHANISM: the deepest cell sits 8,000 words inside the cutoff.** The mock
passes exactly when `int(length * depth) <= 20000`, and the deepest requested
cell puts the needle at word 12,000 of 16,000. Binary-searching the boundary:

| depth | shortest failing length |
|---|---:|
| 0.25 | **80,004** |
| 0.50 | **40,002** |
| 0.75 | **26,668** |

The exercise's longest context is 16,000 — 1.7× short of the first length that
could produce a zero at its most demanding corner.

**FINDING: "run on any model" has no model.** `openai`, `anthropic`,
`transformers`, `torch`, `tiktoken` and `matplotlib` are 6 of 6 probed modules
missing, so both the model and the plot resolve to what the lesson ships: a
regex mock and a printed text grid.

**FINDING: the depth axis only ever falls.** Sweeping depth 0 → 1.0 gives
`[1,1,1,1,1]` at 20k, `[1,1,1,0,0]` at 30k, `[1,1,0,0,0]` at 60k. A staircase,
never a U. The Pitfalls section asks for the "lost in the middle" effect; a hard
positional cutoff cannot express one at any length.

**CONTROL: a predictor that reads nothing reproduces the model exactly.**
`int(length * depth) <= 20000` agrees with the mock on **50 of 50** cells over
ten lengths and five depths. The mock also ignores its `question` argument —
asking "What is the capital of France?" of a pineapple haystack returns
`pineapple` — and `run_niah_grid` prints its rows and returns `None`, so the
pass rates the exercise wants plotted are not values it hands back.

### 2 — The multi-needle score cannot fail

Swept over eight lengths, with the single-needle probe at depth 0.5 scored at the
20,000-word capacity:

| length | shipped `run_multi_needle` | 1 needle @ 20k | 3 needles @ 20k | mean of 3 single probes |
|---|---:|---:|---:|---:|
| 1,000 | 1.0 | 1 | 1.000 | 1.000 |
| 4,000 | 1.0 | 1 | 1.000 | 1.000 |
| 16,000 | 1.0 | 1 | 1.000 | 1.000 |
| 26,000 | 1.0 | 1 | 0.667 | 0.667 |
| 40,000 | 1.0 | 1 | **0.333** | **0.667** |
| 64,000 | 1.0 | 0 | 0.333 | 0.333 |
| 100,000 | 1.0 | 0 | 0.333 | 0.333 |
| 200,000 | 1.0 | 0 | 0.000 | 0.000 |

**ANSWER: 8 of 8 perfect.** `run_multi_needle` returns 1.0 at every length from
1k to 200k. There is no length at which the 3-needle number can drop.

**MECHANISM: one argument.** `run_multi_needle` passes
`effective_capacity=length` — a budget equal to the whole context — while the
single-needle path is scored at a fixed 20,000. The deepest plant sits at word
160,010 of a 200,015-word haystack, always ≈ `0.8 × length`, so a cutoff set to
`length` can never bind. The difficulty is the fixture's argument, not the model.

**FINDING: the requested comparison comes out backwards.** The 3-needle score is
at or above the 1-needle score at all 8 lengths and strictly above it at 3 —
every length past 40,002. "Single-needle success does not predict multi-needle
success," says the lesson; here multi-needle success is unconditional.

**FINDING: at matched capacity the 3-needle number is redundant.** It equals the
mean of the three independent single-needle probes at **7 of 8** lengths.
Nothing is shared between needles, so multi-needle here is three separate
lookups averaged, not attention being juggled.

**MECHANISM: the eighth length is a five-word artefact.** At 40,000 the plants
land at words 8,000 / **20,005** / 32,010, while the middle needle alone lands at
**20,000**, exactly on the cutoff. The five extra words are the shallow needle's
own text — a 0.0125% shift in position moving the score from 0.667 to 0.333.

**CONTROL: the multi scorer never calls the model.** `score_multi_needle` runs
its own regex over the context, so it is an oracle over the fixture. Routing the
question through `mock_retrieval_model` instead caps at **1 of 3** at every
length, since the mock returns one string. And its pattern accepts only the
literal `the magic word is`: needles reworded the way the lesson's own NoLiMa
pitfall recommends score **0.333** there while `score_single_needle` still
returns 1.

### 3 — Every model answers the first assignment

Three "frontier models" — the lesson's mock at three effective capacities — on
`X1 = 42` → `X2 = X1 + 10` → `X3 = X2 * 2`, planted at depths 0.15 / 0.45 / 0.75:

| model | capacity | 1-hop over 1k…256k | 3-hop | retrieval-effective | reasoning-effective |
|---|---:|---|---|---:|---:|
| mock-small | 5,000 | `[1,1,1,0,0,0]` | `[0,0,0,0,0,0]` | **32,000** | **0** |
| mock-mid | 20,000 | `[1,1,1,1,1,0]` | `[0,0,0,0,0,0]` | **128,000** | **0** |
| mock-large | 80,000 | `[1,1,1,1,1,1]` | `[0,0,0,0,0,0]` | **256,000** | **0** |

**ANSWER: three-hop accuracy is 0 of 18 cells.** `openai`, `anthropic` and
`transformers` are all absent, so "3 frontier models" resolves to three
capacities of the shipped mock — and every one of them has an effective
reasoning length of 0 at the lesson's own 0.7 threshold. The number the exercise
asks you to report does not exist.

**MECHANISM: the question is never read.** The mock returns the first regex match
inside its capacity, and `X1 = 42` is the shallowest plant, so at the specified
64k of filler "What is X1?", "What is X2?" and "What is X3?" all answer `42`. The
third hop is answered with the first hop's right-hand side.

**MECHANISM: the regex returns a variable name.** Reordering so `X3 = X2 * 2` is
shallowest does not help — the pattern captures `X3 = X2` and the model returns
`X2`. There is no arithmetic anywhere in `mock_retrieval_model`, so no hop is
representable at any capacity.

**FINDING: one of the two spec-sheet numbers is degenerate.** Retrieval-effective
lengths separate the three models cleanly (32k / 128k / 256k); reasoning-effective
is identical for all of them. The lesson's prescription produces one number that
ranks models and one that cannot.

**FINDING: the specified length is one point, not a curve.** At 64k the three
models score 0, 1, 1 on the retrieval question; the boundaries at 32,000 and
128,000 appear only once the sweep is widened.

**CONTROL: a model that reads nothing beats all three.** Answering the constant
`42` to every question scores **18 of 18** on the one-hop task against the mocks'
**14 of 18** — they return "no answer" whenever the plant is out of capacity —
and ties them at **0 of 18** on the three-hop task. Both halves of the spec sheet
are matched or beaten by a system with no context at all.
