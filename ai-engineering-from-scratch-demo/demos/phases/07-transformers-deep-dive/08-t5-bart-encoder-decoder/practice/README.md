<!-- generated:start -->
# 07-transformers-deep-dive / 08-t5-bart-encoder-decoder

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/07-transformers-deep-dive/08-t5-bart-encoder-decoder/) · upstream spec
`phases/07-transformers-deep-dive/08-t5-bart-encoder-decoder/docs/en.md`

```bash
uv run demo practice run 08-t5-bart-encoder-decoder --ex 1
uv run demo explain 08-t5-bart-encoder-decoder --ex 1
uv run pytest demos/phases/07-transformers-deep-dive/08-t5-bart-encoder-decoder
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`, apply span corruption to a 30-token sentence, verify that concatena… | code | T0 | `ex01_concatenating_gives_the_wrong_order.py` |
| 2 | Medium. Implement BART's `text_infill` noise: replace random spans with a single `<mask>` tok… | code | T0 | `ex02_the_masks_all_land_in_the_first_sixth.py` |
| 3 | Hard. Fine-tune `flan-t5-small` on a tiny English → pig-Latin corpus (200 pairs). Measure BLE… | code | T0 | `ex03_bleu_reads_sixty_eight_when_half_is_wrong.py` |
<!-- generated:end -->

## Answers

The lesson is 192 lines of noise functions with no model. Exercise 1 describes a
reconstruction procedure that does not reconstruct; Exercise 2 asks you to
implement a function the lesson already has; Exercise 3 asks for two fine-tunes
of models that cannot be fetched. All three are **T0**.

### 1 — concatenating reproduces the length, not the sentence

**ANSWER: `round_trip` is exact 300/300; the concatenation the exercise describes
is not.** It gets the length right — 30 tokens — and the order wrong, first
mismatch at index 14, because every masked span is appended at the end instead of
returned to its own position. It *cannot* work: the sentinel **is** the position,
so dropping it leaves the original's multiset rather than its sequence.
`<extra_id_k>` in the source and `<extra_id_k>` in the target are two halves of
one pointer.

**FINDING: at the documented rate, 30 tokens get exactly one span.**

| | value | why |
|---|---:|---|
| `n_mask` | **4** | `int(round(30·0.15))` = `round(4.5)`, and Python rounds halves to even |
| `n_spans` | **1** | `int(round(4 / 3.0))` |
| mean spans over 300 seeds | **1.00** | — |
| mean masked tokens | **2.33** | `int(gauss(3,1))` truncates, then `min(…, n − start)` clips |
| **realised mask rate** | **7.8%** | against the 15% requested |

So `mean_span=3.0` selects the *number* of spans and never their length, and the
corruption is about half the strength it advertises.

**CONTROL: the closing sentinel is load-bearing.** `round_trip`'s parser flushes
a span only when it meets the *next* sentinel, so dropping the trailing
`<extra_id_k>` silently loses the last one. And no two sentinels are ever
adjacent — which follows from there being one.

### 2 — the masks all land in the first sixth

One example at rate 0.3 on 30 tokens:

```text
<mask> brown fox <mask> <mask> <mask> in time saves nine language models learn
statistical patterns subword tokenization helps rare words today and again
forever more
```

Four masks hide 9 tokens in a 25-token source. Sampling the span-length
expression 7,811 times gives lengths **1–6**, so the corrupted text cannot say
how many tokens a mask swallowed — which is exactly the objective.

**FINDING: every mask lands at the front.**

| rate | mean relative `<mask>` position |
|---:|---:|
| 0.15 | **0.158** |
| 0.30 | 0.303 |
| 0.60 | 0.494 |

`text_infill` walks left to right opening a span with a **hardcoded 0.3** per
position while decrementing a budget of `int(n·rate)`. At the lesson's own
default the budget is gone inside the first sixth and **the last two thirds of
every sentence are never corrupted**. That the skew vanishes as the rate rises is
the tell: it is the budget running out, not the noise model. So `rate` controls
how much is masked and never where — in BART those are the same knob.

**FINDING: zero-length spans never occur.** BART's text infilling inserts a
`<mask>` covering nothing, which is what teaches the model that a mask can be
deleted. `max(1, …)` forbids it: 0 appears **0 times** in 7,811 sampled spans.

**CONTROL: the total is right even though the placement is not.** 13.3% realised
at rate 0.15 and 29.8% at 0.30 — unlike `corrupt_spans` next door, which asks for
15% and delivers 7.8%.

### 3 — BLEU reads 68 for a system wrong on more than half

Nothing is installed (`torch`, `transformers`, `datasets`, `sacrebleu`,
`sentencepiece` all absent) and there is no network. What *is* buildable is the
corpus and the metric — 250 pronounceable words, 200 train / 50 held out with
zero overlap, character-level corpus BLEU-4 implemented here — and running that
metric against systems whose errors are known exactly says more than either
fine-tune would.

| system | BLEU-4 | exact |
|---|---:|---:|
| the rule, correct | 100.00 | 50 / 50 |
| vowel branch wrong (`way` → `ay`) | 98.07 | 48 / 50 |
| **moves one letter, not the onset cluster** | **68.47** | **22 / 50** |
| copy the input | 23.37 | 0 / 50 |
| memorised all 200 training pairs | 23.37 | 0 / 50 |

**FINDING: the floor is 23, not 0.** Pig-Latin is a permutation of the input's
characters plus a suffix, so a system that gets *nothing* right still scores
23.37. A quarter of the scale is unreachable from below.

**FINDING: the top of the scale is blunter than the error rate it reports.** The
one-letter system is wrong on **28 of 50** items and reads 68.47. Exact match —
which a deterministic transduction actually admits — reads 48, 22, 0 cleanly on
the same outputs.

**FINDING: memorising all 200 pairs scores exactly the copy baseline.** Held-out
overlap is **0**, so a model that memorised the table perfectly is
indistinguishable from one that learned nothing. 200 pairs either teach the rule
or teach nothing measurable; the partial credit lives in the metric, not the task.

**CONTROL: "the same compute" is a 16× step ratio.** flan-t5-small's 77M against
Llama-3.2-1B's 1.24B is **16.1×**, so equal FLOPs buys the larger model about a
sixteenth of the steps — and the comparison also puts an instruction-tuned
checkpoint against a base one, confounding architecture with pretraining recipe.
