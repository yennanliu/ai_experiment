<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 20-structured-outputs-constrained-decoding

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/20-structured-outputs-constrained-decoding/) · upstream spec
`phases/05-nlp-foundations-to-advanced/20-structured-outputs-constrained-decoding/docs/en.md`

```bash
uv run demo practice run 20-structured-outputs-constrained-decoding --ex 1
uv run demo explain 20-structured-outputs-constrained-decoding --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/20-structured-outputs-constrained-decoding
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Prompt a small open-weights model (e.g., Llama-3.2-3B) without constrained decoding for… | code | T0 | `ex01_valid_json_is_not_compliance.py` |
| 2 | Medium. Same corpus with Outlines JSON mode. Compare compliance rate, latency, and semantic a… | code | T0 | `ex02_one_arm_cannot_vary_and_latency_goes_backwards.py` |
| 3 | Hard. Implement a regex-constrained decoder from scratch for phone numbers (`\d{3}-\d{3}-\d{4… | code | T0 | `ex03_a_validity_test_cannot_see_a_tight_constraint.py` |
<!-- generated:end -->

## Answers

Three exercises about a guarantee. Exercise 1 asks for a rate that no installed
runtime can produce, using a metric that over-counts and a sample size that
cannot resolve the question. Exercise 2 asks for a three-way comparison in which
one arm cannot vary, one moves the opposite way from the lesson's claim, and one
cannot move at all. Exercise 3 asks for an acceptance test that passes on a
decoder able to emit 0.098% of the legal outputs.

All three run at **T0** — `code/main.py` imports only `math`, `random` and `re`.

### 1 — Valid JSON is not compliance

`transformers`, `torch`, `llama_cpp`, `vllm` and `outlines` are all absent, so no
rate is measured. What is measured is the metric, over 14 outputs covering the
failure modes small models produce:

| | count |
|---|---:|
| parse under `json.loads` | 10 |
| parse under an RFC 8259 reader | **9** |
| satisfy `Review(sentiment, confidence, evidence_span)` | **2** |

**ANSWER: 8 of the 14 are valid JSON and not a `Review`.** A compliance rate
reported as a parse rate is an upper bound quoted as a measurement.

**MECHANISM: a bare scalar is a complete JSON document.** `"positive"` parses. So
does `{}`. JSON's grammar and the schema's are different grammars, and only one
of them is being checked.

**FINDING: the parse rate is not parser-independent.** `json.loads` accepts `NaN`
and `Infinity`, which RFC 8259 omits — 10 against 9 on the same corpus.

**FINDING: 100 reviews buys an interval, not a fraction.**

| observed | 95% Wilson interval |
|---|---|
| 100 / 100 | 96.3% – 100% |
| 80 / 100 | 71.1% – 86.7% |

By the rule of three a clean run of 100 bounds the failure rate only at **3.0%** —
three bad rows per hundred, from a run that looked perfect. Certifying 99% needs
**300** clean samples; 99.9% needs **3,000**. The exercise fixes n before naming
a target, which is the one order in which n cannot be chosen.

**MECHANISM: the lesson's own baseline repeats the mistake at n=20.**
`generate_unconstrained` has an exactly computable compliance rate of
**0.0031863** — 4 of 1000 measured — and `main()` prints its result at n=20,
where an all-invalid run is the modal outcome at 93.8%.

**CONTROL: the schema orders its fields the way the lesson warns against.** The
pitfall section says "put `answer` before `reasoning`, and the model commits to an
answer before it thinks" — then the exercise asks for `Review(sentiment,
confidence, evidence_span)`, answer first. Constrained decoding makes that
enforced.

### 2 — One arm cannot vary, and latency goes backwards

`outlines`, `vllm`, `xgrammar`, `lmformatenforcer` and `instructor` are all
absent, so the comparison runs against the lesson's own FSM decoder — the
construction Outlines compiles a schema into.

| axis | constrained | unconstrained |
|---|---:|---:|
| valid outputs / 1000 | **1000** | 4 (exact rate 0.0031863) |
| forward passes per sample | 12 | 12 |
| masked logit positions per sample | **132** | 0 |
| wall clock, 1000 samples | **1.2x** | 1.0x |

**ANSWER: compliance is not a measurement.** `generate_constrained` samples only
from characters `valid_next` returned, so an invalid output is unreachable rather
than unlikely. The other arm's 4/1000 is a property of the alphabet — 12 draws
over 11 characters — not evidence about a model.

**FINDING: latency goes the opposite way from the lesson's claim.** Masking
happens *after* the forward pass, not instead of it, and adds a full vocabulary
sweep per step.

**MECHANISM: the saving the lesson names is available and not taken.** States 3
and 7 have exactly one legal character, so **16.7%** of the forward passes are for
tokens already determined. `generate_constrained` runs them anyway.

**MECHANISM: and the mask cost is what scales.** 132 positions over an
11-character alphabet becomes **1,200,000** over the 100k vocabulary the lesson
says real implementations use. The forward passes stay at 12 either way.

**CONTROL: semantic accuracy cannot move.** At the 10 unforced positions all ten
digits stay legal, so the sampled distribution is the model's own renormalised
over a set it never excluded — chi-square **15.63** against 16.92 at 9 degrees of
freedom. The constraint buys format and nothing else, which is the correct result
and also leaves the third axis with no signal.

### 3 — A validity test cannot see a tight constraint

The decoder already ships and scores 1000/1000. Three mutations of `PhoneFSM`,
each run through the shipped `generate_constrained`:

| FSM | valid / 1000 | reachable numbers |
|---|---:|---:|
| shipped | 1000 | 10,000,000,000 |
| accept state one early | **0** | — |
| extra `-` at the separators | **6** | — |
| **digits 0–4 only** | **1000** | **9,765,625** (0.098%) |

**ANSWER: the acceptance criterion passes on a decoder that is badly wrong.**
Validity is invariant to how much of the legal space the constraint removes, so
the test is blind in exactly the direction constrained decoding fails in
production — an FSM that excludes the right answer.

**MECHANISM: the test is one-sided, and it catches the side that does not
matter.** A mask that is too *loose* is caught immediately. A mask that is too
*tight* is invisible.

**FINDING: 1000 samples is not a coverage test in either direction.** They touch
1e-07 of the space and return 1000 distinct outputs — neither near-exhaustive nor
repetitive enough to notice the gap. **Coverage has to be computed from the FSM**
— the product of the branching factors along the accepting path — not sampled
from it, and that comparison is available before a single sample is drawn.

**FINDING: the sampler has a fallback that ignores the mask.** `sample()` ends
with `return len(probs) - 1`, taken whenever the accumulated probability never
reaches the draw. Index 10 of this alphabet is `-`, which the mask zeroes at every
digit state. The guarantee rests on floating-point summation reaching 1.0, not on
the mask.

**CONTROL: the shipped FSM is exactly the pattern.** Three digits, a separator,
three digits, a separator, four digits, accepting at 12 — the full 10^10 numbers
`\d{3}-\d{3}-\d{4}` admits. The decoder is correct; the verification the exercise
asks for is what does not establish it.
