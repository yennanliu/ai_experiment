<!-- generated:start -->
# 12-multimodal-ai / 22-document-diagram-understanding

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/22-document-diagram-understanding/) · upstream spec
`phases/12-multimodal-ai/22-document-diagram-understanding/docs/en.md`

```bash
uv run demo practice run 22-document-diagram-understanding --ex 1
uv run demo explain 22-document-diagram-understanding --ex 1
uv run pytest demos/phases/12-multimodal-ai/22-document-diagram-understanding
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Your project is 10M invoices per day. Which stack minimizes cost-per-page without losing accu… | code | T0 | `ex01_the_table_has_one_numeric_column_and_the_question_needs_two.py` |
| 2 | Why does LayoutLMv3 outperform pure-CLIP-VLMs on form QA but underperform at scene-text? What… | code | T0 | `ex02_the_bbox_stream_gives_up_scale_invariance.py` |
| 3 | Nougat generates LaTeX. Propose a test case where VLM-native output beats Nougat on LaTeX fid… | explain | T0 | prose, below |
| 4 | Read PaliGemma 2 paper (Google, 2024). What was the key training-data addition that lifted do… | explain | T0 | prose, below |
| 5 | Design a regulatory-safe hybrid: OCR pipeline as primary, VLM as secondary cross-check. How d… | code | T0 | `ex05_a_naive_cross_check_escalates_five_million_fields_a_day.py` |
<!-- generated:end -->

## Answers

Three code solutions and two prose ones. Exercises 1 and 5 are the same problem
at two scales: the lesson's own table can price a stack but not choose one,
because the clause that decides — "without losing accuracy" — has no column.

### 1 — the table has one numeric column and the question needs two

| stack | tokens/page | daily cost at $0.25/M |
|---|---:|---:|
| **OCR pipeline + LayoutLMv3** | **512** | **$1,280** |
| Donut (OCR-free) | 4,096 | $10,240 |
| Nougat | 4,096 | $10,240 |
| VLM AnyRes 4-tile | **2,916** | $7,290 |
| VLM native 2048 | 8,192 | $20,480 |
| VLM native 2576 | 12,000 | **$30,000** |

**ANSWER: the OCR pipeline**, **23.4×** cheaper than the frontier row.

**FINDING: the table is not monotone in era.** The cheapest era-3 row is
**28.8%** cheaper than either era-2 row, so the era ordering carries no cost
information.

**FINDING: "without losing accuracy" cannot be evaluated here** — six rows, one
numeric column, zero accuracy figures. The clause that decides the answer is the
one the evidence does not address.

**FINDING: the 512 does not cover the lesson's own three-stream input.**
`layoutlm_input` returns 8 text ids and **256** patch ids for an eight-word page —
**97.0%** of positions are patches, and the 16 × 16 grid alone is **50%** of the
row that calls it a "small image".

### 2 — the bbox stream gives up scale invariance

**ANSWER: a form's answer is a spatial relation and a scene's is not.** "What is
the total?" is "the token to the right of the token reading Total", and the bbox
stream states that directly. A shopfront photograph has no layout grammar, so the
same stream is a per-token constant occupying positions a patch could have used.

**ANSWER: what it gives up is scale, rotation and crop invariance.**

| | 300 DPI | 600 DPI |
|---|---|---|
| first box | `(100, 50, 300, 80)` | `(200, 100, 600, 160)` |
| boxes changed | — | **8 of 8** |
| patch grid | unchanged | unchanged |

A CLIP tower is invariant to the resolution it resized from; a bbox stream is a
statement in pixels.

**FINDING: the lesson's own text stream is not reproducible between runs.**
`hash(t.text) % 10000` — Python salts string hashing per process, so the ids are
stable within one run and different on the next. The demo prints them as though
they were a tokenizer's output.

**FINDING: the three streams are 97% patches with no stated alignment.** There is
no alignment stream in the returned dict, so the box-to-patch correspondence is
exactly what the model must infer — and exactly what the bbox stream exists to
provide.

### 3 — where Nougat wins and where a VLM does

Drawing on **Math equations and LaTeX output**, which states the split the
exercise is asking to make concrete: "VLMs trained with LaTeX targets produce
usable LaTeX; without explicit LaTeX training, readable but imprecise".

**Nougat wins where the LaTeX is a *grammar* and the page is in distribution.**

> **Test case: a displayed multi-line `align` environment with numbered
> equations, cross-referenced by `\eqref` in the body text.**

Nougat was trained on arXiv source paired with rendered pages, so it has seen the
same constructs tens of millions of times and its output is *parseable by
design* — `\begin{align}`, `&` alignment points, `\\` row breaks, matched braces.
A VLM without LaTeX targets produces something that renders as a picture of the
equation and fails `pdflatex`. The measurable criterion is not similarity: it is
**does the output compile**, and then **does the compiled output match the
source glyph for glyph**.

**A VLM wins where the page is *not* in distribution and the answer needs
context.**

> **Test case: a hand-annotated printout — a printed equation with a
> hand-written correction over one term and an arrow to a marginal note.**

Nougat has one output mode: LaTeX for a clean typeset page. It has no
representation for "this term is struck through in pen" and, worse, no way to
signal that it could not read something — it will emit plausible LaTeX for the
printed version and silently drop the annotation. A VLM can say what it sees,
including that part of it is handwritten and part is printed, because its output
space is language rather than a markup grammar.

**The pattern underneath, which is the general rule.** A model with a *narrow*
output space is more accurate inside its distribution and fails invisibly outside
it. Exercise 5 of this lesson is the same observation at the pipeline level: the
value of the OCR arm is that its failures are deterministic, and the value of the
VLM arm is that it can be asked a question the schema did not anticipate. The
lesson's own recipe — "chain Nougat on the PDF, then a VLM on tricky pages" —
is a routing policy over exactly this, and the hard part is detecting "tricky",
which neither model reports.

### 4 — what PaliGemma 2 added

Drawing on **Era 3 — VLM-native (2024+)**, which records that PaliGemma 2 "trains
specifically for documents + handwriting" and puts it at ~88.4 DocVQA against
Nougat's ~77.3 and a pipelined LayoutLMv3's ~83.

**The addition was document-and-handwriting data in the pretraining mixture, not
a new architecture.** PaliGemma 1 was a general-purpose SigLIP-plus-Gemma
transfer model — strong at captioning, VQA and detection, and trained on a
mixture where documents were a small slice of a broad web corpus. PaliGemma 2's
document lift comes from adding a large, explicitly document-shaped stage:
OCR-style transcription targets, table and chart structure, and — the part the
lesson singles out — **handwriting**.

**Three reasons the data rather than the architecture is the right answer:**

1. **The architecture barely moved.** The family kept the SigLIP tower and the
   Gemma decoder; what changed was scale (a choice of Gemma 2 sizes and
   resolutions) and what it was trained on. Lesson 12.07 rates the whole
   connector-and-architecture axis at **5%** of benchmark variance against
   **10%** for data mix and **60%** for visual-token count.
2. **The benchmark it moves is the one data moves.** DocVQA is a transcription-
   and-layout task; it rewards having seen documents, and it does not reward a
   better attention pattern. Lesson 12.15's exercise 1 makes the same
   decomposition for Janus-Pro: only **17%** of that model's MMMU gain was the
   parameter increase and **83%** was data.
3. **Handwriting is specifically a data problem.** There is no architectural
   feature that makes cursive legible. There is a corpus that contains it, and
   a model that has not seen one cannot read it — which is exactly why the
   lesson's own handwriting section still lists it as the hardest sub-task and
   names the two models trained on it.

**The caveat worth keeping.** "The key addition" is a single-cause framing, and
PaliGemma 2 changed resolution, base-model size and data at once — the same
four-axes-one-outcome shape that Lesson 12.15's exercise 5 finds in Janus-Pro's
table. The honest statement is that the document-specific data is the addition
most likely to be responsible and the one the model card emphasises, and that
nothing published isolates it.

### 5 — a naive cross-check escalates five million fields a day

**ANSWER: agree-and-accept, disagree-and-escalate, and never resolve by
confidence.** An OCR character posterior and a VLM token log-probability are
different quantities on different scales, so a policy that picks the more
confident one is a coin flip wearing a number. Disagreement is a **signal to
stop**, not an input to a decision.

**FINDING: the cross-check cuts undetected error 200×.**

| policy | silently wrong |
|---|---:|
| primary only | **2.000%** |
| agree-and-accept | **0.010%** |

The residue is exactly the case where both err *and* land on the same wrong
value.

**FINDING: and it escalates 6.9% of fields — 4,823,000 a day.** Ten million
invoices at the **7** leaf fields of the lesson's own Donut schema is 70M fields,
and a reviewer at 10 seconds a field is **13,397 hours** of work per day.

**ANSWER: so the design is a field-weighted gate.** Escalate only `total` and
`invoice_number` — **2** of 7 — for **1,378,000** a day, **71%** fewer; then add
an amount threshold so that invoices above a limit are cross-checked on every
field and the rest only on those two. The question is not "how do we resolve
disagreement" but "which disagreements can we afford to see", and the answer is
an auditable, written-down policy — which is what makes the design
regulatory-safe rather than merely accurate.
