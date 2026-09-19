<!-- generated:start -->
# 12-multimodal-ai / 17-video-language-temporal-grounding

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/17-video-language-temporal-grounding/) · upstream spec
`phases/12-multimodal-ai/17-video-language-temporal-grounding/docs/en.md`

```bash
uv run demo practice run 17-video-language-temporal-grounding --ex 1
uv run demo explain 17-video-language-temporal-grounding --ex 1
uv run pytest demos/phases/12-multimodal-ai/17-video-language-temporal-grounding
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | For a 3-minute cooking demo, pick uniform vs dynamic FPS. Justify with a token count. | code | T0 | `ex01_the_budget_parameter_cannot_reach_the_budget.py` |
| 2 | TMRoPE adds what specifically that a simple temporal embedding table cannot do? | explain | T0 | prose, below |
| 3 | Write a JSON schema for temporal grounding that a VLM can learn to emit. Include error cases. | code | T0 | `ex03_sixty_guesses_score_one_point_zero_and_the_honest_three_score_a_third.py` |
| 4 | Read Video-LLaVA's Section 3 on "Alignment Before Projection." Why is this better than traini… | explain | T0 | prose, below |
| 5 | Given the VideoMME leaderboard, what is the gap between the top open model and the top propri… | code | T0 | `ex05_the_lesson_names_four_benchmarks_and_reports_zero_scores.py` |
<!-- generated:end -->

## Answers

Three code solutions and two prose ones. Exercise 3's last finding is the one to
carry away: on the lesson's own demo, sixty guesses with no model involved score
**1.00** and the lesson's own three honest predictions score **0.33**.

### 1 — the budget parameter cannot reach the budget

| strategy | frames | tokens | share of a 32k context |
|---|---:|---:|---:|
| uniform, 32 frames | 32 | **2,592** | 7.9% |
| `dynamic_sample(..., total_budget=32)` | **180** | **14,580** | **44.5%** |

**ANSWER: uniform, by 5.6×.** At 180 seconds the two strategies are not within an
order of magnitude of each other, and the reason has nothing to do with motion.

**FINDING: `total_budget` does not bound anything.** The per-second allocation is
`min(fps_cap, max(1, round(raw)))`, so every second gets at least one frame.

| budget asked | 8 | 32 | 1000 |
|---|---:|---:|---:|
| frames returned | **180** | **180** | 720 |

The reachable range at 180 s is **1 to 4 FPS**, and two of the three budgets give
the identical answer.

**FINDING: the lesson's own demo overshoots its stated budget by 25%.** It calls
`total_budget=12`, prints "dynamic (12 frames total)", and the call returns
**15** — six of its ten seconds round their raw allocation to 0 and are lifted to
1 by the `max`.

**FINDING: so the justification is the floor, not the motion.** A cooking demo is
mostly low motion, which is exactly the case the sampler cannot exploit: the
seconds that should get *fewer* frames are the ones the floor protects. Its
cheapest possible output at this duration is 14,580 tokens.

### 2 — what TMRoPE adds over an embedding table

Drawing on **Qwen2.5-VL and TMRoPE**, which lists three differences; the third is
the one that cannot be recovered any other way.

**A learned temporal embedding table is a lookup on a frame index.** Row *k* is
the embedding for "the k-th frame I was given". Three things follow, and TMRoPE
fixes each:

1. **The table has no units.** Row 15 means "frame 15", and what that *is* in
   seconds depends on the sampling rate the caller happened to use — which
   exercise 1 shows the lesson's own sampler cannot even report reliably. TMRoPE
   rotates by the **timestamp**, so 4.2 s is 4.2 s at any rate, and "at what
   second does the cat jump" becomes an answerable question rather than a
   rescaling problem.
2. **The table is fixed-length.** It has N rows and video has arbitrary duration,
   so a longer clip either wraps, interpolates, or is truncated. A rotation is a
   function, not a table: it extends to any *t* without retraining. This is the
   same argument Lesson 12.01's exercise 5 makes for 2D-RoPE against a learned
   position table, one axis over.
3. **The table cannot express uneven spacing, and this is the one that
   matters.** Dynamic-FPS sampling produces frames at 2 FPS here and 4 FPS there
   — the gaps between consecutive frames are *not* constant. An index table has
   no way to say that frame 15 and frame 16 are 0.5 s apart while frame 16 and 17
   are 0.25 s apart; every row is one step from the next by construction. TMRoPE
   carries the actual time, so the irregularity is in the encoding rather than
   lost.

**And that third point is what couples the two halves of this lesson.** Dynamic
FPS is only useful if the model can tell that the frames are unevenly spaced;
otherwise variable sampling is just noise injected into a uniform positional
scheme. TMRoPE is the precondition for the sampler, not an independent
improvement — which is why the lesson's "2026 best practice" list names them on
consecutive lines.

**What it does not add.** Per-token rotation is not extra *capacity*; it is the
same positional information carried without a parameter. The table costs
N × hidden parameters and TMRoPE costs none, so the change is a strict reduction
in parameters and a strict increase in what can be represented — which is rare
enough to be worth naming as the reason it was adopted everywhere within a year.

### 3 — sixty guesses score 1.00 and the honest three score a third

**ANSWER: four fields, an envelope, three constraints** — each constraint
justified by the case the lesson's own evaluator mis-scores:

```json
{"events": [{"event": "jump", "start": 4.1, "end": 4.3, "confidence": 0.82}],
 "duration": 10.0}
```

`end > start` strictly · `0 <= start < end <= duration` · same-label spans
disjoint.

**ERROR CASE 1 — a zero-length event scores 0.0 when it is exactly right.** `iou`
returns 0.0 whenever the union is zero, so `blink` at [3.0, 3.0] against a ground
truth of [3.0, 3.0] is recorded as a **miss**. Instantaneous events need a stated
minimum span.

**ERROR CASE 2 — a reversed interval scores 0.0 and is not rejected.** [5.0, 4.0]
gives a negative intersection, clamped to 0, and passes through as an ordinary
miss. A schema can enforce ordering; the evaluator cannot and does not.

**ERROR CASE 3 — one prediction is credited for two ground-truth events.**
Matching is by label with no assignment step, so a single `jump` at [4.1, 4.6]
scores **0.667** against *both* [4.0, 4.5] and [4.2, 4.7] and takes recall to
**1.00**.

**FINDING: the whole metric is recall, so guessing wins.**

| predictions | count | recall |
|---|---:|---:|
| the lesson's own three | 3 | **0.33** |
| every half-second window, every label | **60** | **1.00** |

No model is involved in the second row. And a prediction matching no ground truth
costs nothing at all. The `confidence` field is in the schema because the metric
needs a precision term — a recall-only score cannot distinguish a grounding model
from a sweep.

### 4 — why "alignment before projection" beats two encoders

Drawing on **VideoChat and Video-LLaVA**, which is where the lesson records
Video-LLaVA training "a single visual encoder on both images and video frames".

**The name describes the ordering, and the ordering is the whole argument.** Two
ways to build a system that takes both images and video:

- **Projection before alignment** (the thing it beats): one encoder for images,
  another for video clips, each with its own projector into the LLM. Two
  representations, aligned to the LLM separately.
- **Alignment before projection**: one encoder trained on images *and* frames so
  that both land in a shared visual space **first**, then a single projector maps
  that one space into the LLM.

**Three reasons the second wins, in order of how much they matter:**

1. **The LLM sees one distribution instead of two.** With separate towers, a
   video frame and a still photograph of the same scene arrive at the LLM as
   different vectors, and the LLM has to learn that they mean the same thing —
   from video-text data, which is scarcer and noisier than image-text data by
   orders of magnitude. Aligning first means the LLM's image competence transfers
   to video for free, which is the actual result being claimed.
2. **The image data does the video training.** Image-text pairs outnumber
   video-text pairs enormously, so a shared encoder is trained mostly on images
   and the video frames inherit that. Two towers cannot share it: the video tower
   sees only video.
3. **It halves the thing Lesson 12.07 rates at 20% of variance.** The encoder is
   the second-largest axis in that lesson's decomposition, and running two of them
   means tuning it twice with half the data each. A single encoder is not just
   cheaper; it is the axis where data efficiency compounds.

**The caveat worth stating.** This is an argument about *representation*, not
about time. Video-LLaVA aligns frames and images into one space and then has no
mechanism for when each frame occurred — the lesson's own table records it as an
"MLP / 8 frames" system, and its next section explains why that is not enough.
Alignment before projection solves the modality-gap problem and leaves the
temporal one untouched, which is exactly the division of labour between this
exercise and exercise 2.

### 5 — the lesson names four benchmarks and reports zero scores

**ANSWER: the gap cannot be computed here.** VideoMME, TempCompass, EgoSchema and
Video-MMMU appear by name; **0** scores appear anywhere in the lesson's text, and
`arch_compare` has **5** rows and no benchmark column.

**FINDING: the one number the table carries spans 4×.** Frame counts run **8 to
32**, which at Lesson 12.08's pooled 81 tokens a frame is **648 to 2,592** visual
tokens. Every row also differs in compressor, so no two of the five isolate
anything.

**ANSWER: by Lesson 12.07's weights the split is 4 to 1 toward tokens.**

| axis | variance |
|---|---:|
| visual-token count | **60%** |
| LLM size | **15%** |

Frame count, sampling rate and per-frame pooling all sit inside the 60%, so a
stated extrapolation puts four times more of any video gap on the
temporal-and-token side than on base-LLM scale.

**FINDING: and the decomposition cannot express the question.** Six additive
terms, **zero** products of any two — so "how much of the gap is temporal
encoding **vs** LLM scale", which is a question about how the two interact, has
no term to land in. The same weights sum to **115%**, which Lesson 12.07's
exercise 1 measures on the same table.
