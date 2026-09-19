<!-- generated:start -->
# 12-multimodal-ai / 04-flamingo-gated-cross-attention

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/04-flamingo-gated-cross-attention/) · upstream spec
`phases/12-multimodal-ai/04-flamingo-gated-cross-attention/docs/en.md`

```bash
uv run demo practice run 04-flamingo-gated-cross-attention --ex 1
uv run demo explain 04-flamingo-gated-cross-attention --ex 1
uv run pytest demos/phases/12-multimodal-ai/04-flamingo-gated-cross-attention
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Compute Flamingo-9B's visual parameter count: 9B LLM + 1.4B gated cross-attention layers + 64… | code | T0 | `ex01_frozen_llm_spans_a_forty_four_fold_range_of_trained_budget.py` |
| 2 | Implement the gated residual `y = tanh(alpha) * cross + x` in PyTorch. Show experimentally th… | code | T0 | `ex02_the_gate_is_exact_at_zero_and_steepest_there_too.py` |
| 3 | Read OpenFlamingo Section 3.2 (arXiv:2308.01390) on how they handle multiple images in a batc… | code | T0 | `ex03_the_padding_costs_the_vision_tower_and_nothing_else.py` |
| 4 | Why does Flamingo's cross-attention mask let a text token attend to *only the most recent* pr… | explain | T0 | prose, below |
| 5 | In-context few-shot: construct a prompt with 4 examples of "image → color of main object" for… | code | T0 | `ex05_the_answer_token_never_sees_the_example_images.py` |
<!-- generated:end -->

## Answers

Four code solutions and one prose one, all T0 and stdlib. Exercises 3 and 5
arrive at the same 45%-ish figure from opposite ends — batch padding and hidden
example images — which is not a coincidence: both are the cost of encoding
images that the most-recent-image mask will never let a text token reach.

### 1 — "frozen LLM" spans a 44× range of trained budget

**ANSWER: 1.464B trained of 10.464B total — 13.99%.** The frozen LLM is 86.01%.

| adapter | trained share |
|---|---:|
| LLaVA 2-layer projector on a 7B | **0.32%** |
| BLIP-2 bridge on its 8B stack | 2.35% |
| Flamingo-9B | **13.99%** |

**FINDING: all three are described as "a frozen LLM".** A 43.7× range, and
Flamingo trains **65×** the parameters of LLaVA's projector. The phrase names
what is held fixed, not what is being learned.

**FINDING: the resampler the architecture is named for is 0.61% of the model.**
Of the 1.464B trained, gated cross-attention is **95.6%** and the Perceiver
resampler **4.4%**.

**FINDING: one cross-attention block is larger than the whole of BLIP-2's
bridge.** At one block every 4 layers a 32-layer LLM takes 8, **175M** each —
1.15× the 152.2M that Lesson 12.03's arithmetic gives for the entire Q-Former.
Flamingo pays that adapter eight times over.

### 2 — the gate is exact at zero, and steepest there too

**ANSWER: exact, not approximate.** `max |y − x|` is **0.0** and every output
row compares `==` to its input. `tanh(0)` is exactly 0.0 and `x + 0.0*c == x` in
IEEE-754 for every **finite** `c`.

**FINDING: "finite" is load-bearing.** `0.0 * inf` is `nan`, so one non-finite
value anywhere in the cross-attention output makes `y` `nan` with the gate fully
closed. The guarantee belongs to the arithmetic, not to the gate.

**FINDING: the no-op point is the point of fastest change.**

| α | tanh(α) | 1 − tanh²(α) | measured max Δ |
|---:|---:|---:|---:|
| 0.00 | 0.000000 | **1.000000** | 0.000000 |
| 0.01 | 0.010000 | 0.999900 | 0.001969 |
| 1.00 | 0.761594 | 0.419974 | 0.149998 |
| 2.00 | 0.964028 | 0.070651 | 0.189868 |
| 5.00 | 0.999909 | **0.000182** | 0.196935 |

The initialisation chosen because it is a perfect no-op sits on the steepest
part of the curve — a step of 0.01 admits 1.0% of the cross-attention.

**FINDING: the gate has about 2.5 units of usable range.** From α = 2 to α = 5
buys **3.6%** of the gate while the slope falls **5,507×**. The lesson's own
table prints α = 5 as though it were a distinct setting.

### 3 — the padding costs the vision tower and nothing else

**ANSWER: pad the image axis to the batch maximum and carry a mask.** Counts
(1, 3, 5, 2) → a 4 × 5 image tensor: **20** slots for 11 images, **45%**
padding.

**FINDING: the waste is exactly `1 − mean/max`.** (1, 3, 5, 2) and
(10, 30, 50, 20) waste the identical 45%; a batch with equal counts wastes 0%,
however many images each prompt has. The cost depends on the *spread*, not the
size.

**FINDING: none of it is in the attention.** Run over a padded sequence,
`interleaved_mask` attends **0** positions in every padding column, because a
text token only ever reaches a *preceding* image and pads sit after all the
text. The bill is 9 image encodes and **576 of 1,280** resampler latents at 64
per image — computed, then discarded.

**FINDING: sorting the batch halves it.** Bucketing by image count into pairs —
(1, 2) padded to 2, (3, 5) padded to 5 — gives **14** slots and **21.4%**. The
padding strategy is free to fix; the batch sampler is where the cost lives.

### 4 — why only the most recent image

Drawing on **Masked cross-attention for interleaved inputs**, which is where the
lesson states the rule and where `interleaved_mask` implements it.

**The mask constrains cross-attention only.** The frozen LLM's own causal
self-attention over text is untouched, so an earlier image still reaches a later
text token — indirectly, through the text that already attended to it. "Only the
most recent image" is a statement about one hop, not about reachability.

Three reasons it is the right hop to keep:

1. **Per-token cost stops depending on image count.** Exercise 5 measures this:
   every row of the mask sums to exactly **1** at every shot count from 0 to 8,
   so a 9-image sequence costs a text token exactly what a 1-image sequence
   costs it. Full cross-attention would make the mask O(text × images); this
   makes it O(text).
2. **It matches the data.** Interleaved web pages put the caption *after* the
   image it describes, so the most recent preceding image is the one the text is
   about. The mask encodes the annotation convention rather than learning it.
3. **It is what lets test-time shot count exceed training-time shot count.**
   Because the computation a token performs is identical whatever precedes it, a
   model trained on sequences of a few images can be prompted with many. This is
   the property few-shot evaluation depends on, and it is a consequence of the
   mask rather than of scale.

**The tradeoff** is that no single attention step can compare two images.
"Which of these two photographs is bluer?" has to route through text: the model
describes image 1, that description enters the text stream, and the token over
image 2 reads the description rather than the image. Anything that survives the
description survives; anything that does not is gone. The same brittleness shows
up whenever the interleaving convention is violated — a caption placed *before*
its image is associated with the previous one, silently and with no error.

Flamingo accepted that trade because its target was few-shot captioning and VQA,
where each question concerns one image. The 2026 models that need genuine
cross-image reasoning — multi-image comparison, video with temporal grounding —
dropped the mask and went back to putting every image's tokens in the sequence,
paying the O(text × images) attention that Flamingo was built to avoid.

### 5 — the answer token never sees the example images

**ANSWER: it sees exactly one image — its own — at every k from 0 to 8.** The
example images are invisible to the token that produces the answer, so whatever
few-shot learning happens is happening over the example *text*.

**ANSWER: the expected pattern is a step, not a curve.** Text positions grow
1, 3, 5 … 17 while the answer's visual context stays at one image, so nearly all
the gain lands between 0 and 1 shot — where the output format is established as
a bare colour word rather than a sentence. The tail is flat, or drifts down as
the prompt's own label distribution starts biasing the answer.

**FINDING: the mask is 11.1% dense at 8 shots, and every row sums to exactly
1.** 17 text positions × 9 images = 153 cells, **17** True. Cross-attention cost
is linear in text length and independent of image count.

**FINDING: 8 of the 9 images are encoded and then hidden** — **512 of 576**
latents at 64 an image, **88.9%**. Exercise 3's padding waste, arrived at from
the other direction.
