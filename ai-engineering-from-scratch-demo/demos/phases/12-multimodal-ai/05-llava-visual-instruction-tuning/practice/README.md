<!-- generated:start -->
# 12-multimodal-ai / 05-llava-visual-instruction-tuning

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/05-llava-visual-instruction-tuning/) · upstream spec
`phases/12-multimodal-ai/05-llava-visual-instruction-tuning/docs/en.md`

```bash
uv run demo practice run 05-llava-visual-instruction-tuning --ex 1
uv run demo explain 05-llava-visual-instruction-tuning --ex 1
uv run pytest demos/phases/12-multimodal-ai/05-llava-visual-instruction-tuning
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Compute the trainable-parameter count for the 2-layer MLP projector at `1024 → 4096 → 4096`.… | code | T0 | `ex01_the_twenty_two_million_is_a_different_encoder.py` |
| 2 | Construct a LLaVA prompt for a "refusal" case — the image contains a private individual. Writ… | explain | T0 | prose, below |
| 3 | Read the AnyRes section of the LLaVA-NeXT blog. Compute the visual token count for a 1344x672… | code | T0 | `ex03_the_only_nonlinearity_in_anyres_is_the_thumbnail.py` |
| 4 | The LLaVA stage-1 projector is trained with LM loss on captions. What happens if you skip sta… | explain | T0 | prose, below |
| 5 | LLaVA-Instruct-150k uses GPT-4 with COCO captions to generate instructions. For a new domain… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

Two code solutions and three prose ones. This lesson is prose-heavy because
three of its five exercises ask for a design or a paper's result rather than a
number; each prose answer names the lesson section it rests on, and the two code
ones measure `main.py` directly.

### 1 — the 22M is a different encoder

**ANSWER: 20,979,712 parameters — 0.161% of LLaVA-13B.** GELU contributes **0**
and the two bias vectors **8,192** (0.04% of the projector). "With GELU and
bias" names two things and only one of them is a number.

**FINDING: the lesson's own "22M" is a 1408-wide encoder, not this one.**

| encoder width | projector params |
|---|---:|
| CLIP ViT-L, 1024 — *what the exercise names* | **20,979,712** |
| SigLIP SO400m, 1152 | 21,504,000 |
| EVA/BLIP-2, 1408 | **22,552,576** — *what the takeaway prints* |

The same 22.55M is what Lesson 12.03 uses for LLaVA's projector, so the two
lessons agree with each other and not with this exercise.

**FINDING: 80% of the projector does not change the dimension.** The second
layer is 4096 → 4096 — **16,781,312** of 20,979,712 — while the first already
performs the only mapping the projector exists for. A single 1024 → 4096 linear
is **5.0×** smaller.

**FINDING: the parameter fraction is the wrong measure.** 0.161% of the model,
and **576** tokens per image — **28.1%** of a 2,048-token context. The cheap
component sets the context bill.

### 2 — a refusal prompt, and what would reinforce it

Drawing on **The prompt format**, which is where the lesson gives the
SYSTEM / USER `<image>` / ASSISTANT template this fills in.

```text
SYSTEM: A chat between a curious human and an artificial intelligence assistant.
        The assistant gives helpful, detailed, and polite answers.
USER: <image>
      Who is this? Give me her full name and what street this was taken on.
ASSISTANT:
```

Expected response — a refusal that still does the answerable part:

> I can describe what's in the photo, but I won't try to identify the person.
> The image shows an adult in a dark jacket standing on a residential street in
> front of a brick building, with a bicycle leaning against the railing behind
> her. I can't determine who she is or where this was taken, and identifying a
> private individual from a photograph isn't something I'll do even where I
> might guess — a wrong guess names the wrong person, and a right one exposes
> someone who didn't publish this. If you have a question about the scene, the
> clothing or the setting, I'm glad to answer it.

**Why it should refuse zero-shot — and why it probably won't.** The refusal is
already in the base LLM: Vicuna inherits it from instruction tuning, and "tell
me this person's name and address" in *text* is a request its policy covers. The
problem is that the image-conditioned path routes around that policy. Stage 2
trains the projector **and the LLM** on LLaVA-Instruct-150k, which is 158k
GPT-4-generated conversations about COCO photographs and contains essentially no
refusals — so the visual instruction distribution has no examples of declining,
and fine-tuning on it moves the model away from the behaviour it arrived with.
The architecture puts the visual tokens *inside* the prompt, at the position
where the policy would be applied, which is precisely where an unaligned
distribution does the most damage.

**What training data would reinforce it.** Three kinds, and the second and third
are the ones people forget:

1. **Refusal pairs** — (photograph of a private individual, identification
   request, refusal-plus-description response), in the lesson's own template so
   the format matches stage 2 exactly.
2. **Near-miss positives on the same images** — "what colour is her jacket?",
   "how many bicycles are there?" — answered normally. Without these the model
   learns *face ⇒ refuse* and becomes useless for any question involving a
   person, which is a large fraction of what a VLM is asked.
3. **Boundary cases** — a public figure at a press event, a crowd where no one
   is the subject, a person whose face is not visible. These are what turn a
   keyword-like trigger into a policy.

The measurable requirement is that refusal rate rises on (1), stays near zero on
(2), and that general VQA accuracy is unchanged — a safety set that does not
report the second and third numbers has not shown it did anything but break
the model.

### 3 — the only non-linearity in AnyRes is the thumbnail

**ANSWER: 5,184 tokens** — 8 tiles + 1 thumbnail at 576 each — against **576**
at base. A 9× increase for an 8× increase in pixels.

| config | tiles | tokens | × base | × pixels | thumbnail share |
|---|---:|---:|---:|---:|---:|
| 336×336 | 1 | 576 | 1 | 1 | — |
| 672×336 | 2 | 1,728 | 3 | 2 | 33.3% |
| 672×672 | 4 | 2,880 | 5 | 4 | 20.0% |
| **1344×672** | 8 | **5,184** | **9** | **8** | 11.1% |
| 1344×1344 | 16 | 9,792 | 17 | 16 | 5.9% |

**FINDING: the token count is exactly `(pixel ratio + 1) × 576`.** AnyRes is
linear in pixels plus one constant tile; the thumbnail is the whole of the
non-linearity, and its share of the bill falls from 33.3% to 5.9% as tiles grow.

**FINDING: the lesson's own takeaway is 3.4× below its own table.** It prints
"AnyRes: up to 2880 tokens". 2,880 is the 672×672 row, three above the last.

**FINDING: `visualize_context` reports a 253% overrun as "remain 0".** At 5,184
visual tokens, captured from the lesson itself:

```text
window   2048: image 253.1% | text  1.5% | remain      0 tokens
```

The remainder is clamped with `max(remain, 0)`, so a configuration that cannot
fit prints as one that exactly fills.

### 4 — what happens if stage 1 is skipped

Drawing on **Stage 1: projector alignment**, which is where the lesson describes
training the projector alone on captions before touching the instruction data.

**ANSWER: nothing bad — that is the Prismatic ablation's actual finding, and it
is the opposite of what the question's phrasing expects.** Karamcheti et al.
(arXiv:2402.07865) ablate the multi-stage recipe directly and report that the
projector-alignment stage can be removed entirely: single-stage training, with
the projector and the LLM updated together on the instruction mixture from
step 0, matches the two-stage recipe on their evaluation suite while cutting
roughly a fifth to a quarter of the training compute. The stage that the LLaVA
paper presents as necessary turns out to be, under their controlled comparison,
a cost with no measured benefit.

**The mechanism the stage was supposed to serve.** A randomly initialised
projector emits visual tokens that are noise in the LLM's embedding space, and
the worry is that backpropagating through them early damages the LLM's language
ability before the projector becomes useful. Stage 1 freezes everything but the
projector so that mismatch is absorbed by the one module that should absorb it.

**Why removing it is safe in this recipe, and where it would not be.** The
instruction mixture already contains caption-like examples, so the alignment
signal is present in stage 2 anyway; and a warmup plus a low LR gives the LLM
time to adapt rather than be dragged. The dependency to watch is *what else is
trainable*. LLaVA finetunes the LLM in stage 2, so there is a large trainable
surface to absorb the early noise. In a recipe that keeps the LLM **frozen** —
MiniGPT-4, or any adapter-only setup — the projector is the only thing that can
move, and giving it a clean, single-objective stage to converge in is doing real
work. The ablation is a statement about LLaVA's recipe, not about alignment
stages in general.

The wider point is the one worth keeping: LLaVA's two-stage structure was
published as an architecture decision and copied as one. It took a paper whose
entire purpose was controlled ablation to find that one of the two stages was
not load-bearing — which is a good argument for reading the ablation papers and
not the method papers.

### 5 — a four-step domain instruction pipeline

Drawing on **Stage 2: visual instruction tuning**, which is where the lesson
describes LLaVA-Instruct-150k being generated by GPT-4 from COCO captions.

The trick that makes LLaVA's pipeline work is easy to miss: **the generator
never sees the image.** GPT-4 is handed a symbolic description — captions plus
bounding boxes — and writes questions and answers about it. That is what makes
the data cheap, and it is also the source of every failure below.

| # | Step | What it produces |
|---|---|---|
| 1 | Source paired data | images + expert text: radiology reports, or satellite tiles with OSM / land-use labels |
| 2 | Symbolise | turn the expert text into a structured scene description — findings list, regions, boxes, measurements — that a text-only model can read |
| 3 | Generate | prompt a strong LLM with that description plus seed examples, in three flavours: conversation, detailed description, complex reasoning |
| 4 | Filter and format | dedupe, validate against the symbolic source, drop answers referencing anything not in it, render into the LLaVA template |

**What goes wrong at each step.**

**1 — the source text answers a different question than the images do.**
Radiology reports describe what is *abnormal*, not what is present: a normal
left lung is simply unmentioned. Train on that and the model learns that
everything worth saying is a finding, which is exactly the prior you do not want
in a diagnostic aid. Satellite labels have the mirror problem — OSM is stale
relative to the imagery, so the "ground truth" describes a building that was
demolished before the tile was captured.

**2 — symbolisation bounds the instruction distribution by the annotation
schema.** Whatever the report does not encode, the generator cannot ask about,
so the model is never trained to answer it. In COCO this is tolerable because
the schema is broad and shallow. In a domain with a narrow schema, the
instruction set inherits that narrowness and the resulting model looks capable
inside the schema and blank outside it — and no benchmark built from the same
schema will show this.

**3 — the generator hallucinates consistently.** Because it cannot see the
image, it invents details that fit the description: laterality flips, a lesion
given a size the report never stated, a plausible-sounding count. Nothing
downstream can catch this, since every later stage also only sees the
description. Style collapse arrives at the same time: one generator, one voice,
and a model that learns a register rather than a skill. For medicine that is the
dangerous outcome specifically — confident radiologist prose with no evidence
behind it is worse than uncertain prose.

**4 — filtering against the source that generated the data validates nothing.**
It is circular by construction: the only check available is whether the answer
is consistent with the description, which is what the generator was optimising
for. And confidence-based filtering quietly removes the hard cases, so the
training set gets easier, the eval set (built the same way) gets easier with it,
and the reported number improves while the model does not.

**The one step worth adding.** A holdout of *human-verified* pairs — small,
expensive, and read against the images rather than the reports — used only for
evaluation and never for training. It is the only part of this pipeline that
sees a pixel and a label at the same time, and without it none of the numbers
above are measuring the thing they claim to.
