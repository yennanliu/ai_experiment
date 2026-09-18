<!-- generated:start -->
# 12-multimodal-ai / 12-emu3-next-token-for-generation

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/12-emu3-next-token-for-generation/) · upstream spec
`phases/12-multimodal-ai/12-emu3-next-token-for-generation/docs/en.md`

```bash
uv run demo practice run 12-emu3-next-token-for-generation --ex 1
uv run demo explain 12-emu3-next-token-for-generation --ex 1
uv run pytest demos/phases/12-multimodal-ai/12-emu3-next-token-for-generation
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Emu3 produces 4096 tokens per 512x512 image at 8x8 reduction. Compute the equivalent for 1024… | code | T0 | `ex01_a_video_frame_costs_four_times_the_same_sized_still.py` |
| 2 | Read Emu3 Section 3.3 on the video tokenizer. Describe the 3D VQ patch shape and why it is 4x… | explain | T0 | prose, below |
| 3 | Classifier-free guidance weight 5.0 vs 3.0: what visual effect? Trace the math in `code/main.… | code | T0 | `ex03_two_of_the_five_settings_are_not_guidance_at_all.py` |
| 4 | Compute training FLOPs for Emu3-7B at 300B tokens and compare to Stable Diffusion 3. Which wa… | code | T0 | `ex04_the_comparable_is_exact_and_what_is_left_is_the_parameter_count.py` |
| 5 | Emu3 beats SDXL on FID but not on VQAv2 vs specialized VLMs. Explain why the unified-loss app… | code | T0 | `ex05_the_premise_is_backwards_on_the_lesson_own_numbers.py` |
<!-- generated:end -->

## Answers

Four code solutions and one prose one. Two of the four find the lesson's own
numbers saying something other than the exercise assumes: the "comparable"
training budgets are the same number in two units (4), and Emu3 wins the
perception benchmark the exercise says it loses (5).

### 1 — a video frame costs four times the same-sized still

| config | tokens | at 30 tok/s | KV cache |
|---|---:|---:|---:|
| 512×512 | 4,096 | 2:16 | 2.0 GiB |
| 1024×1024 | 16,384 | 9:06 | 8.0 GiB |
| 2048×2048 | **65,536** | **36:24** | **32.0 GiB** |
| video 4 s @8 fps 512×512 | **131,072** | 72:49 | 64.0 GiB |

**ANSWER: 16,384 and 65,536.** Tokens are quadratic in the side, so the latency
table is quadratic too.

**FINDING: the constant rate is the optimistic bound, and it diverges.** An
autoregressive decode of N tokens touches N(N+1)/2 attention pairs — **8.4M** at
4,096, **2.1 billion** at 65,536. A **16×** token count carries **255×** the
attention work, so 36 minutes is a floor and the curve bends away from it exactly
where the exercise is asking.

**FINDING: one 2048×2048 image is 32 GiB of KV cache** — the largest row in the
lesson's own table does not fit on an 80 GiB card beside a 7B model's weights and
activations.

**FINDING: the video rows use a different reduction.** Image rows pass **8**,
video rows pass **4** — a 4× spatial factor before any frame arithmetic. A
4-second 512×512 clip is **2.0×** the 2048×2048 image and **32×** the 512×512
still, and only 8 of that 32 is the video.

### 2 — why the video patch is 4×4×4

Drawing on **The Emu3 tokenizer**, which is where the lesson states the spatial
reduction and the separate time reduction that the video rows of `TokCost` apply.

**The shape is a 3D VQ patch: 4 pixels × 4 pixels × 4 frames, quantized to one
code.** The alternative the exercise names, 8×8×1, is the same *count* of pixels
per token — 64 against 64 — spent entirely on one frame.

**Why the time axis has to be in the patch at all.** If a video tokenizer is
8×8×1, every frame is tokenized independently and the sequence the LM sees is a
concatenation of stills. Two consequences follow immediately, and both are
measurable in exercise 1's arithmetic:

- **Cost is linear in frames with no discount.** A 4-second clip at 8 FPS costs
  32× one frame. With a ×4 time reduction it costs 8×, and the lesson's own
  `TokCost` does exactly this — `frames // time_reduction`.
- **Nothing in a token encodes motion.** Temporal structure has to be recovered
  by attention across token positions, which is the most expensive place to put
  it and the least reliable.

**Why 4×4×4 and not 8×8×8 or 2×2×16.** The patch is a budget split across three
axes, and the three are not interchangeable:

1. **Video is far more redundant in time than in space.** Consecutive frames at
   8 FPS are nearly identical; adjacent 8×8 pixel blocks are not. So the *cheap*
   axis to compress is time, and 4 frames per token is buying redundancy that is
   actually there.
2. **But temporal compression has a hard ceiling that spatial compression does
   not.** Compress 16 frames into one token and a fast motion — a hand moving, a
   cut — is inside a single code and unrecoverable at any downstream layer. Four
   frames at 8 FPS is half a second, which is short relative to most motion and
   long relative to most redundancy.
3. **Spatial compression past 4×4 costs OCR and fine detail** — the same ceiling
   Lesson 12.06 measures for stills. 8×8×1 puts the whole budget on the axis with
   the least redundancy and the most to lose.

**The shape is therefore a statement about where the redundancy is**, not about
the tensor. 4×4×4 keeps the spatial grid at a resolution the tokenizer can
reconstruct, and spends the compression on the axis where four samples genuinely
do carry one sample's worth of information.

### 3 — two of the five settings are not guidance at all

| γ | top prob | margin | entropy |
|---:|---:|---:|---:|
| 0.0 | 0.3084 | 0.0559 | 1.5447 |
| 1.0 | 0.5489 | 0.2160 | 1.0549 |
| **3.0** | **0.7398** | **0.4936** | **0.6293** |
| **5.0** | **0.8438** | **0.6896** | **0.4445** |
| 7.0 | 0.9086 | 0.8175 | 0.3078 |

**ANSWER: 3.0 → 5.0 is +14% on the top probability, +40% on the margin, −29% on
entropy.** The visual effect is fewer and more confident token choices: closer
prompt adherence, less variety between samples, and the saturation and contrast
artefacts of a sampler that has stopped exploring.

**FINDING: the mixing is affine and the effect saturates.** The top-two logit
margin works out to `0.2 + 0.3γ` — a line whose intercept is the *unconditional*
margin — so it grows **1.55×** from 3.0 to 5.0, not the 5/3 a proportional term
would give. The probability margin grows **1.40×**, because a gap of
probabilities can never exceed 1.

**FINDING: two of the five rows are not guidance.** At γ = 1.0 `cfg_mix` returns
the conditional logits exactly; at γ = 0.0, the unconditional ones. The first two
rows are the endpoints being interpolated.

**FINDING: two thirds of the entropy survives the conditional alone.** ln(5) =
1.6094; the conditional sits at **1.0549** (66%). γ 1→3 removes **0.4256** nats,
γ 3→5 removes **0.1848** — the step the exercise asks about is worth less than
half the one before it.

### 4 — the "comparable" is exact, and what's left is the parameter count

**ANSWER: 1.26e22 FLOPs for Emu3-7B; Stable Diffusion 3 is 17% more.** Applying
6ND to a DiT is a stated assumption — its forward pass is the same order as a
transformer's.

**FINDING: the lesson's "comparable" is exact once you convert.** 300M
image-steps × 1,024 latent tokens = **307.2B tokens**, **1.024×** Emu3's 300B.
The two figures in the lesson's parenthetical are the same number in different
units, and it never says so.

**FINDING: so the entire remaining difference is the parameter count.**
1.143 × 1.024 = **1.170**, the FLOPs ratio exactly. Nothing about diffusion
versus autoregression appears in the answer: two model sizes and one unit
conversion.

**FINDING: per image token, the unified model spends far less.** At Lesson
12.10's corpus split as a stated reference, Emu3's image-bearing share is **60%**
— **176M** images against SD3's **300M** steps. Equal total compute, **1.71×**
the image exposure for the specialist. That, and not the loss function, is the
first thing to suspect when a unified model trails on a generation benchmark.

### 5 — the premise is backwards on the lesson's own numbers

| benchmark | Emu3 | rival | relative margin |
|---|---:|---:|---:|
| MJHQ-30K FID vs SDXL | 5.4 | 5.6 | **+3.6%** |
| GenEval vs SDXL | 0.54 | 0.55 | **−1.8%** |
| VQAv2 vs LLaVA-1.6 | 75.1 | 72.4 | **+3.7%** |

**ANSWER: Emu3 *beats* LLaVA-1.6 on VQAv2 and *loses* GenEval to SDXL.** The one
benchmark where the unified model goes backwards is a generation benchmark, not a
perception one.

**FINDING: on a common footing the perception win is the largest of the three.**
A 0.2 on FID, a 0.01 on GenEval and a 2.7 on VQAv2 are three different scales
until they are normalised.

**FINDING: two margins of the same size are described differently.** GenEval's
1.8% is called a "statistical tie"; FID's 3.6% is called a win. Neither has a
variance estimate attached, so the labels differ by **2×** in relative terms and
by nothing in evidence.

**ANSWER: the shape of the margins is the explanation.** The three occupy a
**5.5-point band** centred at 1.8%. A unified loss spends one capacity budget
over the union of the tasks, so it lands near the specialist everywhere and past
it nowhere by much — a band. A specialist produces the opposite signature: one
large margin, and the other benchmarks unreported because it was never trained
for them. The useful question is therefore not which model wins a row, but which
*shape* the reported margins have — and a three-row band centred near zero is
what "unified" looks like when it is working.
