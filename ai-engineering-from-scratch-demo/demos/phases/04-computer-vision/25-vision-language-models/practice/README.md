<!-- generated:start -->
# 04-computer-vision / 25-vision-language-models

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/25-vision-language-models/) · upstream spec
`phases/04-computer-vision/25-vision-language-models/docs/en.md`

```bash
uv run demo practice run 25-vision-language-models --ex 1
uv run demo explain 25-vision-language-models --ex 1
uv run pytest demos/phases/04-computer-vision/25-vision-language-models
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Run three prompts ("what is this?", "count the objects", "describe the scene") through… | code | T1 | `ex01_cmer_conf_gate_never_fires.py` |
| 2 | (Medium) Fine-tune Qwen2.5-VL-3B or LLaVA-1.6-7B with LoRA (rank 16) on 500 images of a targe… | code | T1 | `ex02_val_split_is_one_class.py` |
| 3 | (Hard) Replace the VLM's image encoder with DINOv3 instead of its default SigLIP/CLIP. Re-tra… | code | T1 | `ex03_mean_pool_discards_position.py` |
<!-- generated:end -->

## Answers

All three exercises name a model that cannot be reached from here.
`transformers`, `llava`, `peft`, `qwen_vl_utils`, `timm`, `open_clip`,
`accelerate`, `bitsandbytes`, `datasets` and `openai` every one raise
`ModuleNotFoundError`, and no weights may be downloaded. What the lesson does
ship is runnable — a `Projector`, a `ToyVLM`, a CMER function and a synthetic
token generator — so each exercise is answered against that code, and answering
it turns up three defects in it: a metric whose confidence term never fires, a
validation split that contains one class, and a forward pass that destroys patch
order before the classifier sees it.

All three run at **T1** (`deps_group: vision`): the lesson's `code/main.py`
imports torch at module level, so importing it at all is a T1 act.

### 1 — CMER's confidence gate never fires

The exercise asks for three prompts on five images. Neither half of that is
possible: no VLM is installable, and handing the exercise's own prompt strings
to the lesson's three entry points raises immediately.

| entry point | given `"what is this?"` etc. |
|---|---|
| `cross_modal_error_rate` | **AttributeError** — `'str' object has no attribute 'norm'` |
| `deepstack_features` | **TypeError** — expected Tensor, got str |
| `ToyVLM.forward` | **TypeError** — `linear()` input must be Tensor, not str |

So the 15 answers are built instead at exact image–text cosines — 5 correct at
**0.95**, 5 partial at **0.45**, 5 hallucinated as random unit vectors — worst
deviation from the target cosine **1.07e-07**. That makes the hand grade a
construction rather than an opinion.

**ANSWER: CMER reads 0.2667 where the hand score says 0.3333.** Four of the five
planted hallucinations are caught; the fifth landed at cosine **0.3072**, above
the `sim_threshold` of 0.25.

**FINDING: "partially correct" has no representation.** Sweeping the partial
answers' cosine moves CMER in one step and never lands between:

| partial cosine | 0.05 | 0.15 | 0.24 | 0.26 | 0.45 | 0.95 |
|---|---:|---:|---:|---:|---:|---:|
| CMER | 0.6000 | 0.6000 | 0.6000 | 0.2667 | 0.2667 | 0.2667 |

A three-way hand grade cannot survive a two-way metric, and where the break
falls is the threshold, not the answer. (0.25 exactly is a float knife-edge, so
the sweep steps over it.)

**MECHANISM: the confidence half of CMER never fires.** Forcing every confidence
to a fixed value gives one step at `text_confidence > 0.8` and nothing else:

| all confidences | 0.0 | 0.5 | 0.8 | 0.81 | 1.0 |
|---|---:|---:|---:|---:|---:|
| CMER | 0.0000 | 0.0000 | 0.0000 | 0.2667 | 0.2667 |

The lesson demo's own confidences run **0.85–0.95**, and this grid reuses them,
so on both fixtures CMER is the similarity term alone.

**FINDING: the 0.25 gate is width-dependent, and 32 dimensions is too few.** On
4,000 random unit pairs — every one a hallucination — CMER should read 1.0000:

| embedding width | 32 | 128 | 512 |
|---|---:|---:|---:|
| CMER on all-hallucinated pairs | **0.9180** | 0.9985 | 1.0000 |
| missed | **0.0820** | 0.0015 | 0.0000 |
| 0.25 in units of 1/sqrt(d) | 1.41σ | 2.83σ | 5.66σ |

The lesson's own 32-d fixture lets **8.2%** of hallucinations through as
grounded.

### 2 — The lesson's validation set is one class

**ANSWER: there is no VLM to fine-tune**, so both arms run on `ToyVLM` —
**6,597** parameters, **6,272** of them the projector. Qwen2.5-VL-3B is reported
at 3e9 parameters (quoted, not measured), so the stand-in is roughly 455,000×
smaller than the model the exercise names.

**FINDING: `main()`'s split puts every validation sample in class 4.**
`synthetic_vision_class_data` emits 5 classes × 40 samples *in order*, and the
lesson slices the result with the prefix `X[:int(0.85*len(X))]`:

| split | validation label histogram | constant-"4" predictor |
|---|---|---:|
| lesson (`X[:170]` / `X[170:]`) | **[0, 0, 0, 0, 30]** | **1.000** |
| stratified, same sizes | [6, 6, 6, 6, 6] | 0.200 |

The `val_acc 1.000` the lesson prints is therefore tied by a model that has
learned nothing but which single class the set contains.

**ANSWER: "zero-shot" reads 0.000 or 0.200 depending only on the split.** The
same untrained `ToyVLM` answers class 2 for all 30 samples — histogram
`[0, 0, 30, 0, 0]` — and the lesson's validation set contains no class 2. On a
stratified split the same model scores chance.

| arm | lesson split | stratified split |
|---|---:|---:|
| zero-shot | **0.000** | **0.200** |
| full fine-tune | 1.000 | 1.000 |
| LoRA rank 16 | 1.000 | 1.000 |

**MECHANISM: accuracy saturates in three steps, because the data is separable
before training.** First step reaching 1.000 is 3 / 3 / 1 / 3 out of 150 across
the four runs, and on the lesson split reaching 1.000 only means learning to
answer 4.

| quantity | measured |
|---|---:|
| between-over-within variance ratio (pooled tokens) | **1401.8** |
| closest two class centres | 6.708 |
| per-patch noise (`0.1 * randn`) | 0.0970 |
| the same noise after mean-pooling 16 patches | **0.0246** |

**CONTROL: LoRA rank 16 is not a parameter reduction at this width.** The
adapter carries **3,584** trainable parameters against the projector's 6,272 —
**57%** — because rank 16 is half of `min(32, 64)`. On a 3B model the same rank
is reported to land well under 1%. With the base weights and the head frozen it
still reaches 1.000.

### 3 — Mean-pooling discards position

**ANSWER: DINOv3 is unreachable**, so the swap is staged with three frozen 32×32
linear stand-ins over the lesson's own tokens, chosen so that rank and
class-direction retention vary independently.

**MECHANISM: an encoder swap is not a drop-in.** `Projector` opens with
`Linear(in_features=32)`, and the lesson's own DeepStack trick concatenates 3 ViT
levels along the channel axis to 96. Feeding `deepstack_features` back into the
projector raises `mat1 and mat2 shapes cannot be multiplied (64x96 and 32x64)`.
"Re-train only the projector" is not optional — a new encoder width forces one.

**ANSWER: counting moves with the encoder, and rank is not the axis it moves
on.** The dense task is 16-patch scenes with k object patches in one half,
built from `synthetic_vision_class_data` itself; the head is frozen, so only the
projector is retrained.

| encoder | rank | ‖Wδ‖ | counting | which-half |
|---|---:|---:|---:|---:|
| `dense_r32` (random rotation) | 32 | 1.000 | **0.867** | 0.600 |
| `keeps_delta_r2` (plane containing δ) | 2 | 1.000 | **0.817** | 0.600 |
| `blind_r2` (plane orthogonal to δ) | 2 | 0.000 | **0.100** | 0.600 |
| majority baseline | — | — | 0.217 | 0.600 |
| uniform baseline | — | — | 0.125 | — |

Dropping 30 of 32 dimensions costs **0.050**; removing the single class
direction at the *same* rank costs everything, landing below both baselines.

**MECHANISM: spatial reasoning improves by exactly 0.000, for any encoder.** All
three score **0.600**, which is precisely the validation majority rate, and all
three answer with a constant — prediction histogram `[60, 0]` in every case.
`ToyVLM.forward` runs `pooled = projected.mean(dim=1)`, so shuffling the 16
patches moves the logits by at most **3.73e-08**: float32 round-off on a
reordered sum, not a tolerance. A mean is symmetric in its arguments, so position
never reaches the head whatever feeds it. This is the lesson's own "Spatial
reasoning is still weak" section reproduced in one line of its own code.

**CONTROL: the signal is in the features — only the pooling loses it.** A plain
`Linear(16*32, c)` on the same `dense_r32` features, differing from `ToyVLM` only
in keeping the patch axis, scores **1.000** on which-half and **1.000** on
counting, against the pooled model's 0.600 and 0.867. The 0.600 is the
architecture's number, not the data's.
