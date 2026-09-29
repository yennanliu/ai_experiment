<!-- generated:start -->
# 19-capstone-projects / 60-projection-layer-modality-align

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/60-projection-layer-modality-align/) · upstream spec
`phases/19-capstone-projects/60-projection-layer-modality-align/docs/en.md`

```bash
uv run demo practice run 60-projection-layer-modality-align --ex 1
uv run demo explain 60-projection-layer-modality-align --ex 1
uv run pytest demos/phases/19-capstone-projects/60-projection-layer-modality-align
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Replace CLS pooling with mean pooling over the 196 patch tokens and compare final loss after… | code | T1 | `ex01_mean_and_cls_pooling_both_end_at_0_832_because_all_32_images_pool_to_the_same_vector.py` |
| 2 | Add a learned scalar temperature to the cosine loss (`cos / tau`) and observe what happens wh… | code | T1 | `ex02_under_adam_a_fixed_tau_from_0_01_to_100_trains_the_same_projector_and_a_learned_tau_only_falls.py` |
| 3 | Swap the two-layer MLP for a single linear layer and quantify the loss gap. The non-linearity… | code | T1 | `ex03_a_single_linear_layer_matches_the_mlp_within_0_001_and_both_emit_one_constant_direction.py` |
| 4 | Add a small L2 penalty on the projector weights and watch how it interacts with cosine alignm… | code | T1 | `ex04_l2_cuts_the_output_norm_3x_leaves_cosine_at_0_186_and_shrinks_used_and_unused_directions_alike.py` |
| 5 | Persist projector weights, then reload and run inference without the vision encoder backward… | code | T1 | `ex05_the_reloaded_projector_is_exact_but_needs_the_seed_0_encoder_train_never_saves.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's own `train()` from `code/main.py` (seed 0, 32
synthetic pairs, 200 steps, Adam 3e-4), swapping one module global at a time:
the encoder's pooling, the loss, or the projector class. One fact sits under
all five answers. The frozen, randomly initialised encoder gives all 32
images CLS vectors with pairwise cosine >= 0.99996, so the projector has no
image information to use. It learns one output direction, and the best
constant direction (the normalised sum of the 32 unit caption vectors)
already scores loss 0.8130.

### 1 — mean and CLS pooling both end at 0.832, because all 32 images pool to the same vector

**No measurable difference.**

| pooling | step-199 loss | last 32 steps | all 32 pairs | first 32 steps |
|---|---:|---:|---:|---:|
| CLS | 0.7984 | 0.8320 | 0.8133 | 1.005 |
| mean of 196 patches | 0.7990 | 0.8322 | 0.8133 | 1.001 |

Mean pooling does not train faster here. Both trained heads sit 0.0003
above the 0.8130 constant-output floor. Nothing is aligned: pair every image
with another pair's caption and the last-32-step loss is 0.8321. The
lesson's "about 0.80" is the loss of one pair at step 199, not the model's
loss.

### 2 — under Adam a fixed tau from 0.01 to 100 trains the same projector, and a learned tau only falls

**tau changes the printed loss, not the training.** With the loss
`1 - cos / tau`:

| tau | last-32 cosine | step-199 loss |
|---|---:|---:|
| 0.01 | 0.1680 | -19.16 |
| 1 | 0.1680 | 0.798 |
| 100 | 0.1680 | 0.998 |

tau = 100 does look like a high plateau, but tau = 0.01 shows no gradient
noise. The gradient is scaled by exactly 1/tau and Adam divides that scale
out. A learned tau (a `log_tau` on the projector, starting at 1) falls
1.000 -> 0.949 after 200 steps and 0.833 after 600, and never rises after
step 50. `1 - cos / tau` has no minimum in tau: from step 200 to 600 the
loss drops 0.824 -> 0.791, while the cosine only moves 0.168 -> 0.175, which
at tau = 1 would score 0.825. The whole drop comes from tau shrinking.

### 3 — a single linear layer matches the MLP within 0.001, and both emit one constant direction

**The gap is within noise.** The step-199 loss is 0.7984 for the MLP and
0.7978 for `nn.Linear(768, 512)`. The last-32-step losses are 0.8320 and
0.8318, and both score 0.8133 on all 32 pairs. The linear layer has 393,728
parameters against the MLP's 1,312,256, so the lesson's "1.3M" is right.
Neither head uses the image: each one's 32 outputs have pairwise cosine
>= 0.9999, and both end within 0.0005 of the constant-output floor. "The
non-linearity matters less on synthetic features" holds here only because
no head can beat a constant.

### 4 — L2 cuts the output norm 3x, leaves cosine at 0.186, and shrinks used and unused directions alike

**The penalty shrinks the weights and leaves the alignment alone.** It is
`lam * (||fc1.W||^2 + ||fc2.W||^2)`.

| lam | mean cosine | embedding norm | fc1 kept, used span | fc1 kept, unused span |
|---|---:|---:|---:|---:|
| 0 | 0.1867 | 51.2 | x1.09 | x1.002 |
| 1e-3 | 0.1867 | 43.9 | x0.70 | x0.61 |
| 1e-2 | 0.1862 | 15.5 | x0.26 | x0.19 |

"Used" is the span of the 32 CLS vectors that fc1 actually sees. "Unused" is
the other 736 input dimensions. The penalty does not "mostly" shrink the
unused directions: it shrinks both at a similar rate. The unused part loses
more only in absolute terms, because at init it holds 96% of fc1's squared
norm. With no penalty it barely moves (x1.002). The gradient there is zero,
and Adam's per-element step is the only thing that moves it.

### 5 — the reloaded projector is exact but needs the seed-0 encoder train never saves

**Verified.** The `state_dict` is a 5.25 MB file with 4 tensors and
1,312,256 parameters. Loaded with `weights_only=True` and run under
`torch.inference_mode()`, it reproduces the trained head's embeddings
exactly (max abs difference 0.0) and scores cosine 0.1867. The output does
not require grad, and none of the encoder's parameters has a `.grad`.

The projector alone is not deployable, though. Put the same config's encoder
built at seed 1 in front of it and the cosine is -0.045. On all-zeros
features it is 0.040. The head reaches its one direction through fc1 acting
on the seed-0 encoder's CLS vector, not through its bias. That encoder
exists only as the `torch.manual_seed(0)` call inside `train()`, so a deploy
must re-create it from the seed or ship its weights too. It is also not the
86M-parameter encoder the lesson describes: `train()` builds a depth-4,
mlp_ratio-2 ViT with 19,501,056 parameters.
