<!-- generated:start -->
# 19-capstone-projects / 58-vision-encoder-patches

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/58-vision-encoder-patches/) · upstream spec
`phases/19-capstone-projects/58-vision-encoder-patches/docs/en.md`

```bash
uv run demo practice run 58-vision-encoder-patches --ex 1
uv run demo explain 58-vision-encoder-patches --ex 1
uv run pytest demos/phases/19-capstone-projects/58-vision-encoder-patches
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Replace the sinusoidal position with a learned `nn.Parameter` and compare the first-epoch los… | code | T1 | `ex01_sinusoidal_drops_to_27_47pct_at_the_new_resolution_and_the_interpolated_learned_table_keeps_100pct.py` |
| 2 | Swap the `Conv2d` for an explicit `nn.Unfold` plus `nn.Linear` and assert the outputs match t… | code | T1 | `ex02_nn_unfold_plus_linear_matches_the_conv2d_to_1_2e_6_and_a_channels_last_weight_flatten_is_off_by_0_63.py` |
| 3 | Add support for non-square patch sizes (e.g. 32x16 for wide-aspect inputs) and verify the pos… | code | T1 | `ex03_sinusoidal_2d_already_handles_a_7x28_grid_and_the_config_and_patchembed_are_what_reject_32x16_patches.py` |
| 4 | Profile the patch step at batch sizes 1, 8, 64. The patch projection is rarely the bottleneck… | code | T1 | `ex04_the_patch_step_is_0_66pct_of_vit_b16_flops_but_costs_1_94x_a_layers_attention_score_matmuls.py` |
| 5 | Train the front end as a frozen feature extractor on a 4-class synthetic shape dataset (circl… | code | T1 | `ex05_the_cls_output_is_the_same_vector_for_every_image_so_a_linear_probe_on_it_scores_25pct.py` |
<!-- generated:end -->

## Answers

Every exercise imports and runs the lesson's `code/main.py` (`PatchEmbed`, a
strided `Conv2d`; `sinusoidal_2d`; `VisionFrontEnd`, which prepends CLS and
adds the fixed position table) on CPU with one thread and fixed seeds. All
five need torch, so they are T1 in the `vision` group.

### 1 — sinusoidal drops to 27-47% at the new resolution, and the interpolated learned table keeps 100%

**Learned wins epoch 1 only when it is initialised at unit scale, and at the
new resolution learned wins.** The task has to need position. One bright 8x8
square sits in one of the 16 patch cells of a 32x32 image, and the label is its
quadrant. The model is the lesson's front end (patch 8, hidden 32) plus one
encoder layer and a linear head on CLS. Mean epoch-1 loss over 5 seeds
(4,096 images per epoch):

| position table | epoch-1 loss | 32 px acc (3 epochs) | 64 px acc |
|---|---:|---:|---:|
| sinusoidal (rebuilt for the 8x8 grid) | 1.394 | 100% | 27.0-47.3% |
| learned, std 0.02 (ViT convention) | 1.399 | — | — |
| learned, std 1.0 (bicubic-resized) | 0.605 | 100% | 100% |

At the conventional 0.02 init both tables are still at chance
(ln 4 = 1.386) after one epoch, so that comparison is a tie. The lesson's
resolution claim comes out backwards. `sinusoidal_2d` encodes the integer
patch index, so on an 8x8 grid index 2 is the top half, not the bottom. The
doc says the table "interpolates cleanly"; in fact it extrapolates.
Bicubic-resizing the sinusoidal table the same way gives 100% too. What
carries the model across resolutions is the interpolation, not whether the
table was learned. The code does not help with the change either:
`PatchEmbed.forward` accepts only its configured size, so the model has to be
rebuilt at 64 px and the trained weights loaded into it.

### 2 — nn.Unfold + nn.Linear matches the Conv2d to 1.2e-6, and a channels-last weight flatten is off by 0.63

**They match to 1.19e-6 in float32 and 1.8e-15 in float64.** At the ViT-B/16
default the float32 gap is 84x inside `main.py`'s 1e-4 bar, on outputs up to
2.02. It is 9.5e-7 on a batch of 4. The outputs are not bitwise equal, because
the kernels sum in different orders. Both modules hold 590,592 parameters.
`nn.Unfold` and the lesson's `unfold_then_linear` agree bitwise, because both
lay columns out as `(c, ph, pw)`, which is `Conv2d.weight.reshape(hidden, -1)`.
The one way to get it wrong is the weight layout. A channels-last `(ph, pw, c)`
flatten runs and has the right shape, but is off by 0.625. On a 230x230 input
both spellings return 196 patches and silently drop 2,724 of 52,900 pixels
per channel. Only `PatchEmbed` refuses the size.

### 3 — sinusoidal_2d already handles a 7x28 grid; the config and PatchEmbed are what reject 32x16 patches

**The table handles it, and once the patch step accepts a (32, 16) kernel the
front end does too.** A (32, 16) `Conv2d` is patched into the lesson's
`VisionFrontEnd` and its table is rebuilt with `sinusoidal_2d(7, 28, 768)`.
On a 224x448 input the front end returns (1, 197, 768). The row half of the
table is constant along rows and the column half along columns, all 196 rows
are distinct (nearest pair 3.23), and a patch at (3, 20) lands in token 105 =
1 + 3 x 28 + 20.

The lesson rejects the input in three places: `FrontEndConfig` (TypeError on
`int % tuple`), `PatchEmbed.forward` ("spatial mismatch") and
`unfold_then_linear` (TypeError). The doc's "position embedding norms are
uniform within a row" holds for every position: each norm is
sqrt(384) = 19.596 in any grid. The 7x28 table is exactly the first 7 rows of
the 28x28 one, because it encodes absolute indices.

### 4 — the patch step is 0.66% of ViT-B/16's FLOPs, but costs 1.94x a layer's attention-score matmuls

**The patch step is not the bottleneck at any batch size.** Per image:

| part | FLOPs | share of one layer |
|---|---:|---:|
| patch projection | 231,211,008 | — |
| QKV | 697,171,968 | 24% |
| output projection | 232,390,656 | 8% |
| MLP | 1,859,125,248 | 64% |
| QK^T and AV | 119,221,248 | 4.1% |

The patch step is 0.66% of patch + 12 layers, whatever the batch. In wall
clock its share was about 0.6%, 0.8% and 7% at batch 1, 8 and 64; the check
requires under 25%. "Attention dominates" is loose, though: inside a layer the
MLP dominates. At 197 tokens the quadratic attention matmuls are only 4.1%, and
the patch projection costs 1.94x them. On this CPU (one thread, torch 2.14,
Apple arm64) the `Conv2d` spelling slows sharply past batch 8. At batch 64 it
is about 11x slower than the lesson's own `unfold_then_linear`, which is why
the batch-64 wall share rises. The doc's reason for the `Conv2d` ("faster on
GPU") does not carry over to CPU.

### 5 — the CLS output is the same vector for every image, so a linear probe on it scores 25%

**No: the CLS output does not separate, because it does not depend on the
image.** Over 600 PIL-drawn shapes (400 train, 200 test) the CLS row differs
by 0.0 and equals `cls_token` exactly. Its position row is zero, and nothing
in the front end mixes patches into it. The probe scores 25.0% on train and
test, which is chance.

| probe features | train | test |
|---|---:|---:|
| CLS token | 25.0% | 25.0% |
| mean of the 196 patch tokens | 44.5% | 42.5% |
| raw mean patch (768 pixels) | 48.0% | 42.0% |

The patch tokens carry some shape, but no more than the pixels. Their mean is
one linear map of the mean patch. Training through CLS cannot fix it either:
a cross-entropy backward from a CLS head leaves the patch `Conv2d`'s gradient
exactly 0. The CLS token becomes a summary only once attention (lesson 59)
lets it read the patches.
