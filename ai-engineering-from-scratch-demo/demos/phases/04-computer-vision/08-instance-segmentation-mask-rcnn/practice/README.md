<!-- generated:start -->
# 04-computer-vision / 08-instance-segmentation-mask-rcnn

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/08-instance-segmentation-mask-rcnn/) · upstream spec
`phases/04-computer-vision/08-instance-segmentation-mask-rcnn/docs/en.md`

```bash
uv run demo practice run 08-instance-segmentation-mask-rcnn --ex 1
uv run demo explain 08-instance-segmentation-mask-rcnn --ex 1
uv run pytest demos/phases/04-computer-vision/08-instance-segmentation-mask-rcnn
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Verify your RoIAlign against `torchvision.ops.roi_align` on 100 random boxes. Report t… | code | T1 | `ex01_roialign_parity_and_roipool_drift.py` |
| 2 | (Medium) Fine-tune `maskrcnn_resnet50_fpn_v2` on a 50-image custom dataset (any two classes:… | code | T1 | `ex02_frozen_backbone_finetune_mask_ap.py` |
| 3 | (Hard) Replace Mask R-CNN's mask head with one that predicts at 56x56 instead of 28x28. Measu… | code | T1 | `ex03_mask_head_56_vs_28_ceiling.py` |
<!-- generated:end -->

## Answers

The lesson's `code/` is small and two thirds of it is correct: `roi_align_single`
really does implement RoIAlign, and `freeze_backbone` really does set
`requires_grad = False` on the backbone. What the three exercises find is that
**each of the three numbers they ask you to report is either wrong, unobtainable,
or unable to move** — and that in every case the reason is a detail one layer
below the API.

### 1 — RoIAlign parity and RoIPool drift

The lesson says its kernel and `torchvision.ops.roi_align` "match to within
`1e-5`". On the three boxes in `compare_with_torchvision_roi_align` they do. On
100 random boxes they do not.

| over 100 random boxes (50×50 map, `spatial_scale=1/4`) | |
|---|---:|
| max \|ours − torchvision\| | **0.2509** |
| mean | 0.01003 |
| boxes above 1e-4 | **7** |
| max over the other 93 | **2.52e-05** |

**ANSWER: the verification fails, by four orders of magnitude — but only on 7
boxes in 100.** The other 93 agree to 2.5e-05, which is why three hand-picked
boxes looked fine.

**MECHANISM: the 7 failures are exactly the 7 boxes whose sampling grid leaves
`[0, 49]`.** The two sets are identical, not merely correlated. The worst is box
87 = `[189.8, 187.6, 199.0, 199.0]`, hard against the bottom-right corner, and
its only bad output column is the last one.

**MECHANISM: `F.grid_sample` zero-pads where torchvision's kernel replicates.**
At that cell the sample sits 0.0861 px past `x=49` and 0.0462 px past `y=49`, so
bilinear interpolation puts those fractions of the weight onto nothing:

| | |
|---|---:|
| torchvision (clamps to the border pixel) | 1.9551 |
| predicted `ref × (1−0.0861) × (1−0.0462)` | **1.7042** |
| measured from `roi_align_single` | **1.7042** |

Closed form, to four decimals. **CONTROL:** slide the same box 12 image px (3
feature px) inward — same shape, same size, same aspect — and the difference
drops to **2.15e-05**. Distance to the map edge is the whole variable.

**CONTROL: the agreement that does hold covers one of torchvision's six
configurations.** The from-scratch kernel implements exactly one sample per bin
centre with the half-pixel offset; every other flag setting is off by more than a
unit of signal:

| flags | max \|diff\| |
|---|---:|
| `aligned=True, sampling_ratio=1` | 0.25 |
| `aligned=True, sampling_ratio=2` | 3.19 |
| `aligned=True, sampling_ratio=-1` | 2.30 |
| `aligned=False, sampling_ratio=1` | 3.30 |
| `aligned=False, sampling_ratio=2` | 2.85 |
| `aligned=False, sampling_ratio=-1` | 2.91 |

The RoIPool half of the exercise is a **false premise twice over**. RoIPool
max-pools and RoIAlign bilinearly samples, so subtracting them does not give a
displacement at all. Feed both a *coordinate ramp* — channel 0 holding `x`,
channel 1 holding `y` — and a returned value **is** a feature-map coordinate, so
the difference becomes one. Sliding an 80px box through 41 sub-pixel offsets:

| | mid-map | against the right/bottom edge |
|---|---:|---:|
| peak-to-peak of pool − align | **0.950 px** | **0.950 px** |
| constant part | 1.524 | 1.524 |

**FINDING: RoIPool drifts ~1 feature pixel, and not one bit more at the border.**
The exercise's "on boxes near the border" is wrong: the error comes from rounding
the box, and rounding a box does not know where the box is. The constant 1.524 px
offset is the max-versus-centre convention — max-pooling reports a bin's largest
coordinate, not its centre — which is a convention, not a misalignment.

### 2 — fine-tuning with the backbone frozen

Two departures from the exercise, both forced. The lesson's
`build_custom_maskrcnn` and `load_pretrained_maskrcnn` both hard-code
`weights=...DEFAULT`, so there is **no offline path to this architecture**: every
call fetches a 170 MB checkpoint. Everything here is randomly initialised
instead. And the stock RPN anchors had to be rescaled 4x before training moved at
all.

**ANSWER: the head swap and the freeze leave 41.5% of the detector trainable.**
This is the one result that does not depend on the weights:

| child | trainable / total |
|---|---:|
| backbone (ResNet + FPN) | **0** / 26,854,464 |
| rpn | 1,184,015 / 1,184,015 |
| roi_heads | 17,847,314 / 17,847,314 |
| **total** | **19,031,329 / 45,885,793 (41.5%)** |

The box classifier now emits 3 logits (background + disk + diamond) against
COCO's 91. `freeze_backbone` reaches the ResNet and the FPN and nothing else, so
the RPN trains too.

**ANSWER: mask AP@0.5 = 0.000, while the summed loss falls 4.1x.**

| | AP@0.50 | AP@0.75 | AP@0.90 |
|---|---:|---:|---:|
| after 16 steps, 8 images | **0.000** | 0.000 | 0.000 |

The five losses — `loss_box_reg`, `loss_classifier`, `loss_mask`,
`loss_objectness`, `loss_rpn_box_reg` — are summed exactly as the lesson's
`train_step` does, and the total runs `3.54 → 2.42 → … → 0.89 → 0.87`. This is
the textbook "loss went down, the metric did not move" trap, and the reason is
structural rather than a budget problem: **fine-tuning *is* the transfer, and
there is nothing here to transfer.** The number the exercise asks for cannot be
produced without the checkpoint. At full scale, `python finetune.py --weights
coco --images 50 --epochs 20 --min-size 800` is ~1,000 steps — about 4 GPU-min on
an A10G (~3 cents), or ~3 h on this CPU.

**MECHANISM: every stock anchor here is a low-quality rescue, not a real match.**

| | |
|---|---:|
| best IoU any stock anchor shape reaches on any GT box | **0.56** |
| RPN positive threshold | 0.7 |
| `allow_low_quality_matches` | True |

The stock sizes `(32, 64, 128, 256, 512)` assume 800px inputs; these objects are
16–24 px across, so *no* anchor clears the positive threshold and only torchvision's
low-quality rescue keeps a positive at all. Rescaling the sizes 4x is what makes
the loss move.

**FINDING: the weights are frozen; the BatchNorm buffers are not.**
`maskrcnn_resnet50_fpn_v2` builds its ResNet with **61 real `nn.BatchNorm2d`
layers**, not the `FrozenBatchNorm2d` the v1 model uses, and `requires_grad =
False` says nothing about buffers:

| after 16 training forward passes | |
|---|---:|
| max change in any backbone **weight** | **0.0** (bit-identical) |
| max change in `running_mean` / `running_var` | **29.89** |

The "frozen" backbone's eval-mode output moves with those statistics. A real
freeze needs `model.backbone.eval()` inside the training loop as well.

**CONTROL: the metric is not the problem.** Scoring the ground-truth masks
against themselves at the hardest threshold gives **AP@0.90 = 1.000**, so the
model's 0.000 is the detector, not a broken evaluator.

### 3 — a 56×56 mask head

**ANSWER: the swap is one extra 2x `ConvTranspose2d`, and it costs 2.0x the
parameters and 4x the logit bytes.** The 14×14 patch comes from the lesson's own
`roi_align_single`, at exactly the `output_size` `mask_roi_pool` uses:

| | 28×28 head | 56×56 head |
|---|---:|---:|
| logits per proposal | (3, 28, 28) | (3, 56, 56) |
| `MaskRCNNPredictor` parameters | 263,171 | **525,571 (2.0x)** |
| logit bytes, 100 proposals | 0.94 MB | **3.76 MB (4x)** |

**ANSWER: mAP@0.75 has no room to move.** "Measure mAP@0.75 before and after"
cannot answer the question it is asked to answer, so the explanation is measured
directly instead: push a ground-truth mask through an R×R grid and paste it back
at the object's pixel size — exactly the encode/decode a resolution-R head is
bounded by — and read the IoU that survives. Averaged over a disk, a diamond and
a 5-lobed star:

| box size | 28×28 ceiling | 56×56 ceiling |
|---|---:|---:|
| 24 px | **1.0000** | **1.0000** |
| 64 px | 0.9804 | 0.9986 |
| 128 px | 0.9634 | 0.9852 |
| 256 px | **0.9607** | **0.9806** |

The worst 28×28 number clears IoU 0.75 by **0.21**, so mask resolution never puts
an instance under that threshold, before or after the swap. This is an *oracle*
ceiling — it bounds a perfect predicted mask — so it does not show that mAP@0.75
cannot move. It shows that any movement would come from what the head predicts,
not from the grid it predicts on, which is what the exercise's "explain why" is
actually asking about.

**MECHANISM: the two resolutions separate only on objects bigger than the grid.**
At 24 px both are exactly 1.0000 — a 28×28 grid already over-samples a 24-pixel
box, so the extra deconv is interpolating its own upsampling. At 256 px the
residual error halves, **0.0393 → 0.0194 (2.0x)**.

**FINDING: the threshold that could see this swap sits near IoU 0.97.** Only a
threshold between the two 256 px ceilings separates the heads on the
representation alone. COCO's mAP averages 0.50:0.95 and stops below that, which
is why 28×28 is still torchvision's default despite being visibly coarse on large
objects. The trained comparison, if you want it: `python finetune.py --weights
coco --mask-res 56 --images 50 --epochs 20 --min-size 800`, ~5 GPU-min an arm on
an A10G, about 4 cents.

**CONTROL: the other route to 56×56 sends the bill somewhere else.** Pooling
RoIAlign at 28×28 and keeping the stock single deconv reaches the same output
grid, but then the four 3×3 convs of `mask_head` run at 28×28:

| route to 56×56 | cost |
|---|---:|
| extra deconv | +262,400 parameters, +2.82 MB of logits |
| RoIAlign at 28×28 | mask-head activations **80.3 MB → 321.1 MB** (100 proposals) |

Same output grid, different bill — and on this dataset's 16–24 px objects,
neither one buys a single point of IoU.
