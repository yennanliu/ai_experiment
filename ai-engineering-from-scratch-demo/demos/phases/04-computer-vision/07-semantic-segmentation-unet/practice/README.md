<!-- generated:start -->
# 04-computer-vision / 07-semantic-segmentation-unet

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/07-semantic-segmentation-unet/) · upstream spec
`phases/04-computer-vision/07-semantic-segmentation-unet/docs/en.md`

```bash
uv run demo practice run 07-semantic-segmentation-unet --ex 1
uv run demo explain 07-semantic-segmentation-unet --ex 1
uv run pytest demos/phases/04-computer-vision/07-semantic-segmentation-unet
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Implement `bce_dice_loss` for a binary segmentation task (foreground vs background). V… | code | T1 | `ex01_bce_dice_convergence_speed.py` |
| 2 | (Medium) Replace the `nn.Upsample + conv` up-block with a `nn.ConvTranspose2d` up-block. Trai… | code | T1 | `ex02_transposed_conv_checkerboard.py` |
| 3 | (Hard) Take a real segmentation dataset (Oxford-IIIT Pets, Cityscapes mini split, or a medica… | code | T1 | `ex03_trimap_per_class_dice_gain.py` |
<!-- generated:end -->

## Answers

The lesson's code is correct — `dice_loss`, `combined_loss`, `iou_per_class` and
the U-Net all do what they claim, and exercise 1's new loss reproduces the
lesson's Dice term exactly. What the three exercises turn up is that **two of the
three claims they ask you to verify are conditional on something the exercise
never names**, and the third is unverifiable in this sandbox and unfalsifiable at
this budget.

**1 — the combined loss converges faster, but only under an optimiser the lesson does not use.**

5% foreground is not a knob on `synthetic_segmentation`; the shapes keep a fixed
radius, so the fraction falls out of the image size. `size=86` lands it at
**0.0519** of pixels, and that is where the comparison runs. Since the two arms
minimise different functionals, both are scored on the same held-out foreground
IoU rather than on their own loss values.

| held-out foreground IoU, per epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | mean | epochs to 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SGD, BCE alone (seed 0 / 1) | .000 / .000 | .000 / .000 | .000 / .000 | .062 / .000 | .372 / .617 | .788 / .809 | .871 / .920 | **0.317** | 6, 5 |
| SGD, BCE + Dice | .000 / .000 | .000 / .000 | .000 / .326 | .718 / .741 | .916 / .839 | .922 / .933 | .968 / .913 | **0.520** | 4, 4 |
| Adam, BCE alone | | | | | | | | **0.541** | 4, 5 |
| Adam, BCE + Dice | | | | | | | | **0.511** | 4, 5 |

**ANSWER: under SGD the claim holds, by +0.203 mean IoU and roughly a seed and a
half of epochs.** Same initialisation, same batch order, same data.

**MECHANISM: the majority-class shortcut lowers BCE and *raises* Dice.** Scoring
every constant-logit predictor on the real masks:

| constant predictor | BCE | Dice term |
|---|---:|---:|
| p = 0.5 | 0.6931 | 0.9067 |
| p = 0.0522 (the foreground rate) | **0.2040** | **0.9492** |

Predicting nothing but background captures **71%** of BCE's whole descent to
zero, and makes the Dice term *worse* than the coin-flip predictor did. Autograd
through a constant logit field at exactly p = 0.0519 gives
**d(BCE)/d(logit) = +4.07e-10** against **d(Dice)/d(logit) = −0.02407**: the
shortcut is a stationary point of BCE and is not one of the combined loss.

**CONTROL: under Adam — the optimiser the lesson's own `main()` uses — the
advantage disappears and reverses, −0.030.** Both arms first reach IoU 0.5 in the
same epochs, for both seeds. Dice is a fix for a gradient-*magnitude* imbalance,
and Adam divides gradient magnitude out per parameter before the loss ever sees a
step. The exercise's claim is true, but not of the pipeline it is printed next to.

**CONTROL: "5% of pixels" is a fact about the set, not about any image.**
Per-image foreground runs **0.0261 to 0.0715**, a 2.7x range. Dice is computed
per image, so the imbalance it actually sees is that whole range.

**2 — every transposed variant grows a checkerboard, including the one the textbook rule says is safe.**

`ConvTranspose2d(c, c, k, stride=2, padding=(k-1)//2, output_padding=k%2)` doubles
the side for every k, so all four networks still map to `(12, 3, 64, 64)`.

| up-block | parameters | mean mIoU (2 seeds) | fixed-grid share of Nyquist | boundary/flat Nyquist |
|---|---:|---:|---:|---:|
| bilinear + conv (lesson) | 491,891 | 0.937 | **0.011, 0.028** | 3.89, 5.75 |
| ConvTranspose2d k=2 | 579,171 | 0.855 | **0.862, 0.929** | 0.78, 0.72 |
| ConvTranspose2d k=3 | 687,971 | 0.794 | 0.233, 0.428 | — |
| ConvTranspose2d k=4 | 840,291 | 0.937 | 0.533, 0.993 | — |

**ANSWER: mIoU cannot rank the up-blocks.** The two seeds of one variant differ
by up to **0.244**, against **0.143** between the four means — the comparison the
exercise asks for is entirely inside its own noise, and k=4 lands on the bilinear
block's score to three decimals.

**FINDING: the checkerboard is real and it is only in the transposed versions.**
Split the period-2 (Nyquist) component of the logits into the part that is a
constant offset — a fixed grid — and the part that tracks image content. Bilinear
puts **≤ 0.028** of it in the grid; every transposed run puts **≥ 0.233** there,
and k=2 puts **0.86–0.93**. Its amplitude relative to the logits' own standard
deviation goes **0.0083 → 0.1304**.

**ANSWER: the artifacts sit on flat background, which is exactly what makes them
visible.** Nyquist energy on 2×2 blocks that straddle a class boundary, over
blocks that do not: bilinear **3.89, 5.75** — its high frequencies *are* the
object outlines — against k=2 **0.78, 0.72**, flat across the whole image.

**MECHANISM: uneven overlap is a property of the untrained layer.** Driving each
layer with weights of one and an input of ones, the interior output max/min is
**k=2 → 1.0, k=3 → 4.0, k=4 → 1.0**: Odena's 1/2/4 tiling, exactly.

**CONTROL: "make the kernel divisible by the stride" does not remove the
artifact.** k=2 has perfectly even overlap and still ends with the **largest**
fixed grid of the four — 0.895, against 0.330 for the 4:1-overlap k=3. At stride
2 a 2×2 kernel gives each of the four output phases its own weight and shares
none between them, so once the weights are learned nothing ties the phases
together at all. The overlap argument only ever described initialisation.

**3 — the exercise's target is unreachable, and its last question has an exact answer that needs no training.**

There is no network here, so Oxford-IIIT Pets cannot be downloaded, and
`segmentation_models_pytorch` is not in the `vision` dependency group, so there is
no `smp.Unet` to be within 2 points of. At full scale the run would be
`python train.py --data oxford-iiit-pet --arch unet --size 256 --epochs 60` —
about 1.5 h on one A10G (~$0.40/h spot, ~$0.60 a run), plus ~1 h more for the
`smp.Unet(resnet34, imagenet)` baseline it is meant to match.

What survives is Pets' three-class **trimap** — background, object interior, and a
thin border ring — over *real* photographic pixels cropped from the two
photographs scikit-learn ships, with the lesson's own shapes giving exact ground
truth. Background colour standard deviation **0.336**, against **0.020** for the
lesson's flat green: 17x.

| class | held-out frequency | mean IoU, CE alone | mean IoU, CE + Dice | Δ |
|---|---:|---:|---:|---:|
| background | 0.878 | 0.960 | 0.956 | −0.003 |
| interior | 0.048 | 0.807 | 0.848 | +0.041 |
| border ring | 0.075 | **0.606** | **0.616** | +0.010 |

**ANSWER: the border ring is the hardest class under either loss**, which is the
per-class report the exercise asks for.

**MECHANISM: cross-entropy's gradient share *is* the pixel frequency.** Attribute
the gradient on the logits to the true class of each pixel, at uniform logits:

| | background | interior | border |
|---|---:|---:|---:|
| pixel frequency | 0.878 | 0.048 | 0.075 |
| cross-entropy gradient share | 0.878 | 0.048 | 0.075 |
| `dice_loss` gradient share | 0.680 | 0.131 | 0.188 |
| ratio, Dice over CE | **0.78** | **2.76** | **2.52** |

Cross-entropy matches frequency to **6.0e-08** — which is precisely why a rare
class is under-served. **ANSWER: adding Dice multiplies the interior's share of
the gradient by 2.76 and the border's by 2.52 while cutting background's to
0.78**, because Dice normalises each class by its own area. Those are the classes
that benefit, and the size of the benefit is set by how rare they are.

**CONTROL: 2 IoU points is far below what this recipe reproduces.** mIoU over the
six runs spans **0.6775 to 0.8692** — a **0.1917** spread, **10x** the tolerance
the exercise names. Every per-class Δ from adding Dice (0.003, 0.041, 0.010) is
smaller than that spread, so the training numbers cannot rank the classes; only
the gradient decomposition can.
