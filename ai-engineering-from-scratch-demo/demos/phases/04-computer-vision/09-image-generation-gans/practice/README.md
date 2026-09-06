<!-- generated:start -->
# 04-computer-vision / 09-image-generation-gans

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/09-image-generation-gans/) · upstream spec
`phases/04-computer-vision/09-image-generation-gans/docs/en.md`

```bash
uv run demo practice run 09-image-generation-gans --ex 1
uv run demo explain 09-image-generation-gans --ex 1
uv run pytest demos/phases/04-computer-vision/09-image-generation-gans
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Train the DCGAN above on the synthetic circle dataset and save a grid of 16 samples at… | code | T1 | `ex01_circularity_by_epoch.py` |
| 2 | (Medium) Replace the discriminator's batch norm with spectral norm. Train both versions side… | code | T1 | `ex02_spectral_norm_vs_batchnorm.py` |
| 3 | (Hard) Implement a conditional DCGAN: feed the class label into both G and D (concat one-hot… | code | T1 | `ex03_label_conditioning_signal.py` |
<!-- generated:end -->

## Answers

All three exercises ask a question whose wording assumes the training worked. On
a CPU budget it mostly does not, and saying so precisely turns out to be the
useful answer. The lesson's `synthetic_circles` gives a target you can score
without an Inception network: its circles fill **19.8%** of the 32×32 frame and
score **0.993** on the IoU between their foreground and the equal-area disc at
its centroid. Every number below is that pair — **size** and **roundness** — read
off the generator's own samples.

### 1 — By which epoch do the circles become circular? By none of them.

Trained at the lesson's own `main()` settings (feat=32, 400 images, batch 32,
Adam 2e-4/(0.5, 0.999), spectral-norm D), with a 4×4 PNG grid of 16 samples
written after every one of 26 epochs.

| epoch | 0 | 4 | 8 | 12 | 13 | 16 | 20 | 25 | real |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| painted area | 1.000 | 0.990 | 0.820 | 0.510 | **0.450** | 0.400 | 0.320 | **0.280** | 0.198 |
| disc-IoU | 0.910 | — | — | — | 0.530 | 0.600 | 0.660 | **0.727** | **0.993** |

**ANSWER: there is no such epoch.** Once the blob stops filling the frame — the
only regime where a shape score means anything — the disc-IoU peaks at **0.727**
and ends at **0.727**, against **0.993** for the real circles. Across the
qualifying epochs it runs 0.53 0.60 0.60 0.60 0.60 0.63 0.64 0.66 0.68 0.68 0.65
0.69 0.73: rising, and nowhere near arriving.

**FINDING: at the lesson's own five epochs there is nothing on the canvas but
paint.** After epoch 4 — the last one `main()` runs — **99.0%** of every frame is
above the background, against 19.8% for the data. Whatever the grid shows at
epoch 5, it is not a circle — and nothing in the loss trace `main()` prints says
so.

**ANSWER: the milestone that is reachable is size.** The painted area first drops
below half the frame at **epoch 13** and ends at **0.280**, only **0.082** from
the real 0.198 against **0.802** at epoch 0. Reseeded, that crossing lands at
epoch **10** and **11** and the final disc-IoU at **0.690** and **0.723** — so the
honest answer to "by which epoch" is a window of three epochs wide, not an epoch.

**MECHANISM: the roundness score alone would have said "epoch 0".** A frame that
is entirely foreground scores **0.910** against its own equal-area disc — within
**0.083** of the real 0.993 — because the disc matching a 1024-pixel area covers
most of a 32×32 square. Any shape metric is degenerate until the silhouette is
the right size, which is why area is reported next to it rather than instead
of it.

**CONTROL: the samples are narrower than the data.** Per-pixel standard deviation
over 32 samples from fixed noise is **0.150** at the last epoch against **0.217**
across 400 real images, **1.4×** wider. Part of the colour-and-position
distribution is covered, not all of it.

### 2 — Spectral norm converges faster; batch norm has the lower variance, and that is the trap.

Both arms are the lesson's own `Discriminator`, with `use_sn` the only difference
— `use_sn=True` drops the BatchNorm2d layers and wraps every conv in
`spectral_norm`. Three seeds each, scaled to feat=16 and 256 images so six runs
fit a CI core. Numbers are the mean of the last three of 20 epochs.

| | D loss | G loss | painted area | seed spread (area) |
|---|---:|---:|---:|---:|
| batch norm | **0.06** | **4.08** | 0.99 / 0.96 / 0.97 | **0.0096** |
| spectral norm | **0.30** | **2.11** | 0.78 / 0.93 / 0.79 | **0.0668** |
| real data | — | — | 0.205 | — |

**ANSWER: spectral norm, on the mean.** Its tail G loss is **1.97** lower and its
tail D loss **5.0×** higher — the discriminator is still being made to work.

**MECHANISM: with batch norm the discriminator wins outright, in every seed.**
All three end with D loss below 0.15 (0.08 / 0.06 / 0.04) and G loss above 3.0
(3.80 / 4.18 / 4.26). D separates real from fake almost perfectly, so G's
gradient is a small number through a saturated sigmoid, and its samples never
leave the paint-the-whole-frame regime — **96%+** of the frame against the data's
20.5%.

**ANSWER: batch norm has the lower seed-to-seed variance — 7× lower — and it is
not a point in its favour.** Its three seeds span **0.022** in painted area
because all three are sitting on the same failure. Spectral norm's **0.144** span
appears because some of its seeds get moving and others do not: its best seed
reaches **0.784** where batch norm's best is **0.964**. Variance is only readable
next to the level it is measured around, and "lower variance" scored on its own
would pick the arm that has learned nothing.

Re-run on seeds 3–5 every claim still holds, but one of the three spectral-norm
runs there finishes at the batch-norm arm's level (tail G loss 4.00 against its
siblings' 2.17 and 1.98) — which is the same spread, seen from the other side.

**CONTROL: the swap is not a capacity change.** 42,961 parameters with batch norm
against 42,769 with spectral norm, a difference of **192** (0.45%): two
BatchNorm2d layers out, a power-iteration buffer per conv in, convs untouched.

**CONTROL: winning this comparison is not solving the task.** The best of all six
runs still paints **78.4%** of the frame against 20.5% for the real circles.
Spectral norm moves the arm that is losing more slowly; neither arm made a
circle.

### 3 — Conditioning works, on colour.

`Generator(z_dim=66)` fed `cat([z, one-hot])`, `Discriminator(img_channels=5,
use_sn=True)` fed the image plus two label planes, 45 epochs on lesson 7's
circles-vs-squares data pooled from 64×64 to 32×32. Lesson 7 paints its circles
(0.9, 0.1, 0.1) and its squares (0.1, 0.2, 0.9) on a green ground, so
`mean(R) − mean(B)` separates the two classes by **0.147**; the same 48 noise
vectors are decoded under both labels and the gap between them is the whole
measurement.

| | untrained | epochs 1–10 | peak (ep 21) | last 10 epochs | real data |
|---|---:|---:|---:|---:|---:|
| class colour gap | **+0.00001** | +0.0015 | **+0.541** | **+0.070** | **+0.147** |

**ANSWER: conditioning works.** The gap over the last ten epochs is **+0.070** —
**47%** of the real class gap — and positive in all ten of them (smallest
+0.006), against **+0.00001** from the identical probe before training. Seeds 1
and 2 give **+0.190** and **+0.189**, so the *size* of the gap is a property of
the draw; only its sign survives a reseed.

**MECHANISM: the generator uses the label because the discriminator does.**
Swapping a real image's label plane for the other moves D's logit **−0.94** on
circles and **+1.83** on squares — one direction per class, on 100% of each
class's images. Split up, **0.89** logits of that is image-by-label interaction
and **1.38** is a flat preference for one plane whatever the image. Only the
interaction term is class-dependent, and only it can make G paint blue when it is
handed the square label.

**FINDING: the label is ignored for the first half of training.** The gap averages
**+0.0015** over epochs 1–10 and first clears 0.05 after **21** epochs (24 and 22
on the other two seeds). Two extra input channels out of 66 change nothing at all
until D itself starts reading its own label planes — the conditioning path has to
be learned from both ends before either end has a reason to.

**CONTROL: the gap is real but uncalibrated.** It peaks at **+0.541**, **3.7×**
the data's own 0.147, and still drops to **+0.006** inside the last ten epochs.
"Conditioning works" here is a claim about sign and ordering, not about matching
the class-conditional distribution.

**CONTROL: what is conditioned is colour, not geometry.** Real circles score
**0.994** disc-IoU and real squares **0.834**; samples drawn under the two labels
score **0.044** and **0.121**, both under the blockier real class. The dataset
separates its classes on two axes and 45 epochs buys exactly one of them.

**On the exercise's own wording.** It asks for "a class embedding channel in D",
and an `nn.Embedding(2, 32*32)` plane was tried first — it is *worse*. On two
seeds it produced separation on one and none at all on the other after the same
45 epochs. Plain one-hot planes, which for two classes carry the same
information, trained consistently on all three. The embedding gives D a more
expressive label path than it can use at this scale, and D is the end of the
chain the generator's signal has to come through.
