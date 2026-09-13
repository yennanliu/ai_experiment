<!-- generated:start -->
# 07-transformers-deep-dive / 09-vision-transformers

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/07-transformers-deep-dive/09-vision-transformers/) · upstream spec
`phases/07-transformers-deep-dive/09-vision-transformers/docs/en.md`

```bash
uv run demo practice run 09-vision-transformers --ex 1
uv run demo explain 09-vision-transformers --ex 1
uv run pytest demos/phases/07-transformers-deep-dive/09-vision-transformers
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. Verify the number of patches equals `(H/P) * (W/P)` and the flat pa… | code | T0 | `ex01_patchify_builds_a_grid_and_throws_it_away.py` |
| 2 | Medium. Implement 2D sinusoidal positional embeddings — two independent sinusoidal codes for… | code | T0 | `ex02_not_one_band_completes_a_cycle.py` |
| 3 | Hard. Build a 3-layer ViT (PyTorch), train on 1,000 MNIST images with 4×4 patches. Measure te… | code | T1 | `ex03_the_pairing_is_what_makes_it_visible.py` |
<!-- generated:end -->

## Answers

The lesson is a 147-line patchify-and-embed front end with a parameter model
bolted on. Exercise 1's identities hold, and reading the function that produces
them turns up three things it does and then throws away. Exercise 2 asks you to
implement a function that already exists, then compare it in PyTorch. Exercise 3
asks a question whose answer is below its own noise floor.

Exercises 1 and 2 are **T0**; Exercise 3 is **T1** (numpy + scikit-learn).

### 1 — `patchify` builds a grid, then returns a different one

**ANSWER: both identities hold.** 24×24×3 at patch 6 gives **16** patches of
**108** each, against `(24/6)·(24/6)` and `6·6·3`; the sequence is 17 with
`[CLS]`.

**FINDING: `patchify` assembles a grid of `(row, col)` pairs and discards it.**
`grid` and `grid_row` are filled cell by cell through both loops and never reach
the `return`, which recomputes the shape as `(H // patch_size, W // patch_size)`
from scratch.

**FINDING: `C` is assigned and never read.** The channel count the exercise asks
you to verify is computed on line 3 and dropped — the `assert` below it checks
`H` and `W` only, so deleting the line changes nothing.

**FINDING: the `[CLS]` token gets no positional encoding.** `cls_and_pos` writes
`out = [list(cls)]` and adds `pe` only inside the patch loop. Token 0 carries
position from nowhere, while token 1 is exactly its patch embedding plus
`pe[0][0]`. That contradicts the lesson's own parameter model, which bills
`(num_patches + 1) · d_model` — **151,296** at ViT-Base/16 — for learnable
positions that `pos_2d` never produces.

**CONTROL: the parameter model is accurate to 0.14% anyway.**

| | modelled | published |
|---|---:|---:|
| ViT-Base/16 | 86.5M | 86.6M |
| ViT-Large/16 | 304.1M | 304.0M |
| ViT-Huge/14 | 631.7M | 632.0M |

It is right about the *real* ViT, which uses learnable positions — which is how
the line the code contradicts got there.

### 2 — not one band completes a cycle

**ANSWER: the concatenation is exactly separable.** `pe[i][j] · pe[i'][j']`
varies by at most **3.6e-15** within a `(Δrow, Δcol)` offset across 49 offsets:
the row half and the column half share no dimensions, so their dot products add.
Every position has norm `√(d_model/2)` = 4.898979485566 to twelve places.

**FINDING: not one band is periodic on the lesson's own grid.** The fastest band
has a wavelength of **6.28** positions and the grid is **4**: `0 of 12` bands
complete a cycle at `d_model=48`. The whole positional code is a monotone ramp in
each axis.

**FINDING: a real ViT is barely better.** ViT-Base/16's 14×14 grid gets
**17 of 192** bands — 8.9%. The other 91% traverse less than one period across
the whole image, so they carry a slope rather than a position. Sinusoidal coding
was designed for sequences of hundreds; a patch grid is 14 wide.

**FINDING: the comparison is over 0.17% of the parameters.** Learnable positions
cost 151,296 against 86.5M total; sinusoidal costs zero. That is the entire
budget being traded, which is why the ViT paper's own ablation finds the two
within noise of each other.

### 3 — you cannot tell, and the experiment says so out loud

`torch` is absent and MNIST needs a network, so the ViT is built in numpy over
scikit-learn's `load_digits` — 1,000 train, 797 test, 4×4 patches, 3 residual
attention blocks, hand-written backward pass. The MLP sublayer is omitted to keep
the file inside the repo's 150-line ceiling; both arms share the encoder, so the
comparison is unaffected.

| seed | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| difference, points | −0.50 | −0.63 | **+2.13** | +0.25 | −2.13 | **+2.26** |

**ANSWER: 90.42% against 90.65% — +0.23 points, and 3 of 6 seeds say the
opposite.**

**FINDING: the effect sits under the noise floor, and pairing does not lift it
off.** Accuracies spread **1.23 points** across seeds; the per-seed *difference*
has a standard deviation of **1.71** — larger, so holding the seed shrinks
nothing. Six pairs give `t = 0.33` on 5 df against the 2.57 a two-sided 95% test
wants. Run the exercise's own protocol — one run of each arm — and resolving a
gap this size would take about **220 runs per arm**.

During development, three earlier versions of this file that differed only in the
*order the weight matrices are drawn from the generator* measured the same
comparison at −0.86, +1.46 and +2.17 points. Nothing about the experiment changed
except the random stream, and the answer changed sign twice. That is the same
finding from a different direction.

**FINDING: the pretraining corpus is the fine-tuning corpus.** Both arms see the
same 1,000 images; the masked-patch pass just sees them without labels first, so
it cannot add information — at most it moves the starting point. DINOv2's benefit
is millions of *unlabelled* images, and a null result is exactly what dropping
that ingredient predicts.

**CONTROL: the model is real.** 90.42% against a 10% chance floor.
