<!-- generated:start -->
# 04-computer-vision / 02-convolutions-from-scratch

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/02-convolutions-from-scratch/) · upstream spec
`phases/04-computer-vision/02-convolutions-from-scratch/docs/en.md`

```bash
uv run demo practice run 02-convolutions-from-scratch --ex 1
uv run demo explain 02-convolutions-from-scratch --ex 1
uv run pytest demos/phases/04-computer-vision/02-convolutions-from-scratch
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Given a 128x128 grayscale input and a stack of `[Conv3x3(s=1,p=1), Conv3x3(s=2,p=1), C… | code | T1 | `ex01_sizes_and_receptive_field.py` |
| 2 | (Medium) Extend `conv2d_naive` and `conv2d_im2col` to accept a `groups` argument. Show that `… | code | T1 | `ex02_grouped_and_depthwise.py` |
| 3 | (Hard) Implement the backward pass of `conv2d_im2col` by hand: given the gradient of the outp… | code | T1 | `ex03_im2col_backward.py` |
<!-- generated:end -->

## Answers

Unlike lesson 01, the lesson's code here is correct — `output_size`,
`receptive_field`, `im2col` and both conv kernels all do what they claim. What the
three exercises find is that **two of the numbers they ask you to compute mean
less than they look like they mean**, and that the one trick exercise 3 names has
a failure mode a natural test would miss.

**1 — the sizes and the receptive field are right, and the receptive field is not what the unit sees.**

| after layer | — | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| spatial size | 128 | 128 | 64 | 64 | **32** |
| receptive field | 1 | 3 | 5 | 9 | **13** |

Both verified: the sizes against an `nn.Sequential` of four real `nn.Conv2d`, and
the receptive field against `autograd` — differentiating one centre output back to
the input marks a **13×13** block of non-zero gradient. Formula and measurement
agree.

**FINDING: inside that 13×13 the unit is nowhere near uniform.** With every weight
set to 1:

| | influence |
|---|---:|
| centre input pixel | **169** |
| corner input pixel | **1** |
| central 5×5 (14.8% of the area) | **42.8% of the total weight** |

The theoretical receptive field is an **outer bound**, not what the unit
integrates. Stacked convolutions weight their input like a binomial kernel — the
count of paths from input pixel to output — so influence concentrates hard in the
middle.

**FINDING: at the border it is not 13 either.** The output at `(0, 0)` reaches a
**7×7** block and sums to **0.209** of the influence the centre unit has. The rest
of its window fell on the zeros `pad2d` inserted. "The receptive field at each
layer" is a statement about **interior units only**.

**CONTROL: the size chain is not invertible.** A 127×127 input ends at **32**,
exactly where 128 does — `output_size` floors. The output shape does not tell you
the input shape, which is why decoders carry a skip connection rather than
recomputing it.

**2 — `groups` is a slice of the channel axis, and it buys FLOPs as much as parameters.**

Both kernels take the argument without either being rewritten — the extension
calls them once per group. Against `F.conv2d(..., groups=g)`:

| groups | 1 | 2 | 3 | 6 |
|---|---:|---:|---:|---:|
| max abs difference | 0.0 | 0.0 | 0.0 | 1.9e-06 |
| parameters | **324** | 162 | 108 | **54** |
| multiply-accumulates | **46,656** | 23,328 | 15,552 | **7,776** |

`324 = C·C·K·K` dense against `54 = C·K·K` depthwise — the factor of `C = 6` the
exercise asks for, because each group only ever sees `C_in/g` input channels.

**MECHANISM: the arithmetic falls by the same factor, which the exercise does not
say.** A group's matmul is `(C/g × C/g·K·K) @ (C/g·K·K × HW)` and there are `g` of
them, so the total scales as `1/g`. **Parameters and FLOPs both drop 6×** — and the
second is why depthwise separable convolutions are on phones.

**CONTROL: "reproduces a depthwise convolution" is about cost, not output.** Write
the same depthwise filters into a dense `C × C` weight with zeros off the diagonal
— **270 of 324** entries zero — and the result is identical to **9.5e-07**. A dense
conv can produce every depthwise output. What `groups` changes is that it stops
storing and multiplying the zeros.

**3 — the backward pass is two transposes and an adjoint, and the trick has a blind spot.**

Forward is `w_flat @ cols`, so:

```
dW    = d_out_flat @ cols.T
dCols = w_flat.T @ d_out_flat
dX    = col2im(dCols)
```

Against `torch.autograd.grad`:

| setting | dx | dw |
|---|---:|---:|
| stride 1, pad 1 | **1.9e-06** | 0.0 |
| stride 3, pad 0 | 0.0 | 0.0 |

**FINDING: swap the `+=` for `=` and the input gradient is simply wrong** — the
identical code with `canvas[window] = patch` misses the true `dx` by **13.2** at
stride 1. Every input pixel is read by several windows, so its gradient is a *sum*
of their contributions; assignment keeps only whichever window wrote last.

**CONTROL: at stride = kernel the same mistake is invisible.** At stride 3 with a
3×3 kernel and no padding the windows tile without overlapping, and `=` scores
**exactly 0.0**. A test written only at that setting would pass the broken
implementation — which is why the exercise names *the overlap* rather than the
accumulation.

**MECHANISM: `col2im` is an adjoint, not an inverse.** Fold a lowered field of ones
straight back and it counts how many windows read each pixel: **9 = K²** in the
interior, falling to **4** at a corner. `im2col` writes **2187** numbers from
**243** pixels — it cannot be inverted, and `col2im` returns the transpose of that
map rather than undoing it.
