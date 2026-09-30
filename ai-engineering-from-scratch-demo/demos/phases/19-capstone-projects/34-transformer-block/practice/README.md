<!-- generated:start -->
# 19-capstone-projects / 34-transformer-block

Solutions to all 4 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/34-transformer-block/) · upstream spec
`phases/19-capstone-projects/34-transformer-block/docs/en.md`

```bash
uv run demo practice run 34-transformer-block --ex 1
uv run demo explain 34-transformer-block --ex 1
uv run pytest demos/phases/19-capstone-projects/34-transformer-block
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a `bias=False` flag to every linear in the block. Modern open weights LLMs ship without b… | code | T1 | `ex01_bias_false_saves_82944_of_85m_parameters_and_the_flag_already_ships.py` |
| 2 | Replace `nn.LayerNorm` with a hand rolled RMSNorm and verify the output shape is unchanged. | code | T1 | `ex02_rmsnorm_keeps_the_shape_and_so_does_no_norm_at_all.py` |
| 3 | Add a flag that returns the attention weights for the first head as a `(B, T, T)` tensor. Plo… | code | T1 | `ex03_the_upper_triangle_is_exactly_zero_but_with_dropout_on_v_sees_rows_that_do_not_sum_to_1.py` |
| 4 | Build a sanity check that feeds a `(2, 16, 384)` tensor with `H=6` through both variants and… | code | T1 | `ex04_the_demos_204x_gradient_gap_is_layernorm_eps_and_a_real_loss_gives_1_00x.py` |
<!-- generated:end -->

## Answers

Every exercise imports and runs the lesson's `code/main.py` (a pre-LN/post-LN
`TransformerBlock` with a hand-rolled `LayerNorm`, fused-QKV causal attention
and a GELU MLP) on CPU with fixed seeds. All four need torch, so they are T1.

### 1 — bias=False saves 82,944 of 85M parameters, and the flag already ships

**Twelve blocks at d_model 768 with 12 heads drop 82,944 parameters: from
85,054,464 to 84,971,520, or 0.098%.** Measured against the 124M GPT the lesson
builds next, that is 0.067%.

| linear | biases per block |
|---|---:|
| fused QKV | 2,304 |
| output projection | 768 |
| MLP up | 3,072 |
| MLP down | 768 |
| total | 6,912 |

The flag already exists: `BlockConfig.use_bias` (default `True`) is passed to
all four linears, and after setting it to `False` none of the 48 linears has a
bias. The lesson text never mentions it. Turning it off does not make the block
bias-free, because each `LayerNorm` keeps a learnable `shift`. That leaves
18,432 bias parameters, 22% as many as the flag removes.

### 2 — RMSNorm keeps the shape, and so does no norm at all

**The shape is unchanged: (2, 32, 192) with RMSNorm, the same as with
LayerNorm.** The RMSNorm divides by `sqrt(mean(x^2) + eps)` and has no shift.
It removes 13 x 192 = 2,496 parameters, matches `torch.nn.RMSNorm` to within
1e-6, and gives each output token RMS 1.000 with a mean that is not zero.

The code has no `nn.LayerNorm` to replace. The name appears once, in a
docstring, so the swap targets the lesson's own `LayerNorm` class, which
matches `nn.LayerNorm` to within 1e-6. The shape check itself proves little:
replacing the norm with `nn.Identity` also returns (2, 32, 192). What
separates the three is output RMS, which is 1.000 for LayerNorm and RMSNorm
and 1.25 with no norm.

### 3 — the upper triangle is exactly 0.0, but with dropout on V sees rows that do not sum to 1

**A `return_attn=True` wrapper returns head 0 as (2, 16, 16), and all 240
upper-triangle entries are exactly 0.0.** It captures the softmax with a
forward pre-hook on `attn_dropout`, so no attention math is copied. Every row
sums to 1 to within 1e-6, and the text plot is a clean lower staircase.
Turning the flag on changes the output by 0.0.

With the default `attn_dropout=0.1` in train mode, the tensor that `attn @ v`
actually uses is not a softmax. Here 32 of 272 causal weights are zeroed, and
row sums range from 0.0 (a token that read nothing) to 1.11. The upper
triangle still reads zero, so the mask check passes either way. Only the
pre-dropout capture returns real attention weights.

### 4 — the demo's 204x gradient gap is LayerNorm's eps, and a real loss gives 1.00x

**The sanity check passes and can also fail.** With identical weights and
dropout 0, pre-LN and post-LN blocks on a (2, 16, 384) input with H=6 are not
`allclose`: the maximum difference is 0.58 and the mean 0.07. Two pre-LN blocks
differ by exactly 0.0.

The lesson's evidence for pre-LN does not measure gradient flow:

| embedding gradient | pre-LN | post-LN | ratio |
|---|---:|---:|---:|
| demo loss `sum(final_ln(x)^2)`, eps 1e-5 | 0.002258 | 0.000011 | 203.62x |
| same loss, eps 1e-3 | 0.226 | — | — |
| cross-entropy through a fixed 128-way head | 2.397 | 2.392 | 1.00x |

The demo's loss is B*T*D * var/(var + eps), which comes to 12,287.9 against
12,288: a constant. The gradient that remains is set by eps. The doc also
promises a 12-layer stack "at common learning rates", but `main.py` builds 6
layers and has no optimizer and no learning rate.
