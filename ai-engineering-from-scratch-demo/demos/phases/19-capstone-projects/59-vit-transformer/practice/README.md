<!-- generated:start -->
# 19-capstone-projects / 59-vit-transformer

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/59-vit-transformer/) · upstream spec
`phases/19-capstone-projects/59-vit-transformer/docs/en.md`

```bash
uv run demo practice run 59-vit-transformer --ex 1
uv run demo explain 59-vit-transformer --ex 1
uv run pytest demos/phases/19-capstone-projects/59-vit-transformer
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add register tokens (4 learned vectors prepended after CLS) and rerun. Compare attention map… | code | T1 | `ex01_registers_raise_last_layer_entropy_0_0197_nats_less_than_the_0_0201_four_more_tokens_add.py` |
| 2 | Swap pre-LN for post-LN and train for one epoch on a synthetic shape classifier. Observe whic… | code | T1 | `ex02_without_warmup_post_ln_collapses_to_one_class_on_3_of_3_seeds_and_pre_ln_reaches_67pct.py` |
| 3 | Implement causal masking as an `attn_mask` argument so the same block can be reused as a deco… | code | T1 | `ex03_the_mask_matches_sdpa_and_lesson_61_but_nn_multiheadattention_reads_the_same_tril_as_look_ahead.py` |
| 4 | Profile a forward pass at batch sizes 1, 8, 64 with `torch.profiler`. The MLP layer dominates… | code | T1 | `ex04_the_mlp_holds_64pct_of_block_flops_but_only_51_to_59pct_of_wall_time.py` |
| 5 | Replace one attention head's q-k-v projections with a low-rank LoRA adapter, freeze the rest,… | code | T1 | `ex05_lora_on_head_5_trains_7680_weights_but_its_a_matrix_gets_exactly_zero_gradient_on_step_1.py` |
<!-- generated:end -->

## Answers

Every exercise imports and runs the lesson's `code/main.py` (ViT-Base: the
lesson-58 patch front end, 12 pre-LN blocks of 12 heads, GELU MLP at 4x, a
final LayerNorm) on CPU with fixed seeds. All five need torch, so they are T1.

### 1 — registers raise last-layer entropy 0.0197 nats, less than the 0.0201 four more tokens add

**At the demo's untrained weights, four register tokens change nothing
visible.** Mean last-layer attention entropy goes from 5.2795 to 5.2992
nats. Four extra tokens raise the maximum by ln(201/197) = 0.0201, so the
whole rise comes from the longer sequence.

| | tokens | entropy (nats) | / ln(tokens) |
|---|---:|---:|---:|
| no registers | 197 | 5.2795 | 0.99930 |
| 4 registers | 201 | 5.2992 | 0.99923 |

Both maps are already flat. The registers take 2.04% of the attention,
against 1.99% for any 4 tokens under uniform attention. The smoother maps
reported for registers come from trained ViTs, where registers absorb the
high-norm artifact tokens (Darcet et al., "Vision Transformers Need
Registers", https://arxiv.org/abs/2309.16588). The lesson credits the idea
to the DINOv2 paper, but it comes from this later paper.

The rerun also shows that the demo's "CLS norm ... stabilizes at the final
LayerNorm" is not a property of the image. The norm rises in every block,
from 0.53 to 35.80. The final LayerNorm then sets it to sqrt(768) = 27.713,
and a zero image gives the same 27.713.

### 2 — without warm-up, post-LN collapses to one class on 3 of 3 seeds and pre-LN reaches 67%

**Pre-LN trains stably and post-LN does not.** The model is the lesson's
depth-12 encoder at width 64 (32x32 images, patch 8). It trains for one
epoch on 4,096 noisy squares, discs and plus signs: 128 Adam steps at a
constant lr of 1e-3 with no warm-up.

| seed | pre-LN | post-LN |
|---:|---:|---:|
| 0 | 67.4% | 31.4% |
| 1 | 65.6% | 31.4% |
| 2 | 67.8% | 31.4% |

Every post-LN run ends with loss at ln 3 and predicts one class for all 512
test images. Adding a 32-step linear warm-up to the seed-0 post-LN run, with
nothing else changed, gives 64.1%, so the failure happens at the start of
training.

### 3 — the mask matches SDPA and lesson 61, but nn.MultiheadAttention reads the same tril as look-ahead

**`attn_mask` is one `masked_fill` in a forward bound onto the lesson's
modules, and with no mask the block is bit-identical to the lesson's.** With
a (10, 10) `torch.tril` the weights above the diagonal are exactly 0.
Changing tokens 5-9 moves outputs 0-4 by exactly 0; without the mask it
moves output 0 by 0.38. The result matches
`F.scaled_dot_product_attention(is_causal=True)` and lesson 61's
`CausalSelfAttention` to within 1e-6.

"(seq, seq), lower-triangular" does not say which value means "allowed".
`nn.MultiheadAttention` reads True as "blocked". Given the same tril, its row
0 sees only tokens 1-9, rows 0-8 put all their weight on the future, and
row 9 is NaN.

The lesson also says lesson 61's cross-attention "will use a causal mask".
Lesson 61's `CrossAttention.forward` has no mask argument, and its docstring
says cross-attention uses no mask. The causal mask belongs on the decoder's
self-attention.

### 4 — the MLP holds 64% of block FLOPs but only 51-59% of wall time

**The MLP is the larger half of each block, but it does not dominate.**
Profiled on ViT-Base with `record_function` labels on each block's attention
and MLP:

| per image per block | GFLOP | FLOP share | wall-time share (batch 1 / 8 / 64) |
|---|---:|---:|---|
| MLP | 1.859 | 63.9% | 0.51-0.59, lower at larger batches |
| attention | 1.049 | 36.1% | the rest |
| of which score/value core | 0.119 | 4.1% | 17-26% |

The FLOP split is the same at every batch size. The time split was measured
on one M-series CPU and varies from run to run, so the check asserts only a
0.45-0.75 band. The attention core does 4% of the work but takes about a
fifth of the time. Its small batched matmuls, softmax and reshapes run far
below the throughput of the big linear layers, which keeps attention close
to the MLP at 197 tokens.

### 5 — LoRA on head 5 trains 7,680 weights, but its A matrix gets exactly zero gradient on step 1

**Gradient reaches only the adapter, and only head 5 changes.** A rank-8
adapter on head 5's 192 rows of block 0's fused `qkv` trains 7,680 of
85,647,360 weights. After backward, none of the 149 frozen tensors has a
`.grad`. After two Adam steps the `qkv` output differs in exactly 3 of 36
(projection, head) slices: q, k and v of head 5.

Two things are easy to get wrong. With LoRA's B = 0 init, A's gradient is
exactly 0.0 on the first step, and only B learns until it has moved. And
the lesson's layout is `(3, heads, head_dim)`, so head 5's rows are three
separate 64-row slices. Taking rows 960-1151 as one block, as a
`(heads, 3, head_dim)` layout would, adapts k of heads 3, 4 and 5 instead.
