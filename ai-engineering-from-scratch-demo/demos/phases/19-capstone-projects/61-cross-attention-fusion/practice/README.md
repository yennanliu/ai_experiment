<!-- generated:start -->
# 19-capstone-projects / 61-cross-attention-fusion

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/61-cross-attention-fusion/) · upstream spec
`phases/19-capstone-projects/61-cross-attention-fusion/docs/en.md`

```bash
uv run demo practice run 61-cross-attention-fusion --ex 1
uv run demo explain 61-cross-attention-fusion --ex 1
uv run pytest demos/phases/19-capstone-projects/61-cross-attention-fusion
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a learned tanh gate to the cross-attention residual (the Flamingo trick) and verify train… | code | T1 | `ex01_the_gated_decoder_learns_the_text_first_and_the_image_second_and_its_gate_opens_only_to_0_14.py` |
| 2 | Implement interleaved attention where the same decoder consumes multiple images plus multiple… | code | T1 | `ex02_the_mask_cuts_segment_2_off_image_1_but_self_attention_carries_27pct_back_and_text_before_an_image_goes_nan.py` |
| 3 | Profile the cross-attention vs the self-attention layer at `Nt=64, Nv=576` (a 24x24 grid at h… | code | T1 | `ex03_cross_attention_costs_5_4x_self_attention_and_73pct_of_it_is_the_kv_projection_not_nt_x_nv.py` |
| 4 | Add a query-side dropout on the cross-attention map and measure caption diversity on the demo… | code | T1 | `ex04_map_dropout_takes_greedy_captions_from_1_to_19_distinct_more_varied_than_10_different_images.py` |
| 5 | Swap the cross-attention layer for a Q-Former-style attention block where a fixed 32-token qu… | code | T1 | `ex05_the_32_query_pool_makes_a_decode_step_2_9x_cheaper_but_a_full_pass_at_ex03s_shape_costs_the_same.py` |
<!-- generated:end -->

## Answers

Every exercise imports and runs the lesson's `code/main.py`, a 4-layer
vision-language decoder with causal self-attention, text-to-image
cross-attention, and a feed-forward sub-layer in each block. Everything runs on
CPU with fixed seeds. All five need torch, so they are T1. External facts were
read on 2026-09-29 from the Flamingo paper (https://arxiv.org/html/2204.14198)
and the BLIP-2 paper (https://arxiv.org/html/2301.12597).

### 1 — the gated decoder learns the text first and the image second, and its gate opens only to 0.14

**It converges, and it learns the text-only behaviour first.** A per-layer
`tanh(alpha)` gate with alpha = 0 wraps the lesson's `CrossAttention`. At step
0 the logits are identical for two different images (difference 0.0), and
every cross-attention weight gets a gradient of exactly 0.0. Only the alphas
move. `main.py` has no training loop, so the task is synthetic: count tokens
that the text alone predicts, then an image class that only the image
predicts.

| run | text loss < 0.1 | image loss < 0.1 | image loss when text converges |
|---|---:|---:|---:|
| gated | step 19 | step 34 | 1.256 (chance is ln 4 = 1.386) |
| ungated | step 21 | step 15 | 0.040 |

Both runs end at a text loss of 0.002 and an image loss of 0.001. The gate
settles at |tanh(alpha)| = 0.143 and 0.145 and stays there, so the image path
works through a gate that is mostly closed. The lesson says Flamingo added the
gate "at training-time stability cost". Flamingo says the zero-initialised
gate makes the model match the pretrained LM at the start, "improving training
stability and final performance".

### 2 — the mask cuts segment 2 off image 1, but self-attention carries 27% back, and text before an image goes NaN

**The per-sample `(B, Nt, Nv)` mask works inside cross-attention.** Each text
token sees only the image just before it. The two samples split their 12
tokens differently (6+6 and 4+8). With an all-True mask, the masked forward
matches the reference to 0.0. When image 1 changes, a single masked layer's
output for segment 2 moves by exactly 0.0, against 0.054 unmasked.

**The mask alone does not make segment 2 independent of image 1.** Through the
4-layer decoder, segment 2's logits still move by 0.026, which is 27% of
segment 1's 0.095. Segment 1 reads image 1, and segment 2 reads segment 1
through causal self-attention. A self-attention mask that is also
block-diagonal by segment brings this to 0.0, but only at batch 1:
`CausalSelfAttention` rejects any mask that is not `(n, n)`, so a per-sample
`(2, 12, 12)` mask raises `ValueError`.

A text token that comes before any image gets an all-False mask row. Its
softmax is NaN, and by the last layer 12 of 12 positions are NaN. Zeroing
empty rows fixes it. The lesson's block snippet passes a `cross_mask`, but the
shipped `DecoderBlock.forward` has no such parameter.

### 3 — cross-attention costs 5.4x self-attention, and 73% of it is the K/V projection, not Nt x Nv

**At Nt = 64 and Nv = 576, cross-attention dominates.** The counts are exact
FLOPs from `FlopCounterMode` for one default block (hidden 256):

| sub-layer | FLOPs | vs self-attention |
|---|---:|---:|
| self-attention | 37,748,736 | 1.00x |
| cross-attention, as the decoder calls it | 205,520,896 | 5.44x |
| cross-attention with a precomputed K/V | 54,525,952 | 1.44x |
| feed-forward | 67,108,864 | 1.78x |

The `Nt * Nv` term that the exercise names is only 37.7M, or 18%. Projecting
the 576 image tokens into K and V costs 151.0M, or 73%, and that part scales
with `Nv` alone. Cross-attention costs the same as self-attention at exactly
Nv = Nt = 64 without a cache, and at Nv = D + Nt = 320 with one. The shipped
cache also saves nothing. `VisionLanguageDecoder` rebuilds it on every
`forward`, so a full pass costs 1,275,068,416 FLOPs with `use_cache=True` and
the same with `use_cache=False`. The lesson says K and V are "computed once at
the start of the decode". Wall-clock medians are printed but not asserted.

### 4 — map dropout takes greedy captions from 1 to 19 distinct, more varied than 10 different images

**Under greedy decoding, diversity rises with dropout on the cross-attention
map.** The decoder is the demo's untrained seed-0 decoder. Each setting
samples 20 captions of 9 tokens for one image, with dropout on the
`(B, H, Nt, Nv)` softmax left on while sampling.

| p | distinct / 20 | token disagreement |
|---:|---:|---:|
| 0.0 | 1 | 0.000 |
| 0.1 | 2 | 0.100 |
| 0.3 | 4 | 0.464 |
| 0.5 | 5 | 0.577 |
| 0.9 | 19 | 0.792 |

With temperature-1 sampling the effect cannot be seen: 20 of 20 captions are
already distinct at p = 0, because the untrained next-token distribution is
nearly flat (entropy 6.77 nats against a maximum of 6.93). The added variety
also does not come from the image. Ten different images without dropout give
5 distinct captions with 0.652 disagreement, which is less than one image
gives at p = 0.9. The shipped `cfg.dropout` is applied after the output
projection, not to the map. It defaults to 0.0, the demo runs in eval mode,
and `main.py` never generates text.

### 5 — the 32-query pool makes a decode step 2.9x cheaper, but a full pass at ex03's shape costs the same

**The swap works.** In each layer, 32 learned queries read the image through
one lesson `CrossAttention`, and the text reads those 32 pooled tokens through
a second one. The logits keep their shape (2, 10, 1024) and still depend on
the image. Cached and uncached logits agree to 0.0, and each layer's K cache
shrinks from (2, 8, 197, 32) to (2, 8, 32, 32). The swap adds 1,085,440
parameters.

| layer-0 FLOPs | lesson | Q-Former |
|---|---:|---:|
| Nt=10, Nv=197, whole sub-layer | 56,281,088 | 77,824,000 |
| Nt=10, Nv=197, text side only | 4,638,720 | 2,949,120 |
| Nt=64, Nv=576, whole sub-layer | 205,520,896 | 205,520,896 |
| Nt=64, Nv=576, text side only | 54,525,952 | 18,874,368 |

The pool still projects every image token into K and V, so it only saves work
per decode step, once the pooled tokens are cached. A full pass breaks even at
Nt = Q(2D + Nv)/(Nv - Q), which is exactly 64 at Nv = 576. BLIP-2 runs one
Q-Former before the language model (32 queries of dimension 768,
cross-attention in every other block), not one pool per decoder layer.
