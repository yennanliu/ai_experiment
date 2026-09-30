<!-- generated:start -->
# 19-capstone-projects / 37-loading-pretrained-weights

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/37-loading-pretrained-weights/) · upstream spec
`phases/19-capstone-projects/37-loading-pretrained-weights/docs/en.md`

```bash
uv run demo practice run 37-loading-pretrained-weights --ex 1
uv run demo explain 37-loading-pretrained-weights --ex 1
uv run pytest demos/phases/19-capstone-projects/37-loading-pretrained-weights
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a `dtype` argument to the loader that casts each tensor to a target dtype (`bfloat16`, `f… | code | T1 | `ex01_bf16_halves_the_model_and_still_generates_but_a_cast_then_copy_dtype_argument_is_a_no_op.py` |
| 2 | Add an `expected_layers` argument that refuses to load a checkpoint whose `h.N` indices do no… | code | T1 | `ex02_a_6_layer_checkpoint_loads_into_a_4_layer_model_with_ok_true_and_a_2_layer_one_half_loads_it.py` |
| 3 | Plug the loader into the lesson 35 generation function and produce two side by side samples:… | code | T1 | `ex03_with_lesson_35s_sampler_the_random_init_and_loaded_samples_share_14_of_16_tokens.py` |
| 4 | Add an export path: write the current model state into a fresh safetensors file using the pre… | code | T1 | `ex04_the_export_round_trips_bit_exactly_but_the_shape_check_misses_a_wrong_transpose_on_4_of_16_matrices.py` |
| 5 | Extend `NAME_MAP` to handle the LLaMA naming convention (no biases, RMSNorm, fused qkv layout… | code | T1 | `ex05_the_llama_map_lands_all_15_mapped_tensors_but_ok_stays_false_and_a_gqa_checkpoint_is_refused.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`: `load_safetensors`,
`make_pretrained_to_local` and `make_stub_safetensors`, on the lesson's own
replica `GPTModel`, or on lesson 35's model in exercise 3. No real weights are
downloaded. The fixtures are the lesson's random stub (vocab 256, d_model 192,
4 layers, seed 42) or a small LLaMA-named stub built in the file, all written
to a temp directory. Two published headers were read on 2026-09-29 while
writing the solutions:
https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors and
https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors.

### 1 — bf16 halves the model and still generates, but a cast-then-copy dtype argument is a no-op

**Yes: a float32 model downcast to bfloat16 loads cleanly and generates.** The
loader already assigns with `copy_(tensor.to(target.dtype))`. So the dtype
argument casts the model first, and the loader then casts each tensor as it
assigns it.

| dtype | report | parameter bytes | head tied |
|---|---|---:|---|
| bfloat16 | loaded=52, 0 missing / mismatched | 3,682,560 | yes |
| float16 | loaded=52, 0 missing / mismatched | 3,682,560 | yes |
| float32 | loaded=52, 0 missing / mismatched | 7,365,120 | yes |

The bf16 model returns 32 in-vocab tokens from finite logits.

**The obvious implementation does nothing.** `copy_(tensor.to(bfloat16))` into
a float32 parameter leaves it float32. The parameter holds 0.333984375 for 1/3:
you lose the precision and keep the memory.

**On the stub, bf16 greedy output is not the fp32 output.** The stub's
next-token logits are nearly flat (entropy 5.51 of a possible 5.55 nats). The
smallest gap between the top two logits is 0.0042, and bf16 moves the logits
by 0.0105 (fp16 by 0.0015), so greedy tokens flip. That tells you nothing
about how bf16 would do on trained weights.

### 2 — a 6-layer checkpoint loads into a 4-layer model with ok() True, and a 2-layer one half-loads it

**The guard reads only the names and refuses 4 of 5 checkpoints before any
assignment.** Only the matching 4-layer file loads. Each refused case leaves
all 52 parameters bit-identical. Without the guard:

| checkpoint into a 4-layer model | report | ok() | parameters changed |
|---|---|---|---:|
| 4 layers | loaded=52 | True | 34 |
| 6 layers | loaded=52, unexpected=24 | **True** | 34 |
| 5 layers with h.2 removed | loaded=40, missing=12, unexpected=12 | False | 26 |
| 2 layers | loaded=28, missing=24 | False | **18** |

The loader refuses to assign on a shape mismatch, but not on missing names.
So the lesson's "Always validate the file before any assignment" does not
hold, and a surplus layer is not an error at all.

### 3 — with lesson 35's sampler the random-init and loaded samples share 14 of 16 tokens

**The loader works on lesson 35's model unchanged** (`loaded=52 missing=0
unexpected=0 shape_mismatch=0`, head tied). But lesson 35's `generate`
(seed 0) produces nearly the same sample before and after the load:

| | tokens 1-16 |
|---|---|
| random init | 58 180 187 139 52 235 65 71 **166** 143 220 36 73 **40** 5 248 |
| loaded stub | 58 180 187 139 52 235 65 71 **16** 143 220 36 73 **185** 5 248 |

Lesson 35 initialises with N(0, 0.02), the stub is drawn from the same
distribution, and both give near-uniform next-token distributions (5.49 and
5.51 nats, total variation 0.17). With the same seed the sampler picks nearly
the same tokens, and a seed-1 random init also matches 14 of 16. So the
lesson's gate ("if the post-load samples look like the pre-load samples ...
the mapping silently missed every tensor") would call this correct load a
failure.

**The lesson's own demo passes that gate only because its replica is not
lesson 35.** The replica skips `_init_weights`, so its tied N(0, 1) embedding
makes greedy decoding repeat the last prompt token (entropy about 1e-34).
Any load changes that: a file holding only `wte.weight` (loaded=1,
missing=51) passes the gate too. `quick_generate`'s `seed` does nothing; it
is argmax, the same as lesson 35's `top_k=1`.

### 4 — the export round trips bit-exactly, but the shape check misses a wrong transpose on 4 of 16 matrices

**Zero shape mismatches, and the round trip is exact.** The export writes 52
tensors, transposes back the conv1d-layout names, and leaves out the tied head.
Reloading gives `loaded=52 missing=0 unexpected=0 shape_mismatch=0`, 52/52
parameters bit-identical and the same greedy sample. Re-exporting a
stub-loaded model reproduces the 7,369,512-byte stub byte for byte.

Both shortcuts fail. `save_file(model.state_dict())` raises RuntimeError
because the tied head shares memory, and `.t()` without `.contiguous()` raises
ValueError.

**Zero mismatches does not prove the layout is right.** An export that
forgets the transpose is caught: `shape_mismatch=12` (`c_attn`, `c_fc`,
`mlp.c_proj`). But `attn.c_proj.weight` is square, so a file with just those 4
flipped loads with `shape_mismatch=0` and changes the model's output.

**The Problem section's example would load nothing.** It names
`transformer.h.0.attn.c_attn.weight` of shape `(2304, 768)`. The published
GPT-2 header has no `transformer.` prefix, stores that tensor as [768, 2304],
and has 160 F32 tensors, including one `h.N.attn.bias` mask per layer (these
land in `unexpected`). With the prefix added, the stub loads 0 of 52, and the
loader raises nothing.

### 5 — the LLaMA map lands all 15 mapped tensors, but ok() stays False and a GQA checkpoint is refused

**The lesson has no `NAME_MAP`, so the extension is a `llama_map()` swapped in
for `make_pretrained_to_local`.** `CONV1D_SUFFIXES` is emptied because LLaMA
stores `nn.Linear` layout, and q/k/v are concatenated into one `qkv_proj`
because the lesson's attention is fused. On a 2-layer multi-head LLaMA stub
the report is `loaded=15 missing=6 unexpected=2 shape_mismatch=0`. The fused
matrix equals `cat(q, k, v)` exactly, and the model forwards finite logits.
The gaps come from the architecture, not the naming:

| gap | why |
|---|---|
| `pos_embed.weight` missing | LLaMA uses RoPE, not a position table |
| 5 LayerNorm `shift`s missing | RMSNorm has no shift |
| 2 `mlp.gate_proj` unexpected | SwiGLU has three MLP matrices; the lesson has two |

Loading the RMSNorm weight does not make the model compute RMSNorm. The
lesson's LayerNorm still subtracts the mean, and on a seeded input it differs
from RMSNorm with the same scale by up to 1.06.

**"Fused qkv" is not LLaMA's layout.** The published TinyLlama header has
201 BF16 tensors, with separate `q_proj` [2048, 2048] and `k_proj`/`v_proj`
[256, 2048]: grouped-query attention with 4 KV heads, no biases and an untied
`lm_head`. On a 1-KV-head stub the fused matrix is (96, 64) against the
lesson's (192, 64). The shape check refuses both layers
(`shape_mismatch=2`), and nothing is assigned. The lesson also says its map
is "a dict that the loader iterates". The loader iterates the file's keys
instead.
