<!-- generated:start -->
# 17-infrastructure-and-production / 09-production-quantization

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/09-production-quantization/) · upstream spec
`phases/17-infrastructure-and-production/09-production-quantization/docs/en.md`

```bash
uv run demo practice run 09-production-quantization --ex 1
uv run demo explain 09-production-quantization --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/09-production-quantization
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. For a 70B model at 128 concurrent with 2k context, compute the total HBM… | code | T0 | `ex01_only_nvfp4_fits_one_h100_and_the_code_calls_it_blackwell_only.py` |
| 2 | You have a 7B coding model. Pick a format and justify. If you were wrong about quality tolera… | code | T0 | `ex02_fp8_is_smaller_than_awq_for_a_busy_7b_coder_and_the_code_has_no_quality_axis.py` |
| 3 | Compute the calibration-dataset size needed to calibrate AWQ for a medical domain model. Why… | code | T0 | `ex03_about_128_in_domain_samples_is_enough_and_padding_with_generic_text_undoes_it.py` |
| 4 | Read the Marlin-AWQ kernel paper or release notes. Explain in three sentences why AWQ hits 74… | code | T0 | `ex04_the_741_and_712_are_both_marlin_on_a_32b_and_the_code_gives_them_the_same_4x.py` |
| 5 | When does it make sense to combine AWQ weights with FP8 KV cache vs keeping KV at BF16? | code | T0 | `ex05_fp8_kv_pays_once_the_kv_cache_outweighs_the_weights_and_the_throughput_model_cannot_see_it.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's own `code/main.py` calculator. Exercise 4 also
uses the benchmark the lesson's numbers come from (Jarvis Labs, "vLLM
Quantization Complete Guide"), the Marlin paper (arXiv:2408.11743), the AWQ
paper (arXiv:2306.00978) and vLLM's quantization docs. Exercise 2 cites the
same benchmark and exercise 3 the AWQ paper. All were fetched 2026-09-26.

### 1 — only NVFP4 fits one H100, and the code calls it Blackwell-only

70B, 128 concurrent, 2k context, from `memory_breakdown`:

| format | weights | KV | total | `gpu_check` |
|---|---:|---:|---:|---|
| BF16 | 140.0 | 68.72 | 212.22 | MULTI-GPU |
| GGUF Q5_K_M | 43.75 | 68.72 | 115.97 | H200 141GB |
| GGUF Q4_K_M / GPTQ / AWQ | 35.0 | 68.72 | 107.22 | H200 141GB |
| FP8 | 70.0 | 34.36 | 107.86 | H200 141GB |
| NVFP4 + FP8 KV | 35.0 | 34.36 | **72.86** | **H100 80GB** |

**One format fits: NVFP4 + FP8 KV.** Three findings sit behind that:

- **The code itself calls that format Blackwell-only.** Its KEY FINDINGS
  line says "Blackwell-only". What fits is really 4-bit weights with 8-bit
  KV. AWQ with FP8 KV (`kv_bits=8`) comes to the same 72.86 GB and runs on
  Hopper, but `FORMATS` has no such row.
- **The lesson's prose budget disagrees with its code.** The KV cache trap
  section says ~20 GB of KV and ~60 GB total, "fits on H100 80GB". The code
  computes 68.72 GB of KV (3.4x more) and puts AWQ on an H200. With
  Llama-3-70B's real 80 layers the KV is 85.9 GB.
- **"Fits" has no headroom.** 72.86 GB is above vLLM's default
  `gpu_memory_utilization` budget of 0.9 x 80 = 72 GB. AWQ with BF16 KV fits
  77 concurrent 2k sequences, not 128.

### 2 — FP8 is smaller than AWQ for a busy 7B coder, and the code has no quality axis

**Pick FP8.** The lesson puts code-gen among the workloads where "quality is
non-negotiable". FP8 is also the *smaller* option at 128 x 2k: 18.22 GB
against AWQ's 25.58, because AWQ keeps 21.73 GB of BF16 KV. That contradicts
the lesson's "memory savings are half of INT4". AWQ is smaller only below 42
concurrent 2k sequences.

**The recovery path is FP8 -> BF16** (36.08 GB) on the same H100. It is a
weight swap, and the BF16 checkpoint is 14 GB on disk. If the tolerance turns
out looser than assumed, AWQ is also a same-GPU move. The trigger has to be
an eval, because `Format`'s five fields contain no accuracy. The lesson's own
benchmark source measures what the calculator cannot hold: HumanEval Pass@1
drops 4.3 points for Marlin-AWQ and 10.4 for Marlin-GPTQ.

### 3 — about 128 in-domain samples is enough, and padding with generic text undoes it

The lesson's code has no calibration, so this runs a scaled-down AWQ scale
search over one layer:

- 16 x 256 weights, INT4 in groups of 128, a 21-point alpha grid.
- 3 salient channels (~1%) that fire on 5% of tokens.
- 8 tokens per sample, with disjoint medical and generic channels.

Error is on medical traffic, relative to round-to-nearest.

| medical samples | 1 | 4 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| relative error | 0.93 | 0.81 | 0.72 | 0.69 | 0.68 | **0.67** | 0.66 | 0.66 |

**128 is the knee**, the smallest size within 2% of the plateau, and it
works out to about 51 observed spikes per salient channel. So the size
follows the outlier rate, tokens ~ 50 / rate. That is why the lesson's
"hundreds" and the skill's "500-2000" are the same rule at different rates.
The AWQ paper itself reports good results from 16 sequences, against GPTQ's
192.

**More data is not better if it is not domain data.** Padding the 128
medical samples with 128, 512 and 1152 generic ones gives 0.74, 0.93 and
0.98, so at 10x the data almost the whole gain is gone. Generic-only
calibration scores 1.18, worse than no calibration. Past the knee, in-domain
data stops paying too: 512 samples beat 128 by 1.5%. The toy's domains share
no salient channels, which makes it harsher than the paper's measured
PubMed/Enron swap (+0.5-0.6 perplexity). Trust the ordering, not the
magnitudes.

### 4 — the 741 and 712 are both Marlin on a 32B, and the code gives them the same 4x

The three sentences:

1. The 712 is not raw GPTQ but GPTQ on the same Marlin kernels (raw GPTQ is
   276.60 tok/s, raw AWQ 67.73), so the comparison is two INT4 checkpoints on
   one kernel family.
2. Marlin reads 4-bit weights and computes against FP16 activations, and its
   paper shows "close to maximum (4x) quantization speedup" up to batch
   16-32. That is why both land together at 1.55-1.61x the FP16 run's 461.04
   tok/s, and why the kernel (10.9x for AWQ, 2.6x for GPTQ) matters far more
   than the algorithm.
3. The remaining 4% is one 200-prompt run that neither source explains. What
   AWQ measurably buys is accuracy: HumanEval Pass@1 of 51.8 against 45.7.

Running the check against the lesson turns up three problems:

- **The code gives all three formats 4.00x FP16.** `relative_throughput` is
  16 / weight_bits, so AWQ, GPTQ and GGUF Q4_K_M come out equal. Measured,
  they are 1.607x, 1.545x and 0.202x.
- **None of the numbers come from a 7B.** The source ran Qwen2.5-32B-Instruct
  on one H200 at concurrency 10.
- **"Best Pass@1 among INT4 formats" is a four-way tie.** AWQ, Marlin-AWQ,
  GGUF Q4_K_M and bitsandbytes all score 0.5183.

The exercise asks to read a paper, but its claims can be measured, so it
ships as code.

### 5 — FP8 KV pays once the KV cache outweighs the weights, and the throughput model cannot see it

**Combine AWQ with FP8 KV once concurrency x context puts more bytes in KV
than in the 4-bit weights.** For 70B that is 133,514 tokens in flight (65
sequences at 2k); for 7B it is 42,220 (21). At 70B, 128 x 2k, FP8 KV makes
three changes:

- Bytes per decode step fall from 103.72 to 69.36 GB, 1.5x fewer.
- The total falls from 107.22 GB (an H200) to 72.86 GB (one H100).
- The H100 concurrency ceiling doubles, from 77 to 154 sequences.

**Keep BF16 KV when KV is small.** One 7B 2k sequence has KV at 4.6% of its
bytes, so FP8 KV saves 2.3%. Keep it too when attention accuracy is the
risk. vLLM's docs flag three risks:

- Sliding-window layers are "more sensitive to KV-cache quantization".
- FP8 KV defaults to unit scales unless calibrated.
- With FlashAttention 3 the queries are quantized to FP8 as well.

The lesson's `relative_throughput` cannot see any of this: it scores AWQ,
AWQ + FP8 KV and NVFP4 + FP8 KV at 4.00x alike. At 70B, 256 x 8k, KV is 94%
of the bytes and FP8 KV reads 1.89x fewer. Even so, that case stays
MULTI-GPU at 313.4 GB. Past that point the lever is concurrency or context,
not precision.
