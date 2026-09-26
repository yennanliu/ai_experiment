<!-- generated:start -->
# 17-infrastructure-and-production / 07-tensorrt-llm-blackwell

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/07-tensorrt-llm-blackwell/) · upstream spec
`phases/17-infrastructure-and-production/07-tensorrt-llm-blackwell/docs/en.md`

```bash
uv run demo practice run 07-tensorrt-llm-blackwell --ex 1
uv run demo explain 07-tensorrt-llm-blackwell --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/07-tensorrt-llm-blackwell
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. On a 120B MoE with 30% active parameters, compute the memory-bandwidth-li… | code | T0 | `ex01_the_biggest_jump_is_bandwidth_but_two_thirds_of_the_b200_row_is_hardcoded_multipliers.py` |
| 2 | A customer spends $2M/year on H100 + vLLM. What is the break-even number of Blackwell GPUs th… | code | T0 | `ex02_the_savings_fund_40_b200s_the_load_needs_7_by_the_code_and_551_by_the_lesson.py` |
| 3 | You see accuracy drop 3 points on MATH after NVFP4 weight conversion. Name two recovery paths… | code | T0 | `ex03_fp8_weights_cost_2x_and_in_domain_calibration_cuts_nvfp4_error_65_percent_not_to_fp8.py` |
| 4 | Read the MLPerf v6.0 inference results. Which task has the smallest Blackwell-over-Hopper gap… | explain | T0 | prose, below |
| 5 | Compute the HBM needed for a 405B model at NVFP4 weights + FP8 KV cache at 128k context. Does… | code | T0 | `ex05_the_405b_needs_261_gb_under_2_percent_of_a_gb200_nvl72_and_the_code_calls_it_multi_gpu.py` |
<!-- generated:end -->

## Answers

### 1 — the biggest jump is bandwidth, but two thirds of the B200 row is hardcoded multipliers

**The biggest jump is H100 FP8 → B200, 13.76x, and HBM bandwidth is its
largest factor.** Decode on the 120B / 36B-active row:

| stack | tok/s | $/M |
|---|---:|---:|
| H100 BF16 | 47 | 14.93 |
| H100 FP8 | 93 | 7.46 |
| B200 NVFP4 / FP8 KV | 1280 | 1.04 |

BF16 → FP8 on the H100 is 2.0x. Switching the B200 row's factors off one at a
time splits its 13.76x into bandwidth 8.0 / 3.35 = 2.39x, NVFP4 weights 2.0x,
MTP 1.8x and disaggregation 1.6x.

Only 444 of the B200 row's 1280 tok/s is actually memory-bandwidth-limited. The
remaining 2.88x comes from two constants, `mtp_factor` and `disagg_factor`.
Three other things in the module do not hold up:

- **KV precision never reaches the throughput.** `decode_throughput` reads only
  weight bytes, so the row gives 1280 tok/s with 16-, 8- or 4-bit KV.
- **"Closer to 7x after overhead" is the GPU price.** There is no overhead
  term. 13.76x tok/s divided by the 1.92x hourly price is the 7.16x $/M gap.
  The KEY FINDING's printed factors multiply to 17.28, not ~14.
- **GPT-OSS-120B is 5.1B active of 117B (4.4%), not 30%.** At its real size
  every row is 7.06x faster than the one labelled "GPT-OSS-120B".

### 2 — the savings fund 40 B200s; the load needs 7 by the code and 551 by the lesson

**Break-even is 40 B200s; the workload needs 7.** At 7x, the same tokens cost
$285,714, so 12 months of savings is $1,714,286. The lesson gives no purchase
price, so a B200 is costed at the module's $4.80/hour, $42,048 a year. On that
basis the savings pay for 40.8 GPUs. The $2M buys 2.68e11 tokens at the
module's H100 FP8 price, and those need 6.64 B200s. Buying 7 is repaid in 2.1
months.

**The lesson's own prices size the fleet 83x larger.** At the lesson's
$0.09/M, $2M is 22.2T tokens, or 704,662 tok/s, which needs 551 B200s. That is
more than the savings fund. The module's H100 and B200 $/M are 82.9x and 52.1x
the lesson's quoted figures, because it models one unbatched decode stream per
GPU. The answer is 7 or 551 depending on which set of the lesson's numbers you
use.

**The 7x belongs to GB200 NVL72.** The lesson's figures give $0.09 / $0.012
= 7.5x for GB200 NVL72, and only $0.09 / $0.02 = 4.5x for HGX B200. NVIDIA
specifies an NVL72 as 72 GPUs. $1.71M over 72 GPUs is $23,810 each, and at the
module's GB200 price the savings cover 31.6 GPU-years, under half a rack.

### 3 — FP8 weights cost 2x, and in-domain calibration cuts NVFP4 error 65%, not to FP8

**Quality-first:** keep FP8 weights on the Blackwell stack. This costs 2x
NVFP4, $2.08 against $1.04 per million tokens, and 120 GB of weights still fit
in 192 GB. That is still 3.5x cheaper than the lesson's other quality-first
option, H200 FP8 at $7.29/M.

**Cost-first:** keep NVFP4 and choose per-block scales, the clip ratio of
each 16-element block, on in-domain calibration data. The price is unchanged.

MATH cannot be run without a model, so a seeded 32x256 linear layer stands in.
Its test inputs are in-domain activations with 8 hot channels, held out from
calibration:

| format | relative output error |
|---|---:|
| FP8 E4M3, per tensor | 0.0007 |
| FP4 E2M1, per tensor | 0.1964 |
| NVFP4, max-abs block scales | 0.0126 |
| NVFP4, calibrated on generic data | 0.0115 |
| NVFP4, calibrated in-domain | 0.0044 |

Microscaling does most of the work: per-block scales are 15.6x better than one
per-tensor scale. Calibration has to be *in-domain*. Generic inputs recover 9%
of NVFP4's error, in-domain inputs 65%. Even calibrated, NVFP4 stays 6.3x
FP8's error, which is the lesson's "mitigates but does not eliminate". These
are output-error numbers, not MATH points.

The module cannot tell the two paths apart. `Stack` has no activation-precision
field and no calibration field, so the lesson's "FP8 weights + FP4
activations" compromise is priced the same as plain FP8 weights.

### 4 — the lesson's MLPerf link is v5.1, and v6.0 publishes no Hopper baseline to measure a gap against

*Draws on "The numbers you should memorize".*

**The linked post is not v6.0.** The lesson points to NVIDIA's "Blackwell
Ultra Sets New Inference Records in MLPerf Debut". That post covers MLPerf
Inference **v5.1**, published 9 September 2025. It makes two Hopper
comparisons:

- DeepSeek-R1 offline: 5,842 against 1,253 tokens/s/GPU, 4.66x, which the post
  rounds to "about 5x".
- Llama 3.1 405B interactive: ">5x cumulative improvement" over DGX H200.

**What v6.0 says.** MLCommons published v6.0 on 1 April 2026, with 11
datacenter tests, 5 of them new or updated: GPT-OSS 120B, expanded
DeepSeek-R1, DLRMv3, text-to-video and a VLM. The MLCommons announcement names
no GPU models. NVIDIA's v6.0 coverage that I read (StorageReview's summary of
it) lists Blackwell Ultra results only. Its comparisons are against HGX B200
(up to 29% higher iso-GPU throughput on GPT-OSS-120B) and against NVIDIA's own
submissions six months earlier (up to 2.7x on DeepSeek-R1 from TensorRT-LLM
and Dynamo updates). I found no official v6.0 Blackwell-over-Hopper number.

**The only per-task comparison I found is third-party and unverified.**
Spheron's summary gives approximate ("~") per-GPU H200 → B200 ratios:

| task | H200 → B200 |
|---|---:|
| GPT-OSS 120B | 2.5x |
| SDXL | 2.25x |
| Llama 2 70B | 2.24x |
| YOLOv11 | 1.87x |

On that table the smallest gap is **YOLOv11**, which Spheron explains as
compute-bound rather than bandwidth-bound. I could not check these figures
against MLCommons' results tables.

The explanation is consistent with the lesson's own model. Bandwidth-bound LLM
decode gains the HBM ratio (8.0 / 4.8 = 1.67x over H200) plus whatever lower
precision buys. A small compute-bound vision model gains neither. Every
Blackwell-over-Hopper figure above is 1.9-5x per GPU, well below the lesson's
"11-15x per-GPU LLM throughput".

### 5 — the 405B needs 261 GB, under 2% of a GB200 NVL72, and the code calls it multi-GPU

**It fits easily.** At 131,072 tokens the module gives 202.5 GB of NVFP4
weights plus 58.4 GB of FP8 KV, 260.9 GB in total. That is 1.9% of the 13.4 TB
NVIDIA quotes for the rack. It does not fit on one GPU, because the weights
alone exceed 192 GB. It does fit on one 372 GB GB200 superchip, and the rack
has room for 224 more 128k sequences.

- **The module's layer count is invented.** `64 * sqrt(active / 35)` gives
  217.7 layers, 1.73x Llama 3.1 405B's 126. With the real shape (8 KV heads,
  head_dim 128) the KV cache is 33.8 GB, the total is 236.3 GB, and the rack
  holds 390 sequences.
- **`print_stack` would call it "(multi-GPU)" on the GB200 NVL72 row.** That
  stack's `hbm_gb` is 192, one GPU's HBM, and the fit test compares against
  it.
- **At 128k the KV cache is 22% of the bytes each decode step reads.** Every
  step reads the whole cache, so a step reads 1.29x what `decode_throughput`
  charges, and that function takes no context length.
