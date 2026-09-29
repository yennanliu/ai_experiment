<!-- generated:start -->
# 19-capstone-projects / 07-end-to-end-fine-tuning-pipeline

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/07-end-to-end-fine-tuning-pipeline/) · upstream spec
`phases/19-capstone-projects/07-end-to-end-fine-tuning-pipeline/docs/en.md`

```bash
uv run demo practice run 07-end-to-end-fine-tuning-pipeline --ex 1
uv run demo explain 07-end-to-end-fine-tuning-pipeline --ex 1
uv run pytest demos/phases/19-capstone-projects/07-end-to-end-fine-tuning-pipeline
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run SFT-only vs SFT+DPO vs SFT+GRPO on the same task-specific benchmark. Report which prefere… | code | T0 | `ex01_grpo_beats_dpo_by_7_7_points_at_the_lessons_beta_0_08_and_by_0_6_once_beta_is_swept.py` |
| 2 | Swap Llama 3.3 8B for Qwen3 14B. Measure the $/1M tokens at matched quality. | code | T0 | `ex02_qwen3_14b_costs_0_52_per_1m_tokens_not_the_0_28_the_pipeline_prints_for_both.py` |
| 3 | Measure EAGLE-3 acceptance rate on domain data vs generic ShareGPT. Report the delta and what… | code | T0 | `ex03_a_sharegpt_draft_accepts_0_93_on_generic_and_0_55_on_domain_text_a_2x_latency_miss.py` |
| 4 | Inject 1% of contamination (leak MMLU-Pro answers into training data) and rerun eval. Watch M… | code | T0 | `ex04_one_pct_of_training_leaks_20pct_of_the_benchmark_and_the_lessons_check_still_says_clean.py` |
| 5 | Add LoRA SFT as an alternative to full fine-tune. Measure the quality gap at 10x lower memory. | code | T0 | `ex05_bf16_lora_caps_at_8x_less_memory_and_at_10x_a_4_bit_base_loses_0_198_agreement.py` |
<!-- generated:end -->

## Answers

The lesson's `code/main.py` is a pipeline scaffold. Each stage returns a
hard-coded payload, which is content-hashed into a manifest. Nothing trains,
serves or evaluates. So every exercise runs the measurement for real at toy
scale in numpy, and runs the reference pipeline beside it to show what it
reports instead. Two external facts were read on 2026-09-29: Qwen3-14B's
parameter count (from its Hugging Face model card) and MMLU-Pro's size
(arXiv:2406.01574).

### 1 — GRPO beats DPO by 7.7 points at the lesson's beta 0.08, and by 0.6 once beta is swept

**GRPO wins.** The task is a 4-way toy with 300 prompts, scored on 4,000
held-out prompts and averaged over 5 seeds. SFT is trained on demonstrations
that are 40% wrong.

| recipe | accuracy |
|---|---:|
| SFT only | 0.743 |
| SFT + DPO, beta 0.08 (the lesson's) | 0.858 |
| SFT + DPO, beta 1.0 (best of 0.08 / 0.3 / 1 / 3) | 0.929 |
| SFT + GRPO | 0.935 |

At a fixed step budget, beta also scales the DPO gradient, so the lesson's
small beta trains slowly. Sweeping beta, as Build It step 4 says to, closes
most of the gap. The win is not free either. GRPO makes 720,000 reward calls
where DPO uses 300 preference labels, 2,400 times as many.

**The reference pipeline cannot run this ablation.** `PIPELINE` has 8 stages
and none of them is GRPO. The only eval reads `dpo_checkpoint`, and
`cfg["dpo_beta"]` is ignored: at beta 0.5, all 8 hashes are unchanged.

### 2 — Qwen3 14B costs $0.52 per 1M tokens, not the $0.28 the pipeline prints for both

**$0.516 per 1M tokens, which is 1.84x Llama's $0.28.** Decode at batch 32 is
bound by memory bandwidth, so the pipeline's own anchor is scaled by weight
bytes. That anchor is 4.6 GB for 8.03B parameters (4.58 bits per parameter),
6,400 tokens/s and an implied $6.45/hour. At the same quant, Qwen3 14B (14.8B
parameters) needs 8.48 GB and serves 3,472 tokens/s. "Matched quality" can
only mean the same quant recipe, because the pipeline prints the same quality
for both models.

**Swapping the base changes nothing downstream.** The endpoint hash is
identical: $0.28, quants of 4.6/4.8/5.1 GB, and the same eval deltas.
Fitting 14.8B parameters into 4.6 GB would take 2.49 bits per parameter. The
model card still reruns `config/llama3.3-8b-domainX.yaml`. The lesson's
default, "Llama 3.3 8B", is not an open-weights release: Meta published Llama
3.3 at 70B only.

### 3 — a draft trained on ShareGPT accepts 0.93 on generic text and 0.55 on domain text

**The delta is -0.375: acceptance is 0.926 on generic text and 0.551 on domain
text.** This is real accept/resample speculative decoding with k = 4, and a
draft step costs 1/32 of a target pass.

| draft / text | acceptance | tokens per target pass | speedup |
|---|---:|---:|---:|
| ShareGPT draft / generic | 0.926 | 4.32 | 3.83x |
| ShareGPT draft / domain | 0.551 | 2.11 | 1.88x |
| draft refit on domain mix / domain | 0.777 | 3.23 | 2.87x |

For latency budgets, a budget sized on ShareGPT underestimates domain decode
time by 2.04x. Refitting the draft on domain data cuts that to 1.33x.

**The lesson's 0.74 is a constant.** New data leaves the endpoint hash
unchanged. The lesson's own definition, "fraction of drafted tokens the
target model accepts", also gives a different number: with 4 tokens drafted
per step, it is 0.829 on generic text and 0.279 on domain text. An
acceptance figure needs its definition and its k attached.

### 4 — 1% of the training data leaks 20% of the benchmark, and the lesson's check still says clean

**Benchmark accuracy jumps from 0.443 to 0.571.** Leaking 1% of the training
set (200 of 20,000 documents, half of them lightly edited) exposes 20% of a
1,000-item benchmark. The gate is MinHash LSH over word 3-grams: 64 hashes in
32 bands of 2, confirmed at Jaccard >= 0.5. It passes the clean corpus with 0
flags. It fails the leaked corpus with 200 of 200 leaks caught, including
100 of 100 edited ones, and no false positives. Dropping what it flags
restores 0.443.

Banding matters. With the same 64 hashes in 16 bands of 4, the gate catches
only 95 of the 100 edited leaks, and accuracy after it is 0.445.

**The lesson's `stage_contamination` cannot fail.** It hard-codes
`overlap_examples: 0`, and the dataset artifact holds counts, not text. At
the lesson's scale, 1% of its 255,336 kept examples is 2,553 documents,
which is 21.2% of MMLU-Pro's 12,032 questions.

### 5 — LoRA on a bf16 base cannot reach 10x lower memory; at 10x a 4-bit base loses 0.198 agreement

**At 10x lower memory the quality gap is 0.198 top-1 agreement.** The toy is a
128x128 layer that distils a rank-4-plus-noise update. Memory counts weights,
gradients and Adam state.

| method | memory vs full | teacher agreement | KL |
|---|---:|---:|---:|
| full fine-tune | 1x | 0.9975 | 0.000 |
| LoRA r=4, bf16 base | 5.33x lower | 0.8355 | 0.063 |
| LoRA r=4, 4-bit base | 10.45x lower | 0.7995 | 0.098 |
| LoRA r=8, 4-bit base | — | 0.797 | 0.097 |

Rank 8 does no better than rank 4, so the gap is the full-rank part of the
update, not a shortage of rank.

**Plain LoRA tops out at 8x.** Frozen bf16 weights still cost 2 of full
fine-tuning's 16 bytes per parameter. For the Llama 3 8B architecture, LoRA
at r=16 trains 41.94M parameters, and memory goes from 128.5 GB to 16.7 GB,
7.68x lower. A 4-bit base brings it to 4.9 GB, 26.0x lower. The lesson's
pipeline has nowhere to put the choice: `cfg["lora"] = True` leaves every
hash unchanged.
