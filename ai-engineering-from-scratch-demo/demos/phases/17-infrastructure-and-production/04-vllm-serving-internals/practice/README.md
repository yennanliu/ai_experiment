<!-- generated:start -->
# 17-infrastructure-and-production / 04-vllm-serving-internals

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/04-vllm-serving-internals/) · upstream spec
`phases/17-infrastructure-and-production/04-vllm-serving-internals/docs/en.md`

```bash
uv run demo practice run 04-vllm-serving-internals --ex 1
uv run demo explain 04-vllm-serving-internals --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/04-vllm-serving-internals
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Compare `STATIC` to `CONTINUOUS` on a workload with mixed short and long… | code | T0 | `ex01_static_beats_continuous_4_6x_because_the_toy_bills_a_padded_batch_as_one_request.py` |
| 2 | Modify the toy scheduler to add `--max-num-batched-tokens`. What is the right value for an H1… | code | T0 | `ex02_the_budget_buys_the_toy_no_throughput_and_one_h100_has_kv_room_for_13k_tokens.py` |
| 3 | Re-read the vLLM v0.18.0 release notes. Which combinations of flags are mutually exclusive? L… | code | T0 | `ex03_the_release_notes_list_no_exclusions_and_v0_18_marks_chunked_prefill_with_spec_decode_compatible.py` |
| 4 | Compute the KV cache fragmentation waste for a trace of 1,000 requests with mean 1,500 output… | code | T0 | `ex04_contiguous_wastes_81_6_percent_paged_0_5_and_the_toy_reserves_like_neither.py` |
| 5 | Explain in one paragraph why chunked prefill helps P99 ITL but not throughput in isolation. W… | code | T0 | `ex05_chunking_cannot_raise_throughput_on_an_additive_cost_and_on_a_roofline_it_lifts_it_17_percent.py` |
<!-- generated:end -->

## Answers

The sources were fetched on 2026-09-26:

- the vLLM v0.18.0 GitHub release,
- `docs/features/README.md`, `docs/features/speculative_decoding/README.md` and
  `vllm/engine/arg_utils.py` at the `v0.18.0` tag,
- the Llama 3.3 70B `config.json`,
- the Hugging Face file listing of `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic`.

The H100 figures are assumptions: 81,559 MiB of HBM (nvidia-smi's figure),
3.35 TB/s and 1979 dense FP8 TFLOPS (spec-sheet peaks). Every exercise runs
the lesson's own `code/main.py`.

### 1 — static beats continuous 4.6x, because the toy bills a padded batch as one request

On the shipped mixed workload (prompts of 128–8192 tokens), STATIC does
**4055 tok/s** and CONTINUOUS **886**. The lesson says continuous "should
dominate"; in the toy it loses 4.6x.

| mode | end (s) | prefill | decode | overhead + idle | P99 ITL |
|---|---:|---:|---:|---:|---:|
| STATIC | 2.32 | 1.31 | 0.53 | 0.48 | 0.7 ms |
| CONTINUOUS | 10.63 | 5.65 | 4.71 | 0.27 | 219.3 ms |

The 8.31 s gap is 4.34 s of prefill and 4.18 s of decode. Tail latency adds
nothing to it. The cause is that the two modes use different cost models:

- **Static** charges a window's prefill at its *longest* prompt, and decode
  at `FORWARD_LATENCY_PER_TOKEN * len(window) / 16`.
- **Continuous** charges every token in full, so batching earns it nothing.
  It lands at 886 tok/s, against 768 for NAIVE.

Put both modes on one cost model and continuous wins. With prefill free in
both and decode at 1/16 per token in both, continuous does 6767 tok/s against
static's 6382, with a mean TTFT of 0.1 ms against 155.7 ms.

### 2 — the budget buys the toy no throughput, and one H100 has KV room for 13k tokens

The modified loop gives each iteration a token budget. Decodes cost 1 token
each; prefill takes whatever budget is left, as vLLM V1 does. With no budget
it reproduces `simulate_continuous` sample for sample.

**The right value is at most free blocks × 16.** A token of Llama 3.3 70B KV
is 80 × 2 × 8 × 128 × 2 = 327,680 bytes in BF16.

- At `--gpu-memory-utilization 0.9` an H100 has 76.97 GB to give. The FP8
  checkpoint takes 72.67 GB of it, leaving 4.30 GB.
- That is **819 blocks, or 13,104 tokens**, before activations. With an FP8
  KV cache it is 1,639 blocks.
- vLLM v0.18.0's own default is 8192 for the API server and 16,384 for
  `LLM`, on any GPU of at least 70 GiB that is not an A100. It fits under
  that bound, but it is chosen by device memory, not by counting blocks.

In the toy the budget moves only latency:

| budget (tokens) | throughput | P99 ITL |
|---:|---:|---:|
| 128 | 878 tok/s | 10.8 ms |
| 8192 | 886 tok/s | 262.2 ms |

Throughput stays between those two values across the sweep, because the toy's
cost is linear in tokens.

The reference's 512-token "chunked prefill" does not bound a step either.
Every prefilling request gets its own chunk, so one iteration schedules 2694
tokens chunked and 18,563 unchunked.

The lesson's "1.25 GB per sequence in BF16" at 8192 tokens is the FP8 figure;
BF16 is 2.68 GB. Its headline setup, 70B FP8 on one H100 at 128 concurrent,
leaves each sequence 102 tokens of KV, or 204 with FP8 KV.

### 3 — the release notes list no exclusions, and v0.18 marks chunked prefill with spec decode compatible

**The v0.18.0 release notes state no mutually exclusive flags.** The
compatibility matrix that v0.18.0 ships ("mutually exclusive features") lists
27 incompatible pairs.

| feature | incompatible with |
|---|---|
| speculative decoding | LoRA, pooling, encoder-decoder, async output, multi-step, best-of, beam search, prompt embeds |
| chunked prefill | encoder-decoder, multi-step |
| encoder-decoder | also prefix caching, LoRA, async output, prompt embeds |
| pooling | also logprobs, prompt logprobs, async output, best-of, beam search, prompt embeds |
| multi-step | also LoRA, best-of, beam search |
| prompt embeds | also prompt logprobs, multimodal |

Three pairs are partial: pooling with chunked prefill, pooling with prefix
caching, and multimodal with LoRA. The speculative-decoding docs add
pipeline parallelism, "as of `vllm<=0.15.0`". The matrix still carries V0-era
rows such as multi-step.

**The lesson's gotcha is the opposite of what v0.18.0 says.** The matrix
marks chunked prefill × speculative decoding compatible. The docs limit the
draft-model gap to `vllm<=0.10.0`.

There are two more problems:

- `--speculative-model` is not a v0.18.0 flag. Only `--speculative-config`
  is registered.
- The N-gram line the lesson cites reads "compatible with the async
  scheduler". It says nothing about chunked prefill.

The skill file in `outputs/` hard-codes the same wrong rejection.

### 4 — contiguous wastes 81.6%, paged 0.5%, and the toy reserves like neither

The trace is 1,000 seeded draws from N(1500, 600), clipped to [1, 8192]. The
mean is 1511, and 2 draws were clipped.

| allocator | waste at completion | waste averaged over decode |
|---|---:|---:|
| (a) contiguous, 8192 reserved | **81.6%** (2.19 GB BF16 per request) | 89.4% |
| (b) PagedAttention, 16-token blocks | **0.50%** (7.6 tokens of tail) | 0.86% |
| the toy's reservation | 0.5% | 50.2% |

The toy's row is the reference's own allocator. `simulate_continuous`
charges `blocks_needed()`, the whole prompt + output rounded up to blocks, at
admission. Its pages are 16 tokens, but it holds the final length from the
first step, using an output length a real scheduler cannot know. Growing one
block at a time, which is what PagedAttention does, is not modelled.

### 5 — chunking cannot raise throughput on an additive cost, and on a roofline it lifts it 17%

Chunked prefill changes when work happens, not how much work there is.
Every prefill token is still computed, and each extra slice pays one more
fixed step cost. So on a cost that is the sum of its tokens, it cannot raise
throughput. What it changes is the longest step. A decode token waits behind
one 512-token slice instead of a whole 8192-token prompt, and that is exactly
P99 ITL. The toy shows it: **886 → 885 tok/s**, while P99 ITL falls from
**219.3 ms to 66.1 ms**.

In practice the win comes from the step cost not being additive. A decode
step is memory-bound: for 70B FP8 on an H100 it reads 72.67 GB of weights in
21.7 ms, whatever the batch. Prefill tokens added to that step use compute
that would otherwise sit idle, up to the ridge point (304 tokens here).

| cost model | prefill-first | whole-prompt mix | budget 320 |
|---|---:|---:|---:|
| additive (toy) | 886 tok/s | 886 | 883 |
| roofline | 248 tok/s, 241.2 ms P99 | 250 | **291 tok/s, 23.0 ms** |

The best budget sits at the ridge point. On the roofline, budgets of 128,
256, 320, 512 and 2048 give 246, 286, 291, 273 and 254 tok/s. That is a
property of the hardware, not of free KV blocks.
