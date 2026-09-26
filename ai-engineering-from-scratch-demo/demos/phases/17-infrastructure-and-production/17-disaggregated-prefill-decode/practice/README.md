<!-- generated:start -->
# 17-infrastructure-and-production / 17-disaggregated-prefill-decode

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/17-disaggregated-prefill-decode/) · upstream spec
`phases/17-infrastructure-and-production/17-disaggregated-prefill-decode/docs/en.md`

```bash
uv run demo practice run 17-disaggregated-prefill-decode --ex 1
uv run demo explain 17-disaggregated-prefill-decode --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/17-disaggregated-prefill-decode
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. At what prompt length does disaggregation beat colocation? | code | T0 | `ex01_there_is_no_prompt_length_crossover_disaggregation_wins_every_row_because_the_decode_pool_is_given_faster_hardware.py` |
| 2 | Design the prefill pool and decode pool for a RAG service with P99 prefix length 8K, output 300. | code | T0 | `ex02_three_prefill_gpus_to_two_decode_gpus_because_kv_memory_caps_the_decode_batch_and_the_codes_batch_of_one_says_one_to_eight.py` |
| 3 | Dynamo vs llm-d: pick one for a pure-Kubernetes shop with no Python runtime preference. | explain | T0 | prose, below |
| 4 | Compute KV transfer cost: 4K prefill on 70B FP8 = ~500 MB KV. At RDMA 100 GB/s, transfer = 5… | code | T0 | `ex04_tcp_matters_only_for_ttft_where_it_adds_50_percent_and_streaming_kv_per_layer_hides_both_behind_prefill.py` |
| 5 | MoE expert routing changes KV access patterns. How does disaggregation behave with MoE that a… | code | T0 | `ex05_routing_leaves_the_kv_transfer_alone_and_makes_decode_read_8x_more_weight_per_token_than_dense_at_batch_64.py` |
<!-- generated:end -->

## Answers

Every code exercise runs the lesson's own `code/main.py` simulator. Exercises
2 and 4 add Llama 3.3 70B's KV size (80 layers x 8 KV heads x 128 x 2 bytes
at FP8). Exercise 5 uses DeepSeek-V3's `config.json`, fetched 2026-09-26.
Exercise 3 draws on the llm-d README and 0.5 release post, and on the Dynamo
README and Grove docs, all fetched 2026-09-26.

### 1 — there is no prompt-length crossover: disaggregation wins every row because the decode pool is given faster hardware

**Disaggregation wins at every prompt length.** It wins all 8 rows of the
table, 256 to 32768 tokens, over RDMA *and* TCP. The winner flips on the
prompt/output ratio instead. Each output token saves 1/0.10 − 1/0.18 =
4.444 ms. Each prompt token costs 0.00125 ms of RDMA transfer, or 0.0125 ms
over TCP.

| crossover | RDMA | TCP |
|---|---:|---:|
| colocation wins when prompt > | 3555 × output | 355 × output |
| first losing prompt at 1 output token | 3556 | 356 |

With 0 output tokens colocation wins at any length.

**The whole win is the decode pool's constant.** `DECODE_TOK_PER_MS_DECODE_GPU`
is 0.18 against 0.10 colocated, an "H200-like" GPU. Put the decode pool on
the same GPU and colocation wins every row by exactly the transfer: 0.32 ms at
256 tokens, 40.96 ms at 32768. The simulator has no interference, batching
or GPU count. Splitting onto identical GPUs can therefore only add the KV
transfer.

**The lesson's 512/200 threshold is not in the code.** At (256, 100)
disaggregation wins by 444.1 ms against a 0.32 ms RDMA and 3.2 ms TCP tax. The
skill's "TCP raises the break-even to prompts >2K" is not there either: TCP
wins all 8 rows, and the Winner column compares RDMA only.

**The constants exceed the lesson's own hardware numbers.**

- **Prefill:** 40 tokens/ms at 2 × 70e9 FLOP/token is 5600 TFLOPS. The text
  says "~2000 TFLOPS".
- **Decode:** 100 tok/s at batch 1 needs 7 TB/s of weight reads. The text
  says "~3 TB/s", which bounds decode at 42.9 tok/s.

"Use It" also promises throughput and cost per request. The script prints
latency only.

### 2 — three prefill GPUs to two decode GPUs, because KV memory caps the decode batch; the code's batch of one says one to eight

The exercise gives no request rate, so this design assumes **10 req/s**.
Every request is sized at P99: 8192 prefix tokens, 300 output.

- **Prefill pool: 3 GPUs.** Each prefill is 204.8 GPU-ms at the lesson's
  rate, so 10 req/s keeps 2.05 GPUs busy. Three keeps utilization at or
  below 70%, leaving P99 headroom.
- **Decode pool: 2 H200s.** Beside 70 GB of FP8 weights, a 141 GB H200 holds
  66 requests' KV: 8492 tokens × 125,000 B = 1.06 GB each. A full-batch
  memory-bound step at 4.8 TB/s takes 28.9 ms. At 10 req/s, 86.8 requests
  are in flight, which needs 2 GPUs.
- **SLA:** TTFT is 204.8 ms of prefill plus 10.24 ms of RDMA transfer,
  215 ms before queueing. TPOT is 28.9 ms.

**The code's batch-of-one decode sizes the pools the other way round.** At
0.18 tok/ms a request needs 1666.7 decode GPU-ms against 204.8 prefill ms,
8.14 decode GPUs per prefill GPU. Batched to the KV cap, a request costs
131.5 decode GPU-ms. Prefill then needs 1.56× the decode pool's GPU time, in
line with the skill's "2 prefill : 1 decode for RAG-heavy".

**The decode batch is set by KV memory, and the lesson's KV is 76% of the
real one.** Llama 3.3 70B at FP8 stores 163,840 B/token, against the code's
125,000. At the real size a decode GPU holds 51 requests. The ratio falls to
1.2, and 2 decode GPUs still suffice at this rate, with less headroom.
Batching also costs latency that the code cannot show: TPOT is 28.9 ms, not
5.56 ms.

### 3 — llm-d, because it is built from the Kubernetes objects the shop already runs; "no Python preference" decides nothing

*Draws on "Dynamo vs llm-d".*

**Pick llm-d.** Its README describes it as "a high-performance distributed
inference serving stack optimized for production deployments on
Kubernetes". The lesson's contrast holds in outline. llm-d puts prefill,
decode and the router in as ordinary Kubernetes workloads that the shop
already knows how to scale, roll and observe. Dynamo is an "orchestration
layer above inference engines". On Kubernetes it runs through its own
operator and a `DynamoGraphDeployment` custom resource. Its docs also say the
platform install can add Grove and the KAI Scheduler, which is optional. A
pure-Kubernetes shop can run either one. Only llm-d's control plane is made
of objects it already operates.

**The Python condition does not separate them.** Dynamo is "Built in Rust for
performance, Python for extensibility". llm-d's model servers are vLLM and
SGLang, which are Python engines. Both stacks put a Python runtime on the GPU
nodes, so a shop with no Python preference loses nothing either way.

**Where Dynamo would still win.** Its SLA Planner rate-matches prefill to
decode for you, and it is the one of the two that backs TensorRT-LLM. If the
shop wants automatic pool ratios, or TRT-LLM kernels, the choice flips.

Three of the lesson's facts about llm-d do not match the sources:

- **The founders do not include AWS.** The lesson says "Red Hat + AWS". The
  llm-d README lists Red Hat, Google Cloud, IBM Research, CoreWeave and
  NVIDIA as founders, and AWS appears nowhere on it.
- **`packDomain: rack` belongs to Dynamo.** The lesson lists it under llm-d.
  It is Grove's topology constraint, exposed on Dynamo's
  `DynamoGraphDeployment` as `topologyConstraint`.
- **UCCL is not a scale-to-zero layer.** The Key Terms call it that. The
  llm-d 0.5 post (February 4, 2026) says UCCL "provides a unified
  abstraction over vendor-specific collective primitives", merged into
  NIXL 0.9. Scale-to-zero instead "uses a specialized activator component".
  That also means llm-d moves KV over NIXL, the transport the lesson credits
  to Dynamo.

I did not verify the lesson's claim that llm-d scales prefill on queue depth
and decode on KV utilization.

### 4 — TCP matters only for TTFT, where it adds 50%, and streaming KV per layer hides both behind prefill

**TCP matters, and only for TTFT.** The code's 4K KV is 512 MB. Here is where
the transfer lands in each SLA metric:

| | RDMA | TCP |
|---|---:|---:|
| transfer | 5.12 ms | 51.2 ms |
| TTFT (prefill 102.4 ms) | 107.52 ms (+5%) | 153.6 ms (+50%) |
| TPOT | unchanged | unchanged |
| end to end at 500 output tokens | 2885.3 ms | 2931.4 ms (+1.6%) |

The transfer happens once, between prefill and the first decoded token. Any
TTFT SLO between 107.5 and 153.6 ms fails on TCP only. No latency SLA will
notice the 1.6% end to end.

**Streaming KV per layer hides both links.** Suppose each of the 80 layers'
KV is sent as soon as that layer is prefilled. Then only the last layer's
transfer is exposed: 0.064 ms over RDMA and 0.64 ms over TCP. This holds for
any link faster than 125,000 B/token × 40 tokens/ms = 5 GB/s. A 10 GbE link,
at 1.25 GB/s, is slower than that. It takes 409.6 ms and exposes 308.5 ms.

**The lesson's own transfer figures disagree 4-16x.** The same page says NIXL
transfer is "typically 20-80 ms" for this 4K KV, and 5 ms over RDMA. At Llama
3.3 70B's real FP8 KV the 4K prompt is 671 MB, not ~500 MB. That makes the
transfer 6.71 ms over RDMA and 67.1 ms over TCP.

### 5 — routing leaves the KV transfer alone and makes decode read 8x more weight per token than dense at batch 64

The model is DeepSeek-V3 at 1 byte per value: 61 layers (3 dense), 256
routed experts, top-8, MLA attention. Routing is a seeded uniform top-8.

**The premise is off: routing does not change KV.** KV is attention state,
and every token runs attention whatever experts it picks. The experts are
FFN weights. So the handoff is set by the attention design. MLA stores
61 × 576 = 35,136 B/token, 28% of the lesson's 70B constant. Through the
reference `ms_disaggregated`, a 4K prompt moves 143.9 MB in 1.44 ms over
RDMA.

**What routing changes is weight reads, and the two pools see it
differently:**

| forward pass | experts touched per layer | GB read per step | GB per token (dense 70B) |
|---|---:|---:|---:|
| prefill, 4096 tokens | 256 | — | — |
| decode, batch 1 | 8 | 37.0 | 37.0 (70) |
| decode, batch 8 | 57.1 | 162.3 | 20.29 (8.75) |
| decode, batch 64 | 223.7 | 587.9 | 9.19 (1.09) |
| decode, batch 256 | 256.0 | 670.4 | 2.62 (0.27) |

Prefill touches every expert, 128 tokens each, so the prefill pool behaves
like a dense 37B-active model and stays compute-bound. Decode's expert set
grows with the batch until one step reads the whole model. At batch 64 each
expert serves 2.3 tokens, and a token costs **8.4x** the weight reads of
dense 70B.

The decode pool therefore amortizes only with very large batches spread over
many GPUs, which is wide expert parallelism. 671 GB of FP8 weights is 4.8
H200s before any KV. The two phases want opposite layouts, and
disaggregation is what lets each pool choose its own.

The lesson's simulator cannot express any of this. It has no expert, batch
or MoE term, so MoE reaches it only through the KV-bytes constant, and
routing does not change that constant.
