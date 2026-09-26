<!-- generated:start -->
# 17-infrastructure-and-production / 18-vllm-production-stack-lmcache

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/18-vllm-production-stack-lmcache/) · upstream spec
`phases/17-infrastructure-and-production/18-vllm-production-stack-lmcache/docs/en.md`

```bash
uv run demo practice run 18-vllm-production-stack-lmcache --ex 1
uv run demo explain 18-vllm-production-stack-lmcache --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/18-vllm-production-stack-lmcache
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. At what HBM utilization does LMCache start paying? | code | T0 | `ex01_lmcache_pays_at_no_hbm_utilization_because_a_load_costs_7_5x_the_prefill_it_replaces.py` |
| 2 | A tenant shares a 6K-token system prompt across 200 queries/hour. Compute expected LMCache sa… | code | T0 | `ex02_the_tenant_loses_time_to_lmcache_at_the_lessons_rates_and_saves_only_below_0_4_ms_per_block.py` |
| 3 | The LMCache server is a single point of failure. Design the HA strategy (replicas, fallback t… | code | T0 | `ex03_fallback_to_native_makes_an_outage_cheap_and_at_the_lessons_rates_an_outage_makes_the_cluster_faster.py` |
| 4 | LMCache stores to Ceph on spinning disk. For a 4K-token KV at 70B FP8 (500 MB), what's the re… | code | T0 | `ex04_one_spinning_disk_reads_the_kv_42x_slower_than_the_lessons_re_prefill_and_no_10gbe_ceph_cluster_can_catch_it.py` |
| 5 | Argue whether the vLLM 0.11.0 asynchronous path is "free" — where does the overhead hide? | code | T0 | `ex05_the_async_path_is_free_only_on_an_idle_copy_queue_at_85_percent_store_load_a_hits_load_takes_10x_longer.py` |
<!-- generated:end -->

## Answers

### 1 — LMCache pays at no HBM utilization, because a load costs 7.5x the prefill it replaces

**At none. In this model the break-even is a per-block cost, not a
utilization.** A re-prefill costs 16 / 40 = 0.4 ms per block and an LMCache
load 3.0 ms, 7.5x the work it saves, so every LMCache hit adds time. The
shipped run prints 0.98x or 0.99x. The module has no utilization input, so
this solution adds one: every config gets the same per-engine capacity of
`slots` prefixes, and utilization is the 6-prefix working set over it. `run()`
reproduces `simulate()` exactly for CPU_OFFLOAD and LMCACHE.

| utilization (slots) | 3.0 ms/block | 0.4 | 0.2 | 0 | LMCache hits |
|---|---:|---:|---:|---:|---:|
| 100% (6) | 0.9676 | 1.0 | 1.0026 | 1.0052 | 18 |
| 150% (4) | 0.8742 | 1.0 | 1.0112 | 1.0226 | 70 |
| 300% (2) | 0.7883 | 1.0 | 1.0211 | 1.0431 | 135 |
| 600% (1) | 0.7579 | 1.0 | 1.0252 | 1.0517 | 164 |

The load cost decides the sign in every row. Utilization only sets how many
hits there are.

**The simulator has no HBM.** `hbm_capacity_blocks_per_engine = 900` is
assigned and never read. Only NATIVE_ONLY evicts, at 4 prefixes.
CPU_OFFLOAD and LMCACHE never evict, and CPU_OFFLOAD's printed 1.01-1.02x
comes from that unlimited capacity. It never reads an offloaded block back.

**The baseline depends on PYTHONHASHSEED.** NATIVE_ONLY evicts with
`set.pop()`, whose choice follows the string hash. Over hash seeds 0..9 its
total runs from 373983 to 376133 ms and its avoided re-prefills from 111 to
128, so the printed LMCache speedup changes from run to run.

**Decode is 97.6% of the time.** 366033 of the 374883 ms FIFO baseline is
decode, so a free cache could not beat 1.024x there.

### 2 — the tenant loses time to LMCache at the lesson's rates, and saves only below 0.4 ms per block

**The expected saving is negative.** One 6K prefill costs 150 ms. Loading the
same 375 blocks from LMCache costs 1125 ms, 975 ms more per hit. The hour runs
on 4 engines of 4 prefix slots, with m of the lesson's template requests
between the tenant's queries:

| other requests per tenant query (m) | 0 | 1 | 4 | 16 |
|---|---:|---:|---:|---:|
| tenant LMCache hits / hour | 3 | 25 | 69 | 145 |
| saved at 3.0 ms/block (s/hour) | -2.9 | -24.4 | -67.3 | -141.4 |
| saved at 0.4 | 0 | 0 | 0 | 0 |
| saved at 0.2 | 0.2 | 1.9 | 5.2 | 10.9 |
| native prefill left to save (s/hour) | 0.6 | 3.9 | 10.5 | 21.9 |

**Native caching already takes almost all of the ceiling.** Against no cache
at all, the most any cache can save is 199 x 150 ms = 29.85 s an hour. On a
quiet cluster engine-local prefix caching misses once per engine and leaves
0.6 s, so LMCache can only win back time that other tenants' eviction
creates.

**The lesson's "<1K tokens: transfer > re-prefill" has no threshold in this
model.** Both costs are linear in tokens. A load is 7.5x the prefill at 6K
and 8K tokens, and block rounding makes it 7.56x at 1K and 7.68x at 500.

### 3 — fallback to native makes an outage cheap, and at the lesson's rates an outage makes the cluster faster

**Make the connector fall back to native prefill, then add a write-through
replica.** The design, run with the server down for requests 50-149 (200
requests, 4 engines x 4 slots, 0.2 ms/block so that the cache pays):

| strategy | total ms | failed | LMCache hits |
|---|---:|---:|---:|
| no outage | 370733 | 0 | 70 |
| single server, lookups error | 301335 | **37** | 38 |
| fallback to native prefill | 373208 | 0 | 33 |
| failover to a cold standby | 370958 | 0 | 66 |
| write-through (warm) replica | 370733 | 0 | 70 |

(native only: 374883.) Fallback is what removes the single point of failure:
the outage costs 2475 ms of recompute and no requests. A restarted DRAM
server comes back empty, so a replica, cold (+225 ms) or warm (+0), only buys
back hit rate. `single` looks fast only because it drops 37 requests.

**At the lesson's 3.0 ms/block an outage makes the cluster faster.** No
outage: 428833 ms. With the server down, fallback takes 396658 ms and native
374883. The warm replica keeps the slowest configuration running.

**In the shipped configuration the server only matters during warm-up.** With
unlimited engine caches the last LMCache hit is request 89. An outage from
request 90 on changes nothing. One from 88 fails 5 requests without fallback,
because a failed request never fills the engine's cache and the same prefix
fails again there.

### 4 — one spinning disk reads the KV 42x slower than the lesson's re-prefill, and no 10GbE Ceph cluster can catch it

**About 4.3 s against 0.10 s.** The disk side is a stated model, not a
measurement: RADOS's default 4 MiB objects, 150 MB/s sustained per 7200 rpm
spindle, and 8 ms of seek plus rotation per object. 500 MB is 120 objects,
which one spindle reads in 4.32 s. The lesson's 40 tokens/ms re-prefills
4096 tokens in 102.4 ms. Its own DRAM-tier price for the same KV is 768 ms.

| OSDs | 10GbE | 25GbE | 100GbE |
|---:|---:|---:|---:|
| 1 | 4.315 s | 4.315 | 4.315 |
| 12 | 0.40 | 0.36 | 0.36 |
| 60 | 0.40 | 0.16 | 0.072 |

**Striping cannot close the gap on 10 or 25GbE.** The client link alone
needs 0.40 s and 0.16 s. Matching 102.4 ms takes 100GbE and at least 60
OSDs. Replication does not help, since by default Ceph serves reads from the
primary OSD.

**The disk wins only against a much slower prefill.** 40 tokens/ms on 70B is
5.6 PFLOP/s, 2.83 H100s at 100% of dense FP8 peak (1979 TFLOPS). One H100 at
50% MFU needs 0.58 s, and 8 OSDs on 10GbE read the KV in 0.40 s, under it.

**500 MB is not 4K tokens of a 70B model.** With Llama-3-70B's geometry (80
layers, 8 KV heads, head dim 128) an FP8 KV cache is 160 KiB a token: 671 MB
at 4096 tokens. 500 MB holds 3052 tokens. With a BF16 KV cache behind FP8
weights the 4K figure doubles to 1.34 GB.

### 5 — the async path is free only on an idle copy queue: at 85% store load a hit's load takes 10x longer

**No. Async takes the store off the request path and puts its cost in the
copy queue.** A hit still needs its KV on the GPU before it can decode. The
model uses 500 MB transfers at the 83.4 GB/s DMA rate from the vLLM blog,
6.0 ms each. Loads arrive at 10% of capacity and stores at 0-85%, as seeded
Poisson streams:

| store load | hit load, shared queue (mean / p99 ms) | hit load, own queue | store wait, own queue |
|---:|---:|---:|---:|
| 0% | 6.35 / 11.75 | 6.35 / 11.75 | — |
| 30% | 8.01 / 21.29 | 6.35 / 11.75 | 7.31 / 17.4 |
| 60% | 13.14 / 45.99 | 6.35 / 11.75 | 10.55 / 33.41 |
| 85% | 62.35 / 280.06 | 6.35 / 11.75 | 23.29 / 94.4 |

That is ordinary M/D/1 queueing: at 95% total load the formula gives 63 ms.

**A separate queue moves the cost to HBM residency.** Full-duplex PCIe with
separate copy engines keeps loads at 6.35 ms. A block being offloaded still
cannot be freed until its copy finishes: 23 ms on average at 85%, 62 ms if
the queue is shared. In a preemption-heavy workload that HBM is exactly what
is short.

**The lesson's own offload is pure overhead, and its "async" is a 0.1
factor.** CPU_OFFLOAD charges 0.15 ms/block (10% of 1.5) on the request path
for the 6125 blocks of its 24 misses, 918.75 ms in all, and never reads one
back. The blog reports the real-world version: its custom copy kernel costs
6% throughput at a 0% hit rate, because it competes with the model for GPU
cores.

**The lesson dates and credits the async path wrongly.** It says "vLLM 0.11.0
(January 2026) adds an asynchronous offload path". GitHub dates v0.11.0 to
2025-10-02. January 8, 2026 is the date of the vLLM blog post, which says
0.9.0 "extended the connector API to support asynchronous loading and
storing" and that the offloading feature arrived in 0.11.0. Its headline
gains, up to 4x lower TTFT and 5x throughput, came with the 0.12.0 memory
layout. The lesson's note about speculative-decoding interactions is not in
the blog and was not verified here.
