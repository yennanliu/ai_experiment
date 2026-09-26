<!-- generated:start -->
# 17-infrastructure-and-production / 06-sglang-radixattention

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/06-sglang-radixattention/) · upstream spec
`phases/17-infrastructure-and-production/06-sglang-radixattention/docs/en.md`

```bash
uv run demo practice run 06-sglang-radixattention --ex 1
uv run demo explain 06-sglang-radixattention --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/06-sglang-radixattention
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Compare FCFS and cache-aware on the same workload. Where does the delta c… | code | T0 | `ex01_cache_aware_serves_the_rag_queue_in_fcfs_order_so_there_is_no_delta.py` |
| 2 | Modify the workload so prompts randomly permute `[system, tools, context]`. Re-run. What happ… | code | T0 | `ex02_permuting_the_prefix_drops_hit_rate_to_8_percent_and_most_of_the_drop_is_the_budget.py` |
| 3 | Compute the HBM cost of keeping a 2,000-token system prompt resident as one radix branch on L… | code | T0 | `ex03_the_system_prompt_costs_250_mib_once_and_16_copies_cost_3_9_gib.py` |
| 4 | Read the SGLang RadixAttention paper. Explain in three sentences why tree-shaped LRU eviction… | code | T0 | `ex04_block_lru_strands_a_branch_by_evicting_its_first_block_and_leaf_first_order_is_the_fix.py` |
| 5 | A customer reports only 8% cache hit rate. Name three likely causes and the diagnostic you wo… | code | T0 | `ex05_the_8_percent_trace_fails_two_diagnostics_and_the_budget_is_the_bigger_one.py` |
<!-- generated:end -->

## Answers

### 1 — cache-aware serves the RAG queue in FCFS order, so there is no delta

**On the workload the lesson compares, there is no delta.** Both schedulers
score 69.1%, because cache-aware serves the 80 requests in exactly FCFS order.
Its score is max over prefixes of (requests sharing it) x (prefix tokens). The
SYSTEM+TOOLS prefix every request shares wins for all of them at 80 x 2300 =
184000, so every score ties, and a stable sort of ties is the arrival order.
The script's closing "hit rate clears 80% on RAG" is not what it prints.

**Where a delta exists, it is all prefill.** On the scrambled workload
cache-aware lifts hit rate from 8.4% to 26.5%, which is 41400 more prompt
tokens served from cache.

- **Decode savings are zero by construction.** A `Request` has only `rid` and
  `segments`, with no output tokens, and a reused prefix changes no decode
  step.
- **Queue delay is a consequence, not a source.** On a serial-server model
  where each request costs its uncached tokens, mean completion time falls
  from 104525 to 70181 token-units, because there is less prefill to wait
  behind. The reorder also pushes one request back 71 places.

**The budget, not the scheduler, caps the RAG hit rate.** One RAG path is
125 + 19 + 32 + 4 = 180 blocks, and the budget is 160, so TOOLS and a document
never fit beside SYSTEM. Every request reuses SYSTEM and nothing else.

| budget | FCFS | cache-aware | grouped by document |
|---:|---:|---:|---:|
| 160 | 69.1% | 69.1% | 69.1% |
| 180 | 82.5% | 82.5% | 96.0% |
| 250 | 89.3% | 89.3% | 96.0% |
| 400 | 96.0% | 96.0% | 96.0% |

A real depth-first order, grouping requests by document, gets the whole
ceiling from 180 blocks. The lesson's cache-aware scheduler never differs from
FCFS here. SGLang's paper (arXiv:2312.07104) sorts by *matched prefix length*
against the live cache. Its depth-first optimality theorem also assumes a
cache at least as large as the longest request, which the toy's budget is not.

### 2 — permuting the prefix drops hit rate to 8%, and most of the drop is the budget

**Hit rate falls from 69.1% to 8.4%.** The module already ships this change as
`workload_scrambled`, but that function also redraws each request's document.
A controlled version, which only shuffles `workload_rag`'s first three
segments, lands in the same place: 5.5% to 15.1% over five seeds. The radix
key is the ordered path, so each of the 3! orders is its own branch. SYSTEM is
first in only 25 of 80 requests, and the distinct 3-segment heads go from 4
to 24.

**Why: mostly the budget.**

| | fixed order | scrambled |
|---|---:|---:|
| 160 blocks | 69.1% | 8.4% |
| unbounded | 96.0% | 79.5% |

With memory free, ordering costs 16.6 points. At 160 blocks the loss is
60.6 points, because 24 branches compete for a cache that cannot hold even one
180-block request. Fixing the order is still the lever: it also shrinks the
working set the budget has to hold.

The toy shows no 6.4x, though Use It promises "the 6.4x collapse". Hit rate
falls 8.2x, and prefill tokens that must be computed rise 2.96x
(70800 → 209500). The lesson's 6.4x is a throughput figure, and this toy
measures no throughput.

### 3 — the system prompt costs 250 MiB once, and 16 copies cost 3.9 GiB

Llama 3.1 8B has 32 layers and 8 KV heads (grouped-query attention over 32
query heads), with head_dim 128, in bf16. That makes
2 x 32 x 8 x 128 x 2 bytes = **128 KiB of KV per token**.

| | tokens held | HBM | share of 80 GB |
|---|---:|---:|---:|
| one radix branch | 2,000 | 250 MiB (262.1 MB) | 0.33% |
| 16 sequences, no reuse | 32,000 | 3.91 GiB (4.19 GB) | 5.2% |

Sharing the prefix saves 15 copies, which is 3.66 GiB. The 16.06 GB of bf16
weights are the same either way. Each sequence's own suffix costs the same
with or without sharing, so it drops out of the comparison. Grouped-query
attention already cuts the cost 4x: computed from 32 KV heads instead of 8,
the prefix would be 1000 MiB and the batch 15.6 GiB.

In the toy's units, a 2,000-token prompt is 125 blocks, which is what
`code/main.py` computes. The concept diagram says 124, and it gives a
500-token document 31 blocks where the code gives 32. Each block is 2 MiB, so
`KV_BUDGET_BLOCKS = 160` is 320 MiB: 2,560 tokens, less than one 2,860-token
RAG request.

### 4 — block LRU strands a branch by evicting its first block, and leaf-first order is the fix

Three sentences:

1. A cached prefix is usable only from the root down, so a KV block is worth
   keeping only while every block before it is resident. SGLang's tree LRU
   "evicts the least recently used leaf first", which keeps shared ancestors
   reusable "until those ancestors become leaves" (arXiv:2312.07104v2).
2. Block-shaped LRU picks the oldest block anywhere. When a cold branch's
   blocks tie, it can take the branch's first block, which loses the whole
   branch for the price of one block and leaves the rest resident but
   unreachable.
3. Under prefix-heavy load many requests hang off a few shared branches, so
   that stranding shows up directly as lost hits.

Measured on the lesson's RAG workload at 16-token blocks:

| budget | tree LRU | block LRU (front-first ties) | block LRU (suffix-first ties) |
|---:|---:|---:|---:|
| 160 | 76.8% | 0.0%, all 160 blocks stranded | 80.5% |
| 200 | 85.3% | 82.1%, 20 stranded | 85.3% |
| 250 | 91.9% | 87.4%, 34 stranded | 91.9% |
| 300 | 95.1% | 94.1%, 32 stranded | 95.1% |

The last column shows the real lever is eviction *order*. A flat block pool
that evicts the ends of paths first matches the tree. The tree makes that
order structural rather than a tie-break someone has to get right. The
lesson's own `RadixCache` is not strictly tree-shaped. It can evict a parent
and then insert the child, so it holds up to 4 unreachable blocks. It also
evicts whole segments, which is why it gets 69.1% at 160 blocks where
block-level tree LRU gets 76.8%. The paper quotes were checked against the
arXiv HTML of v2.

### 5 — the 8% trace fails two diagnostics, and the budget is the bigger one

Each cause is built as a workload over the lesson's own cache, and every
diagnostic is run on every workload:

| cause (workload) | hit | orderings per segment set | distinct first segments | replay ceiling | tokens missed after eviction |
|---|---:|---:|---:|---:|---:|
| inconsistent ordering (`workload_scrambled`) | 8.4% | **6** | 6 | 79.5% | **71.0%** |
| dynamic content first (per-request timestamp) | 0.0% | 1 | **80 of 80** | **0.0%** | 0.0% |
| budget too small (156 blocks) | 0.0% | 1 | 1 | 96.0% | **96.0%** |
| healthy (`workload_rag`, 160 blocks) | 69.1% | 1 | 1 | 96.0% | 27.0% |

The three diagnostics, all run on logged prompts:

1. **Orderings per segment set.** Group requests by the set of components and
   count the distinct orders they arrive in. More than 1 means the template is
   not fixed. Fix: one canonical order, which gives 69.1%.
2. **First-segment cardinality.** Count distinct first segments (or distinct
   first-block hashes) against the number of requests. When it approaches the
   request count, something unique leads the prompt. A replay ceiling of 0%
   confirms that no amount of memory would help. Fix: move the dynamic field
   to just before the question, which gives 66.7%.
3. **Unbounded replay.** Rerun the trace with no budget, and count the missed
   tokens whose exact path had been cached and was then evicted. A high
   ceiling with a low actual rate means capacity. Fix: 157 blocks gives 69.1%.

**The lesson's own 8% trace fails two diagnostics, and the budget is the
bigger failure.** 71.0% of its tokens are misses after eviction. Ordering
explains only the gap between the 96.0% and 79.5% ceilings. Run all three
diagnostics before settling on the first cause that fits. In this toy the
budget cause is also a cliff: 156 blocks gives 0.0%, and 157, which is SYSTEM
plus one document (125 + 32), gives 69.1%.
