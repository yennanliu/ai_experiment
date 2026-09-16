<!-- generated:start -->
# 10-llms-from-scratch / 12-inference-optimization

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/12-inference-optimization/) · upstream spec
`phases/10-llms-from-scratch/12-inference-optimization/docs/en.md`

```bash
uv run demo practice run 12-inference-optimization --ex 1
uv run demo explain 12-inference-optimization --ex 1
uv run pytest demos/phases/10-llms-from-scratch/12-inference-optimization
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Modify the KV cache profiler to compare FP16 vs FP8 vs INT4 KV cache quantization. For Llama… | code | T0 | `ex01_exactly_four_times_and_the_weights_never_move.py` |
| 2 | Extend the continuous batching simulator to track GPU utilization (fraction of batch slots fi… | code | T0 | `ex02_the_tail_the_exercise_names_breaks_its_own_claim.py` |
| 3 | Implement a grouped-query attention (GQA) version of the KV cache where `num_kv_heads < num_q… | code | T0 | `ex03_the_calculator_already_does_gqa_the_class_does_not.py` |
| 4 | Build a prefix cache that uses LRU eviction. Set max_entries to 500 and generate 1,000 reques… | code | T0 | `ex04_the_field_it_never_reads_is_the_one_that_works.py` |
| 5 | Extend the speculative decoding simulator to implement tree-based speculation (EAGLE-2 style)… | code | T0 | `ex05_every_tree_is_slower_than_the_chain.py` |
<!-- generated:end -->

## Answers

`code/main.py` is a serving-side toolkit: a KV cache and a memory profiler, two
batching simulators, a prefix trie, and a speculative-decoding simulator. Four of
the five exercises ask for a number the toolkit's own arithmetic already
determines, and the fifth asks for a feature the toolkit already has.

All five are **T0** on the `math` group (`uv sync --extra math`).

### 1 — exactly four times, and the weights never move

**ANSWER: 126 → 252 → 504 users, exactly 2.000× and 4.000×.** Not "roughly".
`max_users_at_4k` is `available_for_kv / per_token / 4096`, and
`available_for_kv` is **157.6 GB in all three arms** — the KV dtype appears only
in the denominator, so the prediction restates the formula.

**FINDING: the exercise quantises the smaller of the two terms.**

```text
320 GB card  =  130.4 GB fp16 weights  +  32.0 GB flat overhead  +  157.6 GB KV
                ^^^^^^^^^^^^^^^^^^^^^ 41% of the card, never touched
```

Quantising the *model* to INT4 instead takes 126 users to **204** (1.62×), and
the two terms are independent.

**FINDING: the 10% overhead scales with the card, not with the work.**
`gpu_memory_gb * 0.1` reserves 32 GB at 4×A100 for activations, fragmentation
and the CUDA context — none of which grow with the card.

**MECHANISM: 320 KB per token at fp16** (`2 × 80 layers × 8 KV heads × 128
dims`) is 1.25 GB per user at 4K context, against 0.31 GB at INT4.

### 2 — the tail the exercise names breaks its own claim

**ANSWER: continuous batching holds 0.240, not >0.80.** It is **exactly 1.000**
while the queue still has work (262 of 2295 steps) and 0.142 for the remaining
89% of the timeline. The scheduler is full whenever there is anything to fill it
with; the metric is measuring the workload after that.

**FINDING: the distribution the exercise specifies is what breaks its
prediction.**

| output lengths | longest | continuous | static |
|---|---:|---:|---:|
| Pareto(1.5, 20) | **2275** | **0.240** | 0.200 |
| exponential, same mean | 323 | 0.858 | 0.426 |
| constant, same mean | 88 | 0.893 | 0.893 |

A Pareto with shape 1.5 has **infinite variance**, so one request ten times the
next longest is the expected draw, not a bad one.

**FINDING: the two schedulers converge at both ends of the tail** — 1.20×,
2.01×, 1.00×. Where every request is the same length, static batching is already
optimal.

**MECHANISM: refilling a slot cannot shorten the longest request.** 83% of the
timeline runs with exactly 1 of 8 slots filled.

### 3 — the calculator already does GQA; the class does not

**ANSWER: exactly 8.0×, at every context length** — 320 KB/token against 2560.
`2 × layers × kv_heads × head_dim × bytes` is linear in `kv_heads`, so the "8×
reduction" is `64 / 8` written out. `kv_cache_memory` already takes
`num_kv_heads` and the 70B config already sets it to 8; what has to be written
is the MHA *baseline*.

**FINDING: GQA is not an optimisation here, it is the deployment.** 4×A100-80GB
at 4K context seats **126** users with 8 KV heads and **15** with 64 — one MHA
user needs 10.0 GB of cache against 1.25.

**FINDING: the lesson ships two KV-cache models and they disagree.** `KVCache`
takes a single `num_heads` and allocates K and V at that width, so a cache built
for 64 query heads is 8× the size the calculator reports for the same model.

**MECHANISM: there is no query-head field in a model config.** Queries are
consumed in the step that produces them; only K and V persist.

### 4 — the field it never reads is the one that works

| policy | hit rate | nodes |
|---|---:|---:|
| the lesson's cache, 500 entries | 0.484 | 500 (none ever evicted) |
| **LRU**, 500 entries | **0.451** | 500 |
| **LFU on `hit_count`**, 500 entries | **0.601** | 500 |
| the lesson's cache, unlimited | 0.623 | 35,774 |

**ANSWER: LRU — the policy the exercise names — scores below evicting nothing at
all, and misses the 55% bar. LFU clears it**, recovering 96% of the unlimited
cache's hit rate from 1.4% of its nodes. `TrieNode.hit_count` is maintained on
every lookup by the lesson and read by nothing.

**FINDING: the reference has no eviction, and fails silently.** `insert` returns
the index it stopped at and nothing checks it; a second 1,000-token prompt into a
full cache adds **0** nodes. That the frozen cache still scores 0.484 is the
point — freezing the first few prompts is a policy, and it beats recency here.

**MECHANISM: recency is the wrong key for a workload that churns its leaves.**
Every request appends 20 unique tokens. Recency cannot tell a shared prefix from
the tail of the request that just used it; the five shared prefixes are 200 of
the 500 nodes, and keeping exactly those is worth **+0.150**.

**FINDING: "hit rate" counts a single shared token as a hit** (`depth > 0`). At
a 5,000-token vocabulary that nearly agrees with full-prefix reuse; at a small
one it goes to 1.0 with nothing reused.

### 5 — every tree is slower than the chain

| plan | accepted | drafted nodes | cost | speedup |
|---|---:|---:|---:|---:|
| linear K=3 | 1.952 | 3 | 15.0 | 1.97× |
| linear K=5 | 2.689 | 5 | 17.0 | **2.17×** |
| linear K=8 | 3.329 | 8 | 20.0 | 2.16× |
| tree 2×3 | 2.766 | 14 | 26.0 | 1.45× |
| tree 4×3 | 2.990 | 84 | 96.0 | **0.42×** |
| tree 2×5 | 4.431 | 62 | 74.0 | 0.73× |

**ANSWER: the tree accepts more per round and is slower at every setting.** The
exercise's own 2×3 tree buys **+0.077** accepted tokens over a 5-token chain for
**9** more drafted ones, and 4×3 is slower than not speculating at all.

**MECHANISM: acceptance here is a coin, not a comparison.**

```python
target_probs = target_model.get_probs(context, draft_tokens)   # computed
draft_p = draft_model.get_probs(context + ..., token)          # computed
if r < draft_model.acceptance_rate:                            # neither used
```

A branch is one more independent draw: per-level survival goes 0.800 → 0.960 →
0.998 for 1, 2 and 4 branches, at `b` drafts per node.

**FINDING: what makes EAGLE-2's trees work is not in the model.** Real tree
speculation branches where the draft model is *uncertain*, so the candidates sit
where the chain would have lost. Here every draw is independent of the tokens,
and the optimal tree has no branches.

**FINDING: the flat `verify_cost` is 71% of a K=5 round**, which is why the
linear speedup is flat from K=5 to K=8 (2.17× against 2.16×).
