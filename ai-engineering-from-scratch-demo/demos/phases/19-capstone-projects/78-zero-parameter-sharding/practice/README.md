<!-- generated:start -->
# 19-capstone-projects / 78-zero-parameter-sharding

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/78-zero-parameter-sharding/) · upstream spec
`phases/19-capstone-projects/78-zero-parameter-sharding/docs/en.md`

```bash
uv run demo practice run 78-zero-parameter-sharding --ex 1
uv run demo explain 78-zero-parameter-sharding --ex 1
uv run pytest demos/phases/19-capstone-projects/78-zero-parameter-sharding
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Extend to ZeRO-2 by sharding gradients: each rank only stores the gradient for its shard, ach… | code | T1 | `ex01_zeroing_non_shard_grads_before_reduce_scatter_trains_each_shard_on_one_ranks_batch_and_nothing_notices.py` |
| 2 | Add a memory profiler that prints actual fp32 byte usage on rank 0 versus the formula predict… | code | T1 | `ex02_rank_0_holds_19052_bytes_not_the_formulas_12124_because_nothing_in_the_code_is_fp16.py` |
| 3 | Measure the per-step wall-clock time of vanilla DDP versus ZeRO-1 and decompose into forward,… | code | T1 | `ex03_zero1_comm_takes_1_9_to_3_8x_ddps_on_gloo_so_throughput_does_not_hold.py` |
| 4 | Implement gradient clipping under ZeRO-1: the L2 norm must be computed across all shards via… | code | T1 | `ex04_without_the_allreduce_each_shard_clips_by_33_to_77pct_of_the_norm_as_far_off_as_no_clipping.py` |
| 5 | Implement a "naive ZeRO" with allreduce instead of reduce_scatter, measure the wire-time diff… | code | T1 | `ex05_naive_allreduce_zero_is_3x_faster_on_gloo_so_reduce_scatter_wins_on_bytes_not_time.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py` (`ZeroOptimizer`, `MiniMLP`
with P = 1,732 parameters, seed 13, batch 8 per rank, lr 0.05) on real gloo
process groups. Each file starts itself once per rank as a subprocess on
localhost, with a 90 s timeout, and kills any rank still alive at the end.
Variants are switched in by wrapping `ref.dist.reduce_scatter`, so the
lesson's optimiser runs unmodified. Nothing needs a GPU. Timings depend on
machine load: they are reported in the check details, and only wide ratios
are asserted. Installed torch: 2.14.0.

### 1 — zeroing the non-shard gradient before the reduce_scatter trains each shard on one rank's batch, and nothing notices

**ZeRO-2 is sharding the gradient *after* the reduce_scatter.** Freeing the
full gradient as soon as the rank's summed shard arrives ends bit-identical
to ZeRO-1 (max parameter difference 0.0 on every rank).

| run | gradient bytes held after reduce_scatter | max diff vs ZeRO-1 | final loss |
|---|---:|---:|---:|
| ZeRO-1 as shipped | 8,660 | — | 2.48 |
| ZeRO-2 (free after reduce_scatter) | 1,732 | 0.0 | 2.48 |
| recipe read literally (zero before) | 8,660 | 0.907 | 2.786 |

**Zeroing before the reduce_scatter is a bug.** The other ranks have just
zeroed shard r, so what comes back is the rank's own gradient on 20 of 20
steps. Each shard learns from 8 samples instead of 32, and zeroing in place
saves no memory.

**The lesson's own checks miss it.** The broken run still has 4
bit-identical ranks (the all_gather rebuilds the same model everywhere), and
its loss still falls. Those are the two things `tests/test_zero.py` asserts.

### 2 — rank 0 holds 19,052 bytes, not the formula's 12,124, because nothing in the code is fp16

**The profiler reads 19,052 bytes on rank 0. `memory_table` predicts 12,124.**

| term | formula | actual |
|---|---:|---:|
| params | 2P = 3,464 | 4P = 6,928 |
| grads | 2P = 3,464 | 4P = 6,928 |
| master + m + v shards | 12P/4 = 5,196 | 5,196 |

The shards match exactly. The gap is exactly 4P: every tensor is float32.

**The real saving is 31.2%, not the printed 56.2%.** DDP with
`torch.optim.Adam` measures 27,712 bytes. In fp32 the ZeRO master shard is a
duplicate of the parameter slice (bit-identical after the step).

**`step()` allocates 20P = 34,640 bytes of temporaries** to save 5,196
bytes of state. That is more than vanilla DDP's whole per-rank state. At
N = 3 the padding shows too: 6,936 bytes per rank against 12P/3 = 6,928.

### 3 — ZeRO-1's comm takes 1.9-3.8x DDP's on gloo, so "throughput holds" does not hold

**Comm is most of the step, and ZeRO-1's comm is the larger one.** Here is
one run, in ms per step (median of 5 interleaved blocks of 40 steps):

| hidden | run | forward | backward | comm | optimizer |
|---|---|---:|---:|---:|---:|
| 32 | DDP | 0.02 | 0.05 | 0.40 | 0.13 |
| 32 | ZeRO-1 | 0.02 | 0.05 | 1.51 | 0.08 |
| 1,024 | DDP | 1.29 | 0.94 | 4.78 | 5.00 |
| 1,024 | ZeRO-1 | 1.24 | 1.47 | 9.41 | 4.12 |

Both runs reach the same parameters (max difference 4.6e-7).

**"Memory wins, throughput holds" is false here.** Over 8 runs, ZeRO-1's
reduce_scatter plus all_gather took 2.5-3.8x as long as DDP's single
all_reduce on the lesson's model, and 1.9-2.7x at 1.07M parameters. The
whole step was 1.3-2.8x slower. The doc's "same wire traffic" argument
assumes a bandwidth-bound ring. Gloo on loopback is latency-bound.

### 4 — without the allreduce each shard clips by 33-77% of the norm, and lands as far off as no clipping

**Clipping between the reduce_scatter and Adam, with an all_reduce of the
local norm squared, matches torch.** Compared with a single process using
`clip_grad_norm_` on the same global batch, the parameters are within
3.0e-7 and the norms within 4.8e-6. All 20 steps clip (norm 1.54 to 13.98).

**Without the all_reduce, each rank clips by its own shard's norm.** At
step 1 those norms are 0.331, 0.389, 0.378 and 0.772 of the global norm.
The parameters end 0.413 from the target, close to not clipping at all
(0.503).

**A rank-agreement check cannot see the bug.** All three runs keep every
rank bit-identical.

### 5 — naive allreduce ZeRO is 3x faster on gloo, so reduce_scatter wins on bytes, not time

**Naive ZeRO trains the same model (to 2.4e-7), and on this gloo it is
faster.** Its gradient collective is 2.5-3.2x faster per step on the
lesson's model. At 1M floats the two are close (naive 1.02-1.22x faster).

**The case for reduce_scatter is bytes on the wire.** These are ring volumes
per rank per step (ZeRO paper, section 7.1, https://arxiv.org/pdf/1910.02054,
read 2026-09-29):

| gradient | ZeRO-1 | naive | DDP |
|---|---:|---:|---:|
| lesson model (1,732 floats) | 10,392 B | 15,588 B | 10,392 B |
| 1M floats | 6.29 MB | 9.44 MB | 6.29 MB |

The naive version sends 1.5x the bytes. On a bandwidth-bound NCCL ring that
decides it. On latency-bound gloo loopback, the extra bytes cost nothing
measurable.
