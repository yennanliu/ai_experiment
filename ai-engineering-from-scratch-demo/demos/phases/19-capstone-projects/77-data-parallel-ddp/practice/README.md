<!-- generated:start -->
# 19-capstone-projects / 77-data-parallel-ddp

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/77-data-parallel-ddp/) · upstream spec
`phases/19-capstone-projects/77-data-parallel-ddp/docs/en.md`

```bash
uv run demo practice run 77-data-parallel-ddp --ex 1
uv run demo explain 77-data-parallel-ddp --ex 1
uv run pytest demos/phases/19-capstone-projects/77-data-parallel-ddp
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add gradient buckets of configurable size and measure the speedup vs one-allreduce-per-parame… | code | T1 | `ex01_one_25mb_bucket_syncs_64_small_grads_35x_faster_but_1mb_weights_gain_only_1_5x.py` |
| 2 | Implement `no_sync()` as a context manager and verify gradient accumulation matches a single-… | code | T1 | `ex02_no_sync_matches_the_baseline_to_1_5e_8_and_forgetting_it_costs_4x_the_calls_not_the_answer.py` |
| 3 | Add a `find_unused_parameters` mode where the forward sometimes skips one of the MLP layers;… | code | T1 | `ex03_without_the_flag_one_rank_aborts_and_three_block_and_pytorch_raises_instead_of_deadlocking.py` |
| 4 | Replace gloo with `torch.distributed.barrier()`-only synchronisation to feel the difference b… | code | T1 | `ex04_barrier_only_ranks_drift_0_23_apart_while_rank_0s_loss_stays_within_3_6pct_of_true_ddp.py` |
| 5 | Measure the gradient-sync overhead as a fraction of step time for batch sizes 1, 16, 256 and… | code | T1 | `ex05_sync_is_93pct_of_the_step_at_batch_1_16_and_256_and_compute_catches_up_only_past_4096_rows.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py` (`MiniMLP`,
`DistributedDataParallel`, `make_dataset`, `reference_single_process`) on
real gloo process groups. Each file starts itself once per rank as a
subprocess on localhost, with a timeout, and kills any rank still alive at the
end. Nothing needs a GPU. Timings depend on machine load. They go in the
check details, and only gaps with a 2x or wider margin are asserted. The
PyTorch facts come from the installed torch 2.14.0 itself:
`_DEFAULT_BUCKET_CAP_MB` and `default_pg_timeout`.

### 1 — one 25 MB bucket syncs 64 small gradients 35x faster, but on 1 MB weights buckets gain only 1.5x

**On a 32-layer `Linear(64, 64)` model, one bucket is about 35x faster than one
all-reduce per parameter.** The run uses 4 ranks and times the sync alone:

| cap | all-reduce calls | median ms | speedup |
|---|---:|---:|---:|
| per parameter (lesson) | 64 | ~25 | 1x |
| 64 KB | 11 | ~4.5 | 5.5x |
| 256 KB | 3 | ~1.4 | 17x |
| 1 MB / 25 MB | 1 | ~0.7 | 35x |

The gradients are bit-identical in every case. The cost is the number of
calls, about 0.39 ms each, not the 532 KB of data.

**Buckets pay only when tensors are small.** With `Linear(512, 512)` layers
every weight is 1 MB. Caps up to 1 MB still make 64 calls, and the extra
copies make them slightly slower than the lesson's sync. 25 MB (2 calls)
gains only 1.5-2x (1.5x on an idle machine).

**The lesson does not bucket.** The doc says the tiny model goes "into one
bucket", but `sync_grads` makes 6 calls per step on `MiniMLP`. The doc's ~25 MB
figure matches torch's default.

### 2 — no_sync matches the baseline to 1.5e-8, and forgetting it costs 4x the calls, not a wrong answer

**`no_sync` accumulation over K = 4 matches a single process that walks all
16 microbatches.** After 10 steps the largest parameter gap is 1.5e-8, while
the parameters moved by 0.084. All ranks are identical, and PyTorch's own
`no_sync()` gives the same 1.5e-8.

**Forgetting `no_sync` is wasteful but not wrong.** It makes 240 all-reduces
instead of 60 per run. The resulting model is the same to 1.5e-8, because
averaging an already averaged gradient changes nothing.

**The promised "byte-equal parameter equivalence" does not hold.** Even in the
lesson's own setting (K = 1) the gap is 1.5e-8. The lesson's test compares to
4-5 decimal places, so it cannot see this.

### 3 — without the flag one rank aborts and three block, and PyTorch's DDP raises instead of deadlocking

**Giving each unused parameter a zero gradient before `sync_grads` fixes
it.** The forward skips the middle layer on one rank per step. With the zero
fill, all 6 steps run and the parameters match the single-process baseline to
3.0e-8. PyTorch's `find_unused_parameters=True` matches it to 1.5e-8.

**Without the flag you do not get a clean deadlock:**

| wrapper, no flag | rank 0 (skipped at step 0) | ranks 1-3 |
|---|---|---|
| lesson `sync_grads` | SIGABRT (exit -6): gloo size mismatch, 4 calls vs 6 | 0 steps; hit the 5 s timeout or hang past it |
| PyTorch DDP | finishes step 0, raises "Expected to have finished reduction" at step 1 | 0 steps; block until the timeout |

With torch's default timeout of 1,800 s, the blocked ranks stall for 30
minutes. The doc says "the allreduce deadlocks". Real DDP fails fast on the
rank that skipped.

### 4 — barrier-only ranks drift 0.23 apart while rank 0's loss stays within 3.6% of true DDP

**A barrier syncs time, not values.** Gloo stays: `barrier()` itself runs on
it. The table shows the largest per-element gap between ranks after each step:

| step | all-reduce | barrier only |
|---:|---:|---:|
| 1 | 0.0 | 0.0538 |
| 5 | 0.0 | 0.1714 |
| 20 | 0.0 | 0.2307 |

A barrier costs about 0.12 ms against about 2.2-3.6 ms for the 6 all-reduces.

**The loss curve does not warn you.** Rank 0's barrier-only loss stays within
3.53% of true DDP for all 20 steps, even though every rank is now its own
model trained on a quarter of the data.

**The doc's seed warning does not apply to this code.** When each rank seeds
`SEED + rank` before building the model, the constructor's broadcast overwrites
the difference. The spread stays 0.0 and the loss gap 0.0. The doc says "the
test for parameter equivalence fails on step 1".

### 5 — sync is 93% of the step at batch 1, 16 and 256, and compute catches up only past 4,096 rows

**The sync is about 93% of the step at all three batch sizes, and the share
barely moves:**

| rows per rank | compute ms | sync ms | sync / step |
|---:|---:|---:|---:|
| 1 | ~0.15 | ~2.4 | 0.93 |
| 16 | ~0.12 | ~2.3 | 0.94 |
| 256 | ~0.14 | ~2.2 | 0.93 |
| 4,096 | ~0.7 | ~2.2 | 0.75 |
| 32,768 | ~9 | ~2.4 | 0.21 |

Sync depends on the gradient, not the batch: 6 tensors and 6,928 bytes at
every size, one all-reduce per tensor. So sync is 6 times gloo's call latency.
Compute is flat up to 256 rows because at that size the time is framework
overhead, not arithmetic.

**The doc's aim of making sync "nearly free relative to compute" is far off on
the lesson's own model.** Compute overtakes sync only somewhere between 4,096
and 32,768 rows. The same bytes in one flat all-reduce take about 0.4 ms
instead of 2.3 ms, which is exercise 1's point again.
