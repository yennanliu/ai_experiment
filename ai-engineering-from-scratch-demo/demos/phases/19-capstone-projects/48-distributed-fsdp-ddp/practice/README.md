<!-- generated:start -->
# 19-capstone-projects / 48-distributed-fsdp-ddp

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/48-distributed-fsdp-ddp/) · upstream spec
`phases/19-capstone-projects/48-distributed-fsdp-ddp/docs/en.md`

```bash
uv run demo practice run 48-distributed-fsdp-ddp --ex 1
uv run demo explain 48-distributed-fsdp-ddp --ex 1
uv run pytest demos/phases/19-capstone-projects/48-distributed-fsdp-ddp
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run with `--world-size 4` and confirm the param spread stays under 1e-3 across the run. | code | T1 | `ex01_every_rank_stays_bit_identical_at_world_size_4_and_the_broadcast_does_no_work.py` |
| 2 | Replace the manual averaging with `dist.all_reduce(op=dist.ReduceOp.AVG)` and time the differ… | code | T1 | `ex02_reduceop_avg_runs_on_gloo_though_documented_nccl_only_and_one_flat_call_is_4x_faster.py` |
| 3 | Add a post-backward hook to the DDP wrapper so the all-reduce overlaps with the rest of the b… | code | T1 | `ex03_the_hook_saves_20_to_40pct_but_on_the_lessons_model_none_of_it_is_overlap.py` |
| 4 | Implement the FSDP re-shard step: after the forward pass, replace the full tensor with the lo… | code | T1 | `ex04_resharding_cuts_params_to_1_over_n_but_right_after_forward_it_breaks_backward.py` |
| 5 | Switch the backend to `nccl` on a CUDA box. Note which environment variables change and which… | code | T1 | `ex05_master_addr_and_port_stay_the_ifname_var_renames_and_the_lessons_cpu_tensors_fail_nccl.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py` on real gloo process groups:
each file starts itself once per rank as a subprocess on localhost, with a
60 s timeout, and kills any rank still alive at the end. Nothing needs a GPU.
Timings depend on machine load, so they are reported in the check details
and only differences of 2x or more (or a 10% saving seen at 20-40%) are
asserted. External facts come from the PyTorch 2.14 distributed docs
(https://docs.pytorch.org/docs/2.14/distributed.html, read 2026-09-29) and
from the installed torch 2.14.0's own `ReduceOp` docstring.

### 1 — every rank stays bit-identical at world size 4, and the broadcast does no work

**The spread is exactly 0.0, not just under 1e-3.** The lesson checks the
spread once, at the end, on the *sum* of each rank's parameters. This
solution records every rank's full parameter vector before each of the 6
steps and after the last one. All four ranks are bit-identical at all 7
points. The manual all-reduce matches the single-process gradient to 2.1e-8.

**The broadcast is not what keeps them equal.** Every rank seeds the same RNG
right before `make_model`, so the ranks start identical anyway:

| run at world size 4 | max element-wise spread | sum spread |
|---|---:|---:|
| shipped | 0.0 | 0.0 |
| `broadcast_module` disabled | 0.0 | 0.0 |
| a different seed per rank, with broadcast | 0.0 | 0.0 |
| a different seed per rank, no broadcast | 0.457 | 4.23 |

**The reported losses are each rank's local loss.** They are 1.3705,
1.5942, 1.3202 and 1.3473, and `rank_main` makes no collective call on them.
The doc itself warns: "If you average gradients but not the loss the
dashboard lies". The `losses` list is also in queue-arrival order. The
committed `outputs/ddp-demo.json` lists rank 1 before rank 0.

### 2 — ReduceOp.AVG runs on gloo, though it is documented as NCCL-only, and one flat call is 4x faster

**Only one line changes, and the time difference is noise.** With the
lesson's `all_reduce_grads_` replaced by one `ReduceOp.AVG` call per tensor,
the gradients are bit-identical to SUM-then-divide at world sizes 2, 3 and
4. The lesson's own gradient check returns the same norm and diff. The two
versions time within about 30% of each other, in either direction from run
to run. The division alone costs about 0.005 ms, under 1% of a sync, so
removing it cannot save more than that.

**torch's own docstring says this should not work.** It says "AVG is only
available with the NCCL backend", and that AVG "divides values by the world
size before summing". On this gloo build AVG runs, and at world size 3 it
matches sum-then-divide bit for bit. Divide-then-sum differs by 1.2e-7.

**The cost is the number of collectives.** One all-reduce over all 596
gradients, flattened into one buffer, beats the lesson's four per-tensor
calls at every world size, by 3.4x to 5.4x across runs:

| world size | 4 per-tensor calls (ms) | 1 flat call (ms) |
|---:|---:|---:|
| 2 | 0.55-0.86 | 0.13-0.21 |
| 3 | 1.4-2.0 | 0.37-0.43 |
| 4 | 1.8-4.0 | 0.50-1.04 |

That is the bucketing the doc's "Use It" credits to production DDP.

### 3 — the hook saves 20-40% of the step, but on the lesson's model none of it is overlap

**The hook cuts the step by about 20-40% at both sizes, and the gradients
are bit-identical.** `register_post_accumulate_grad_hook` starts an async
all-reduce as each gradient is ready, and the step waits on all of them
before dividing. Two gloo ranks, medians of 15 steps:

| model | no sync | async after backward | `sync_grads()` | hook |
|---|---:|---:|---:|---:|
| lesson, 596 params | 0.1-0.3 ms | ≈ hook | 0.8-1.9 ms | 0.6-1.2 ms |
| scaled, 2.1M params | 3-6 ms | 7-19 ms | 11-24 ms | 8-16 ms |

**On the lesson's model, the saving is not overlap.** The whole forward and
backward takes less time than the hook saves, so there is nothing to hide
behind. Issuing the same async calls after backward saves as much: the gain
is from keeping four collectives in flight at once. Even at 2.1M parameters
the hook beats that control by only 0.1-3 ms.

**The lesson's docstring says the wrapper already does this.** `main.py`
promises a wrapper that "averages gradients in a post-backward hook", but
`MinimalDDP` registers no hook, and the trainer calls `sync_grads()`.

### 4 — re-sharding cuts parameters to 1/N, but right after forward it breaks backward

**Per-rank parameter memory drops from 2,384 bytes to 1,192 / 804 / 596 at
world sizes 2 / 3 / 4**, the same on every rank. The output computed from
the gathered weights is bit-identical to the unsharded model's. At 3 ranks
it is not exactly 1/N, although the doc says "the memory win is exact":
each tensor is padded to a multiple of 3, so a rank holds 201 floats
against 198.7. The gathered buffer is 2,412 bytes.

**Re-sharding right after forward, as the exercise words it, makes backward
fail.** Autograd kept the parameter tensors, and swapping their `.data` for
the shard gives backward the wrong shapes. It raises a shape-mismatch
`RuntimeError` at every world size. Gathering again before backward, as
production FSDP does, gives gradients bit-identical to the unsharded model's.
Those gradients are full size, though. At 4 ranks, parameters plus
gradients drop from 4,768 to 2,980 bytes (37.5%), not to a quarter.

**The lesson's sketch never shards anything.** `fsdp_round_trip_sketch`
says it keeps "per-rank memory at 1/world_size", but every rank still holds
all 2,384 bytes afterwards. It checks the gathered copy with
`torch.allclose`, not the equality the doc calls "bit-equal".

### 5 — MASTER_ADDR and MASTER_PORT stay, the interface variable is renamed, and the lesson's CPU tensors would fail NCCL

**MASTER_ADDR and MASTER_PORT stay the same. GLOO_SOCKET_IFNAME becomes
NCCL_SOCKET_IFNAME. TP_SOCKET_IFNAME is not an NCCL variable. Under torchrun,
RANK, WORLD_SIZE and LOCAL_RANK are set for you, and LOCAL_RANK picks the
GPU.** Measured here, with no CUDA:

- the lesson's `init_process_group` sets exactly GLOO_SOCKET_IFNAME,
  MASTER_ADDR, MASTER_PORT and TP_SOCKET_IFNAME;
- two env:// ranks given only RANK, WORLD_SIZE, MASTER_ADDR and MASTER_PORT
  all-reduce 1 + 2 = 3, so the rendezvous variables are backend-neutral;
- `is_nccl_available()` is False, and `rank_main` with backend "nccl"
  returns "Distributed package doesn't have NCCL built in" at once, with no
  hang.

**The lesson's code would fail on a CUDA box too.** The doc says "the only
changes are `backend="nccl"`, device tensors, and `torchrun`", but there is
no device to change. `make_model`, `rank_main`,
`manual_all_reduce_matches_single_process` and `fsdp_round_trip_sketch`
never mention a device, `cuda` or `.to(`, so every tensor is on the CPU.
The docs' backend table marks every NCCL collective as unsupported on CPU.
`main()` also checks only that gloo is available.
