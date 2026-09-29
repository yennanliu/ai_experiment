<!-- generated:start -->
# 19-capstone-projects / 46-gradient-accumulation

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/46-gradient-accumulation/) · upstream spec
`phases/19-capstone-projects/46-gradient-accumulation/docs/en.md`

```bash
uv run demo practice run 46-gradient-accumulation --ex 1
uv run demo explain 46-gradient-accumulation --ex 1
uv run pytest demos/phases/19-capstone-projects/46-gradient-accumulation
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Re-run the sweep with `--num-steps 100` and plot samples per second against effective batch.… | code | T1 | `ex01_samples_per_sec_flattens_by_effective_batch_16_to_32_and_every_points_loss_sits_at_ln_16.py` |
| 2 | Add a wrong scaling variant (no division) and show the parameter diff at step 1 against the r… | code | T1 | `ex02_without_the_division_sgds_first_step_is_4x_too_big_but_adamws_is_1_000015x.py` |
| 3 | Swap SGD for AdamW and confirm the optimizer state advances once per effective step, not once… | code | T1 | `ex03_adamws_step_counter_reads_10_after_10_effective_steps_and_per_micro_stepping_moves_2_4x_as_far.py` |
| 4 | Introduce a real `DistributedDataParallel` wrapper and route the `no_sync_context` to its met… | code | T1 | `ex04_the_stock_no_sync_stand_in_reports_1_sync_per_step_while_real_ddp_all_reduces_all_n.py` |
| 5 | Modify the equivalence check to compare two different micro splits (2 by 8 vs 4 by 4) and exp… | code | T1 | `ex05_2x8_and_4x4_agree_to_one_float32_ulp_and_only_bf16_needs_the_1e_4_bound_relaxed.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`: a 3-layer GELU MLP trained
on random data, with `train_one_optimizer_step` accumulating scaled losses
and stepping once. All runs are CPU, float32 unless stated, and seeded.
Only exercise 1 depends on wall-clock time, and it asserts only the shape of
the curve, with wide margins.

### 1 — samples per second flattens by effective batch 16-32, and every point's loss sits at ln 16

**The curve flattens by effective batch 16-32 (accum 4-8), and is flat past
64.** The run uses the lesson's own arguments with `--num-steps 100`
(micro-batch 4), with the grid extended to accum 64. The sweep runs 3 times,
and each point's rate is effective batch / its best median step time. The
plateau is the median rate at effective batch 64-256. One run:

| effective batch | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| samples/s | 26,337 | 30,938 | 33,961 | 35,786 | 36,836 | 37,365 | 37,646 |
| share of plateau | 70% | 83% | 91% | 96% | 99% | 100% | 101% |

Each step pays a fixed cost: `zero_grads`, a grad norm that calls `.item()`
on all 6 parameters, and `optimizer.step()`. That cost is about half of one
micro-batch's forward and backward (0.05-0.07 ms against 0.105 ms, fitted).
By accum 4-8 it is 6-12% of the step, and the rate sits near the
forward+backward ceiling. Over ten runs effective batch 4 sat at 69-72% of
the plateau. The check asserts only < 85%, and flat by effective batch 64.

**The loss has nothing to smooth.** `synthetic_batch` draws the targets
independently of the inputs. The average loss at all 7 points is 2.774-2.780, identical over the 3 repeats,
within 0.01 of ln 16 = 2.7726, the loss of a uniform guess over 16 classes.
**The shipped `outputs/accum-curve.json` did not come from the documented
command.** It holds accum 1, 2 and 4 at 8 steps each, but the defaults are
accum 1-16 at 25 steps.

### 2 — without the division SGD's first step is 4x too big, but AdamW's is 1.000015x

**The unscaled variant lands 0.0215 from the full-batch reference after
step 1, exactly 3 x lr x max|g| = 3 x 0.1 x 0.0717.** This uses 4 chunks
of 4 on `equivalence_check`'s fixture. The update is 4.0000002x the correct
one. The correctly scaled variant is 1.5e-8 from the reference.

| optimizer | update ratio, unscaled / scaled | step-1 diff from reference |
|---|---:|---:|
| SGD | 4.0000002 | 0.0215 |
| AdamW | 1.000015 | 7.8e-4 |

**AdamW all but hides the bug.** Adam normalises the gradient by its own
root mean square, so a 4x gradient gives almost the same step. The lesson's
"the optimizer step is 16 times too big" is true of SGD only. **The lesson's
log would still show it.** The logged loss reads 8.4088 instead of 2.1022,
and the logged grad norm 1.2109 instead of 0.3027. Both are 4x, under either
optimizer.

### 3 — AdamW's step counter reads 10 after 10 effective steps, and per-micro stepping moves 2.4x as far

**Once per effective step.** `run_config` hard-codes SGD, so the lesson's
sweep is run with `torch.optim.SGD` built as AdamW. After 10 steps at accum 1,
4 and 16, the `step` counter of all 6 parameters reads 10, while the model
saw 10, 40 and 160 micro-batches. Five accumulated AdamW steps match five
full-batch steps to 6.3e-8 in the parameters and 5.5e-12 in `exp_avg_sq`.
**Stepping per micro-batch instead** pushes the counter to 20 and the bias
correction to 1 - 0.999^20. It lands 0.116 from the full-batch run, which
itself moved only 0.049 from its initial weights.

### 4 — the stock no_sync stand-in reports 1 sync per step while real DDP all-reduces all N

**Routed to `DistributedDataParallel.no_sync`, sync_calls and real
all-reduces both drop by N-1 per effective batch.** The setup is a
one-process gloo group, with all-reduces counted by a pass-through comm hook,
over 3 effective steps:

| N | routed: sync_calls / all-reduces | no no_sync | stock `_NoSyncCtx` |
|---:|---:|---:|---:|
| 2 | 3 / 3 | 6 / 6 | 3 / 6 |
| 4 | 3 / 3 | 12 / 12 | 3 / 12 |
| 8 | 3 / 3 | 24 / 24 | 3 / 24 |

With and without no_sync, the gradients are bit-identical. **The lesson's
stand-in reports a saving it does not make.** `sync_counter` counts the code
path, so around a real DDP model it reads 1 per step while DDP still
all-reduces on every backward. World size 1 makes each all-reduce an
identity, so the counts are real but the network cost is not measured.

### 5 — 2x8 and 4x4 agree to one float32 ulp, and only bf16 needs the 1e-4 bound relaxed

**No tolerance needs relaxing in float32.** Here "2 by 8" means 2
micro-batches of 8. Split-vs-split gradient differences on
`equivalence_check`'s fixture (max|g| = 0.0717):

| dtype | 2x8 vs 4x4 | 8x2 vs 4x4 | lesson bound |
|---|---:|---:|---:|
| float32 | 9.3e-9 | 7.5e-9 | 1e-4 |
| float64 | 2.1e-17 | 2.1e-17 | 1e-4 |
| bfloat16 | 4.9e-4 | 4.9e-4 | 1e-4 (fails) |

The splits weight every sample 1/16 either way. Only the order in which
chunk gradients are summed into `param.grad` changes, so float32 differs by
about one rounding step (eps x max|g| = 8.6e-9). **In bfloat16 the gap is
one bf16 ulp at the gradient's magnitude (2^-11).** The fixed absolute 1e-4
fails, and about 1e-3 is needed. A bound written as k x eps(dtype) x max|g|
would carry across dtypes, where the lesson's fixed absolute bound does not.
