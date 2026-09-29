<!-- generated:start -->
# 19-capstone-projects / 45-gradient-clipping-amp

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/45-gradient-clipping-amp/) · upstream spec
`phases/19-capstone-projects/45-gradient-clipping-amp/docs/en.md`

```bash
uv run demo practice run 45-gradient-clipping-amp --ex 1
uv run demo explain 45-gradient-clipping-amp --ex 1
uv run pytest demos/phases/19-capstone-projects/45-gradient-clipping-amp
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Replace the synthetic Inf injection with a real loss spike (multiply one batch's target by 1e… | code | T1 | `ex01_a_1e8_target_spike_is_clipped_not_skipped_under_the_lessons_bf16_and_only_fp16_skips_it.py` |
| 2 | Add a `--bf16` mode that switches autocast to BF16 instead of FP16. BF16 has a wider exponent… | code | T1 | `ex02_bf16_leaves_the_lessons_own_demo_at_1_in_20_skips_and_its_cpu_loop_was_already_bf16.py` |
| 3 | Add a unit test that the gradient-clip wrapper returns the pre-clip and post-clip norm correc… | code | T1 | `ex03_the_clip_wrapper_never_calls_clip_grad_norm_and_on_a_generator_reports_a_clip_it_skipped.py` |
| 4 | Add a rolling-window skip-rate computation and a CLI flag that fails the run if the rate exce… | code | T1 | `ex04_one_skip_on_step_0_reads_as_a_100pct_rate_and_the_demo_prints_0_over_its_own_skip.py` |
| 5 | Wire the loop to write the canonical CSV (`step, lr, grad_l2_pre_clip, grad_l2_post_clip, los… | code | T1 | `ex05_the_lessons_csv_leaves_no_file_after_ctrl_c_and_the_flush_matters_only_under_sigkill.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`: the seed-7 toy model from
`build_toy_model`, `AmpTrainState` on CPU (AdamW at lr 1e-2, `max_norm=1.0`),
and the demo's 20 steps. There is no CUDA here. CPU autocast supports BF16
and FP16, but the lesson turns the GradScaler off on CPU, so nothing is
loss-scaled unless a solution enables a CPU GradScaler to stand in for the
CUDA path. That is stated wherever it matters.

### 1 — a 1e8 target spike is clipped, not skipped, under the lesson's BF16 default; only FP16 skips it

**Under the lesson's own settings the skip path does not trigger.** CPU
autocast runs BF16. The spiked loss is 9.79e15 and the gradient norm 1.19e8,
both finite, so the step is taken and clipped to norm 1.0. Under FP16
autocast the same spike skips as `non_finite_grad`: the backward pass
overflows FP16's 65504. Sweeping the factor:

| factor | 1e5 | 1e6 | 1e8 | 1e18 | 1e19 |
|---|---|---|---|---|---|
| BF16 | step | step | step | step | `non_finite_loss` |
| FP16 | step | `non_finite_grad` | `non_finite_grad` | `non_finite_grad` | `non_finite_loss` |

The loss check only fires once the FP32 loss itself overflows, at 1e19.

**Clipping is what keeps the run alive.** Clipped, the BF16 run ends at
0.0614 after 20 steps (0.0555 with no spike) and 2.4e-5 after 100.
Unclipped (`max_norm=1e30`), that single gradient inflates AdamW's second
moment. The run ends at 0.475 after 20 steps and 0.677 after 100, so it is
worse at 100 than at 20.

### 2 — `--bf16` leaves the lesson's own demo at 1 skip in 20, and its CPU loop was already BF16

**On the lesson's demo the skip rate does not drop to zero.** It is 1/20
under FP16 and 1/20 under `--bf16`. The demo's skip is
`inject_inf_into_first_grad`, which writes +Inf into a gradient whatever the
dtype. The claim does hold on a real ×1e8 spike: FP16 skips 1/20, BF16 0/20.

**The mode the exercise asks to add already exists.**
`AmpTrainState(device_type="cpu")` defaults to `torch.bfloat16`, so the
shipped demo never ran FP16. The FP16 mode is the one this exercise had to
add. On CPU the logged scaling factor is 1.0 on every step. An enabled CPU
GradScaler shows the halving the doc describes: steps 4-6 log 65536, 65536,
32768. The halving appears one row late, because the skip row records the
scale from before `update()`.

### 3 — the clip "wrapper" never calls `clip_grad_norm_`, and on a generator it reports a clip it skipped

**The test passes on the lesson's wrapper and fails on two broken versions.**
It uses real gradients from one backward pass (global norm 1.0080091) at two
thresholds that should not clip: 10.0 and the norm itself. It asserts that
pre == post == the global norm, that the norm is within 1e-6 of torch's
float32 total norm (gap 4.9e-8), and that every gradient is bit-identical
afterwards. A version that always reports `post = max_norm` fails one
assertion, and so does a version that always rescales.

**Two claims in the lesson do not hold.** The doc and docstring call the
function "a wrapper around `torch.nn.utils.clip_grad_norm_`", but it
re-implements the clip and never calls it. The two also differ at the
boundary. At `max_norm` equal to the norm, torch still multiplies the
gradients by max/(norm + 1e-6), while the lesson leaves them untouched.
Passed `model.parameters()` and asked to clip to 0.1,
`clip_global_l2_norm` returns (1.008, 0.1), but the norm afterwards is
still 1.008. The norm pass uses up the generator, so the scaling loop runs
over nothing. `AmpTrainState` passes a list, so the shipped loop is not hit.

### 4 — one skip on step 0 reads as a 100% rate, and the demo prints 0 over its own skip

**`--max-skip-rate` fails a steady 10% skip rate and passes clean and 2%
runs.** Each run is 300 real steps with a 100-step window, a 5% threshold
and 100 steps of patience:

| skips injected | skips | exit | fails at step |
|---|---:|---:|---:|
| none | 0 | 0 | none |
| every 50th step | 6 | 0 | none |
| every 10th step | 30 | 1 | 108 |
| step 0 only | 1 | 0 | none |

The lesson's `rolling_skip_rate` divides by the steps seen so far, not by
the window. One skip on step 0 therefore reads 1.0, 0.5, 0.333 and stays
over 5% for 19 steps. With a patience of 1 the flag would fail that run at
step 0; the 100-step patience keeps it quiet. The lesson's own demo prints
`skip_count=1 final_skip_rate=0.0000`: its 10-step window has moved past
the skip on step 5 by the time it reports.

### 5 — the lesson's CSV leaves no file after Ctrl-C, and the flush matters only under SIGKILL

**Flushed after every row, the canonical CSV survives Ctrl-C with all 8
rows.** The solution runs the loop in a child process that sends itself a
real signal after step 7, then reads what is on disk. The header is the
exercise's 8 columns in order, and row 5 is `skipped=1, non_finite_grad`.

| writer | SIGINT (Ctrl-C) | SIGKILL |
|---|---|---|
| lesson's `write_step_log_csv` after the loop | no file | not run |
| streaming, flush every row | 8 rows | 8 rows |
| streaming, no flush | 8 rows | 0 bytes |

For Ctrl-C itself, the flush is not what saves the rows. The
KeyboardInterrupt unwinds the `with open(...)` block, and closing the file
writes the buffer. The flush earns its place when the process dies without
unwinding: a SIGKILL, an OOM kill, or `os._exit`. Surviving an OS crash
would also need `os.fsync`, which this solution does not measure.
