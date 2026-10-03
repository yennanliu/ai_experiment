<!-- generated:start -->
# 00-setup-and-tooling / 12-debugging-and-profiling

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/00-setup-and-tooling/12-debugging-and-profiling/) · upstream spec
`phases/00-setup-and-tooling/12-debugging-and-profiling/docs/en.md`

```bash
uv run demo practice run 12-debugging-and-profiling --ex 1
uv run demo explain 12-debugging-and-profiling --ex 1
uv run pytest demos/phases/00-setup-and-tooling/12-debugging-and-profiling
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `debug_tools.py` and read through each section's output. Modify the dummy model to introd… | code | T0 | `ex01_a_zero_divide_before_tanh_is_caught_one_step_late.py` |
| 2 | Profile a training loop with `cProfile` and identify the slowest function. | code | T0 | `ex02_the_slowest_function_is_random_gauss_and_matmul_has_no_row.py` |
| 3 | Use `tracemalloc` to find which line in your data loading pipeline allocates the most memory. | code | T0 | `ex03_the_end_snapshot_names_the_wrong_line.py` |
| 4 | Set up TensorBoard for a simple training run and identify whether the model is overfitting. | code | T0 | `ex04_val_loss_doubles_while_val_accuracy_peaks.py` |
| 5 | Use `breakpoint()` inside a training loop. Practice inspecting tensor shapes, devices, and gr… | code | T0 | `ex05_the_gradient_at_the_breakpoint_is_stale.py` |
<!-- generated:end -->

## Answers

The lesson's `code/debug_tools.py` is ten torch demos around five helpers
(`debug_print`, `Timer`, `detect_nan`, `check_gradient_health`, `check_shapes`).
torch is not installed here or in CI, and without it `main()` runs **2 of its 10
sections** and returns 1. The helpers only duck-type their arguments, though, so
the solutions call them on numpy stand-ins with the torch methods they use
(`.isnan()`, `.data.norm()`, `.device`), with torch's semantics kept where they
matter (divide-by-zero gradients, `.grad` accumulation). Everything is **T0** on
numpy.

### 1 — a divide-by-zero before tanh is caught one step late

The 784-256-10 dummy MLP, with `/ 0` put into the forward pass in two places:

| where the `/ 0` goes | loss at step 0 | NaN gradients | `detect_nan` | `check_gradient_health` |
|---|---|---|---|---|
| after the ReLU | nan | all 4 | **True** | norm nan |
| before a tanh | **3.206** | `0.weight`, `0.bias` | **False** | norm nan, **no warning** |
| nowhere (control) | 3.408 / 2.663 | none | False | 12.47 / 8.67 |

**ANSWER:** after the ReLU the detector catches it at step 0 — ReLU emits exact
zeros, and `0/0` is nan.

**FINDING: before a tanh nothing catches it.** `tanh(±inf) = ±1`, so the loss is
finite, while the backward pass computes `0 · inf = nan`. `detect_nan` only looks
at gradients once the *loss* is nan; `check_gradient_health` checks `norm > 100`
and `norm == 0`, both False for nan, so it prints the nan norm and no warning.
SGD writes the nan into the weights and the detector fires at step 1.

### 2 — the slowest function is `random.gauss`, and the matmuls have no row

A 30-step numpy MLP loop with a per-sample Python loader, under cProfile:

**ANSWER: `random.gauss`**, 491,520 calls (30 x 64 x 256), inside `load_batch`,
which is ~98% of the loop's cumulative time; forward, backward and SGD are ~1%.

**FINDING:** sorted by `cumtime`, as the lesson's `python -m cProfile -s cumtime`
prints it, the order is `train`, `load_batch`, `gauss` — the entry point always
leads, because cumulative time includes callees. Own time (`tottime`) is the
column that names the bottleneck.

**FINDING:** `@` is an operator, not a call, so 0 rows mention matmul; its cost is
charged to `forward`'s own time.

**CONTROL:** the call counts are exact; the lesson's `Timer` printed 120 lines for
the 30 steps.

### 3 — one snapshot at the end names the wrong line

A three-line CSV pipeline over 2000 x 32 values:

| line | per-line peak | in one end snapshot |
|---|---:|---:|
| `lines = text.splitlines()` | 688 KiB | — |
| `rows = [[float(v) ...]]` | **2,124 KiB** | 6.6 KiB |
| `data = np.array(rows, float32)` | 313 KiB | **250 KiB (top)** |

**ANSWER: the float-parsing line** — 6.8x the array line, 3.6x the 594 KiB of
text: a 24-byte float object plus an 8-byte list slot per value.

**FINDING:** the lesson's recipe (start, run, one `take_snapshot()`) names
`np.array`, because the lists were freed by then; the process peak of 3.05 MiB
is 12x what its top line shows. `tracemalloc.reset_peak()` per line, or a
snapshot while the intermediates are alive (the CONTROL), names the parsing line.

**FINDING:** the lesson's own `demo_memory_tracking` is a tie — 100 x
`bytearray(40000)` against one `bytearray(4000000)`, exactly 4,000,000 bytes
each; 3,913 against 3,906 KiB, ranked by object headers.

### 4 — validation loss doubles while validation accuracy reaches its best

Logistic regression on 100 training points (50 features, 5 informative), logged
through an `add_scalar(tag, value, step)` stand-in for `SummaryWriter`:

| step | train loss | val loss | val accuracy |
|---|---:|---:|---:|
| 57 (val-loss minimum) | 0.329 | **0.565** | 69.8% |
| 1999 (end) | 0.060 | **1.191** | **71.1%** (run best) |

**ANSWER: overfitting, by the lesson's rule** — train loss down, val loss up from
step 57; train accuracy 100%.

**FINDING:** the lesson's own TensorBoard snippet logs only `loss/train` and `lr`
— no validation scalar, so its overfitting rule has no curve to read.

**FINDING:** the rising val loss is overconfidence: val accuracy is at its best
where val loss is worst, so early stopping on val loss keeps the less accurate
model. **CONTROL:** with 2000 training points val loss ends within 1% of its
minimum.

### 5 — at the lesson's breakpoint, the gradient on screen is stale

The lesson's Part 2 `training_step`, one NaN in sample 2 of step 3, and a real
`pdb` fed a fixed command script:

```text
(Pdb) p inputs.shape                              -> (16, 8)
(Pdb) p inputs.device                             -> 'cpu'
(Pdb) p int(np.isnan(outputs).sum())              -> 1
(Pdb) p np.argwhere(np.isnan(inputs)).tolist()    -> [[2, 4]]
(Pdb) !dt.debug_print('outputs', outputs)         -> ... has_nan=True
(Pdb) !dt.check_gradient_health(model)            -> Total gradient norm: 15.1664
```

**ANSWER:** shape, device, the NaN and its location are all one command away.

**FINDING: the gradient there is not this loss's.** The lesson's `breakpoint()` sits
before `loss.backward()`, so `.grad` still holds a finite norm of 15.17 while the
loss is nan and this step's own gradient is nan in 8 of 8 entries. With no
`zero_grad` in the snippet it is the sum of steps 0-2's gradients.

**FINDING:** the NaN reaches the weights, so steps 3 and 4 both stop: one bad
sample stops every later step.

**CONTROL:** the lesson's thresholds disagree — `> 100` in the prose, `> 10` in
`debug_tools.py`. The clean losses here are 17.6, 7.1 and 12.7, so the `> 10`
version would also stop on 2 healthy steps.
