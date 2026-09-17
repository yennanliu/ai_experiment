<!-- generated:start -->
# 10-llms-from-scratch / 34-gradient-checkpointing

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/34-gradient-checkpointing/) · upstream spec
`phases/10-llms-from-scratch/34-gradient-checkpointing/docs/en.md`

```bash
uv run demo practice run 34-gradient-checkpointing --ex 1
uv run demo explain 34-gradient-checkpointing --ex 1
uv run pytest demos/phases/10-llms-from-scratch/34-gradient-checkpointing
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Verify correctness. Run `model_forward` + `model_backward` (full activations) vs `model_forwa… | code | T0 | `ex01_the_test_cannot_fail_and_the_hazard_is_absent.py` |
| 2 | Sweep segment size `k` from 1 to `L`. Plot FLOP overhead and memory. Find the knee of the curve. | code | T0 | `ex02_the_knee_is_at_one_and_the_curve_is_an_undercharge.py` |
| 3 | Implement selective checkpointing: store the attention-module input but not its intermediates… | code | T0 | `ex03_the_constant_is_right_only_at_the_seq_the_exercise_picks.py` |
| 4 | Add offload. Save segment inputs to a simulated "CPU buffer" (a separate list). Measure "PCIe… | code | T0 | `ex04_the_breakeven_is_the_interconnect_a_list_cannot_have.py` |
| 5 | Benchmark a real PyTorch transformer with and without `torch.utils.checkpoint`. Measure memor… | code | T3 | `ex05_the_instrument_returns_zero_without_raising.py` |
<!-- generated:end -->

## Answers

`code/main.py` is a numpy MLP stack with a hand-written backward pass, a
checkpointed forward and backward, and four closed-form cost models:
`checkpoint_cost`, `activation_memory_mb`, `memory_after_checkpoint` and
`optimal_segment`. The implementation is correct. The cost models describe a
different implementation than the one shipped beside them, and the exercises are
scored on the cost models.

Exercises 1–4 are **T0** on the **math** group (numpy); Exercise 5 is **T3** on
**llm** (it runs a real PyTorch transformer).

### 1 — the test cannot fail, and the hazard it guards is absent

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| max grad diff | **0.0** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**ANSWER: bitwise equal at every segment size**, including the ones that do not
divide the layer count — not "within machine precision". No tolerance is in play.

**MECHANISM: `model_backward` never reads the stored intermediates.**
`layer_backward` recomputes `h_pre` and `h` from `x_in`, so the only thing the
full-activation path uses from its `activations` list is the per-layer input —
which is exactly what the checkpointed path saves. At k=1 the two lists are both
7 tensors and element-for-element identical.

**FINDING: the test does discriminate**, so the 0.0 is a result and not a no-op:
scaling one saved segment input by 1.0001 moves the gradients by 2.7e-04.

**FINDING: the hazard is absent from the model.** Real checkpointing must
reproduce the original forward's RNG state; the module has no stochastic op.
Adding dropout at rate 0.5 and recomputing with a **fresh** mask moves the
gradients by **120% of their own magnitude**; the same seed returns to 0.0. That,
not float error, is what "identical to machine precision" guards against.

### 2 — the knee is at k=1, and the curve is an undercharge

| k | FLOP overhead | memory | share of achievable saving |
|---:|---:|---:|---:|
| 1 | **0.0%** | 8,724 MB | **93.5%** |
| 2 | 16.7% | 4,563 MB | 99.9% |
| 4 | 25.0% | 2,684 MB | 99.5% |
| 8 | 29.2% | **2,147 MB** | 100.0% ← sqrt(64) |
| 16 | 31.2% | 2,684 MB | 99.5% |
| 64 | 32.8% | 8,724 MB | 93.5% |

**ANSWER: there are two curves and only one has a knee.** Against a
no-checkpoint baseline of 103,079 MB, k=1 removes 91.5% of the activation memory
at a modelled zero cost — 93.5% of everything checkpointing can achieve. The
remaining 6.4 points cost 29.2 points of compute. The FLOP curve is monotone,
bounded by 33.3%, and 75% spent by k=4.

**FINDING: the formula charges `(k-1)/k` and the code recomputes `k/k`.**
`model_backward_checkpointed` calls `model_forward` over the whole segment;
`checkpoint_cost` charges for k−1 of its layers. So the real recompute is one
full extra forward at **every** k — a flat 33.3% — and the rising curve the
exercise asks to plot is the `n/k` layer-forwards the model never charges for.

Counting `layer_forward` calls settles it. The full path performs 24; the
checkpointed one performs exactly **24 more at every k**:

| k | 1 | 2 | 4 | 8 | 12 | 24 |
|---|---:|---:|---:|---:|---:|---:|
| charged | **0** | 12 | 18 | 21 | 22 | 23 |
| performed | **24** | 24 | 24 | 24 | 24 | 24 |

At k=1 that is the whole cost: the model says free and the code does an entire
extra pass.

**FINDING: the sqrt-L rule picks the dearest of the memory-tied options.**
`memory_after_checkpoint` is flat near its minimum — at L=32 every k from 4 to 8
ties — and `optimal_segment` returns the large end of the tie at **6 of 9** layer
counts swept, for up to 2.8 points of compute bought with no memory.

### 3 — the constant is right only at the seq the exercise picks

| seq | 1024 | 2048 | 4096 | **8192** | 16384 |
|---|---:|---:|---:|---:|---:|
| `checkpoint_cost(selective=True)` | 5.00% | 5.00% | 5.00% | **5.00%** | 5.00% |
| true attention FLOP share `s/(6h+s)` | 0.0204 | 0.0400 | 0.0769 | **0.1429** | 0.2500 |
| honest overhead | 0.68% | 1.33% | 2.56% | **4.76%** | 8.33% |
| softmax volume ÷ modelled memory | 0.67 | 1.33 | 2.67 | **5.33** | 10.67 |

**ANSWER: 5.0000% selective against 32.2917% full at k=32, a 6.46x reduction** —
and 5.0000% at every sequence length and every segment size, because
`checkpoint_cost` takes no `seq` and its selective branch ignores
`segment_size`. "For a 32-layer model at seq=8192" narrows nothing.

**FINDING: seq=8192 is the one length in the range where `attention_fraction=0.15`
is correct** — within 5% there, off by **7.4x** at seq=1024. The
parameterisation is invisible precisely because of where the exercise evaluates
it.

**FINDING: the half of selective checkpointing that needs the sequence length has
no function.** `memory_after_checkpoint` has no selective mode and no name in the
module mentions selective, so the memory saving the technique exists for cannot
be computed. The exercise asks for the FLOP overhead — the half that does not
need `seq`.

**MECHANISM: the memory model is linear in seq**, exactly 16.0x from 1024 to
16384. The `s²` softmax volume the lesson's own Key Terms calls "the O(L²)
problem" that "dominates activation memory at long contexts" is **8,590 MB per
layer at seq=8192 against a modelled 1,611 MB** — and is not in the model.

### 4 — the breakeven is an interconnect a list cannot have

| hidden | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| recompute ÷ copy | 17x | 15x | 30x | 24x | 50x | 67x | **120x** |

**ANSWER: there is no breakeven inside the toy.** Copying a layer's input is 15x
to 120x cheaper than recomputing it at every width, and the ratio *grows* with
width: recompute is `O(h·inner)`, the copy is `O(h)`. The sweep runs away from
the crossing.

**FINDING: bytes/time measures this machine's cache hierarchy.** The 6.8 MB of
segment inputs copy at ~75 GB/s; a single 67 MB array at ~20 GB/s — about 4x
apart, straddling the 25 GB/s of the PCIe gen4 link the exercise names. A list in
the same address space cannot be slower than memcpy.

**FINDING: the traffic is unmeasurable in place.** The copies are **0.24%** of the
checkpointed forward's time against **8.6%** run-to-run jitter on the forward
itself — a factor of 36. The figures above come from timing them in isolation.

**MECHANISM: the breakeven is `h = F / (6·BW)`.**

| link | PCIe gen4 x16 | PCIe gen5 x16 | NVLink 3 | NVLink 4 |
|---|---:|---:|---:|---:|
| bandwidth | 25 GB/s | 64 GB/s | 300 GB/s | 900 GB/s |
| breakeven hidden | 2,080 | 812 | 173 | **58** |

A 36x range set entirely by the one quantity the "CPU buffer" does not have. At
the lesson's own defaults (64 layers, seq 8192, hidden 8192) a layer's recompute
is 42.3 ms against a 10.7 ms round trip for 134.2 MB — offload wins 3.9x.

### 5 — the instrument returns zero without raising

| | saved bytes | tensors | reduction |
|---|---:|---:|---:|
| no checkpoint | 152.166 MB | 121 | — |
| checkpoint k=1 | 13.631 MB | 13 | 11.2x |
| checkpoint k=2 | 7.340 MB | 7 | 20.7x |
| checkpoint k=4 | 4.194 MB | 4 | 36.3x |
| checkpoint k=12 | **2.097 MB** | **2** | **72.6x** |

**ANSWER: `torch.cuda.max_memory_allocated()` returns 0 and does not raise.**
Without CUDA it reads 0 before the step and 0 after, for both arms — the
exercise's measurement reports that checkpointing saved nothing, silently.

**FINDING: `torch.autograd.graph.saved_tensors_hooks` counts it exactly, on any
device.** The pack hook fires on every tensor autograd retains, so summing
`numel * element_size` is the saved-activation volume by construction — exact
rather than a high-water mark, and needing no GPU.

**MECHANISM: PyTorch's recompute is one extra forward at every k**, so its FLOP
overhead is `fwd / (fwd + bwd)` = **33.3% flat** — not the `(k-1)/k` curve
`checkpoint_cost` plots. What k changes is only the memory; the gradients are
identical at every k.

**FINDING: measured step time rises with k anyway** — +17% to +28% when the run
has the machine to itself, +30% to +104% when it does not — while the number of
`checkpoint` calls *falls* 12x across that same range. So it is neither call overhead nor FLOPs: with `use_reentrant=False` a
segment's recomputed graph is materialised in full and held until that segment's
backward finishes, so a larger k trades saved bytes for a larger transient.
