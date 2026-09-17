<!-- generated:start -->
# 11-llm-engineering / 08-fine-tuning-lora

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/11-llm-engineering/08-fine-tuning-lora/) · upstream spec
`phases/11-llm-engineering/08-fine-tuning-lora/docs/en.md`

```bash
uv run demo practice run 08-fine-tuning-lora --ex 1
uv run demo explain 08-fine-tuning-lora --ex 1
uv run pytest demos/phases/11-llm-engineering/08-fine-tuning-lora
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Rank ablation study. Run the demo with ranks 2, 4, 8, 16, 32, and 64. Plot final loss vs. ran… | code | T1 | `ex01_the_task_is_random_labels_so_the_loss_never_halves.py` |
| 2 | Target module comparison. Modify inject_lora to target only layer "0", only layer "2", only l… | code | T1 | `ex02_the_three_layers_differ_in_shape_not_in_role.py` |
| 3 | Quantization error analysis. Take the trained model's weight matrices before and after quanti… | code | T1 | `ex03_the_function_named_nf4_is_a_fifteen_level_int4.py` |
| 4 | Multi-adapter serving. Train two LoRA adapters on different subsets of the data (even indices… | code | T1 | `ex04_the_loader_reads_two_of_the_four_keys_it_saved.py` |
| 5 | Merge vs. unmerged inference. Compare the output of the LoRA model before and after merge_lor… | code | T1 | `ex05_the_merge_is_fifty_percent_faster_for_five_percent_fewer_macs.py` |
<!-- generated:end -->

## Answers

All five exercises are **T1** on the `llm` group (torch-CPU); the whole lesson
runs in under two seconds. CI runs at `DEMO_TIER=T0` and skips it.

The lesson's LoRA implementation is correct — the merge is exact to 5e-7, the
adapters swap cleanly, the quantiser round-trips at 7% of the weight RMS. What
the exercises measure instead is the fixture: `create_demo_data` draws the
inputs and the labels **independently**, so exercises 1 and 2 are asking about
the shape of a memorisation curve.

### 1 — the labels are random, so the loss never halves

| rank | trainable | % of total | final train loss | ratio to previous | held-out |
|---:|---:|---:|---:|---:|---:|
| 2 | 4,628 | 1.1% | 0.0855 | — | 0.1000 |
| 4 | 9,256 | 2.2% | 0.0759 | 1.13× | 0.1026 |
| 8 | 18,512 | 4.4% | 0.0612 | 1.24× | 0.1062 |
| 16 | 37,024 | 8.4% | 0.0579 | 1.06× | 0.1061 |
| 32 | 74,048 | 15.5% | 0.0578 | 1.00× | 0.1027 |
| 64 | 148,096 | 27.1% | **0.0612** | **0.94×** | 0.1003 |

**ANSWER: the loss never halves at any rank.** The largest consecutive ratio is
1.24× against the 2.0× the exercise looks for, and the curve turns **up** at the
last doubling.

**MECHANISM:**

```python
x = torch.randn(n_samples, d_model)                      # inputs
y = torch.randint(0, n_classes, (n_samples,))            # labels, independently
```

The targets' variance is **0.0900** exactly — the MSE a constant 0.1 prediction
scores, and the floor for any model that has not memorised the sample.

**FINDING: every loss below 0.09 is memorisation.** Training loss reaches 0.0578;
held-out loss never goes below 0.1000, and drifts *upward* with rank. The
improvement in the training column is 500 random labels being learned by heart.

**CONTROL: give the task a signal** — targets set to a fixed random linear map of
the inputs — and final loss falls monotonically with rank (0.9627 → 0.6488 →
0.4557 at ranks 2, 8, 32) with held-out following it down.

### 2 — the three layers differ in shape, not in role

| targets | trainable | final loss | epochs to own −10% | epochs to 0.09 | held-out |
|---|---:|---:|---:|---:|---:|
| `"0"` | 6,144 | 0.0542 | 16 | 3 | 0.1039 |
| `"2"` | 8,192 | 0.0619 | 14 | 2 | 0.1049 |
| `"4"` | 4,176 | 0.0879 | **2** | 15 | 0.1010 |
| all three | 18,512 | **0.0157** | **19** | 2 | **0.1439** |

**ANSWER: all three wins, by having 2.3× the adapter of the next best.**

**FINDING: the ranking is not stable in the parameter count either.** Layer 0's
6,144 parameters beat layer 2's 8,192.

**MECHANISM: the analogy does not hold.** The three linear layers are
`(256, 512)`, `(512, 512)` and `(512, 10)` — different shapes, different
positions, none interchangeable. In an attention block `q_proj` and `v_proj` are
the *same* shape and play different roles; here the shape *is* the difference.

**FINDING: "convergence speed" against each run's own endpoint inverts the
ranking.** Layer 4 "converges" in 2 epochs because it never gets anywhere; the
all-three arm takes 19 because it is still improving when the run ends. Against
a shared threshold the order goes back.

**FINDING: none of it survives held-out.** Every arm is above 0.09, and the best
training arm is the worst held-out arm.

### 3 — the function named `nf4` is a 15-level symmetric int4

| block size | MSE | max abs error | relative RMS | correlation |
|---:|---:|---:|---:|---:|
| 32 | 6.054e-06 | 0.00446 | 0.0682 | 0.997702 |
| 64 | 6.335e-06 | 0.00446 | 0.0697 | 0.997591 |
| 128 | 6.480e-06 | 0.00446 | 0.0705 | 0.997539 |
| 256 | 6.559e-06 | 0.00446 | 0.0709 | 0.997507 |

**ANSWER: 7% of the weight RMS, and block size is worth 4% of it** — an 8.3%
MSE spread for an 8× change, with the max absolute error identical to five
decimal places.

**FINDING: one of the sixteen levels is unreachable.**

```python
scales = blocks.abs().max(dim=1, keepdim=True).values / 7.0
quantized = torch.round(blocks / scales).clamp(-8, 7)
```

Dividing by `max/7` means the most negative value in a block maps to −7. The
codes that ever appear are `[-7 … 7]` — **15 of 16**, a 4-bit format used at
**3.91 bits**.

**FINDING: it is not NF4, and the reconstruction proves it.** NF4 is a fixed
16-value codebook, so a dequantised tensor could hold 16 distinct numbers. This
one holds **28,080**, because every block carries its own scale. The function is
block-wise symmetric integer quantisation: correct, useful, misnamed.

**CONTROL: interleave runs of 32 elements at 100×.** The fraction of small
elements reconstructed as exactly 0.0 goes **0.069 at block 32 → 1.000 at block
64 and 256**. The knob works; `nn.Linear`'s uniform, frozen initialisation gives
it no dynamic range to track.

### 4 — the loader reads two of the four keys the saver wrote

**ANSWER: the swap works.** Two adapters, three layers each, differ by up to
0.5044 on the same input, and every frozen parameter is bit-identical before and
after both loads.

```python
# save_lora_adapter writes, per layer:
adapter_state[f"{name}.A"]      adapter_state[f"{name}.rank"]
adapter_state[f"{name}.B"]      adapter_state[f"{name}.alpha"]

# load_lora_adapter reads:
module.A.data = adapter_state[a_key]
module.B.data = adapter_state[b_key]
```

**FINDING: a rank-4 adapter loads into a rank-8 host, and is wrong.**

```text
before load:  A (256, 8)   rank 8   scaling 2.0
after  load:  A (256, 4)   rank 8   scaling 2.0      <- trained at 4.0
```

`module.A.data = tensor` rebinds, so the shape changes with no error while
`rank` and `scaling` stay behind. The adapter then differs from *itself* in a
correctly sized host by up to **0.2067**.

**MECHANISM:** `LoRALayer.__init__` sets `self.scaling = alpha / rank` once and
`forward` multiplies by it. Nothing re-derives it from `A.shape[1]`, so a shape
change cannot correct it.

**CONTROL: two lines.** Read `rank` and `alpha` back from the state the saver
already wrote and recompute `scaling` — the mismatched load becomes
bit-identical, max difference **0.0**.

### 5 — the merge is 1.5× faster for 4.6% fewer multiply-accumulates

**ANSWER: the outputs agree to 5.4e-07**, comfortably inside the 1e-5 tolerance.

| | MACs / sample | tensor ops | wall clock |
|---|---:|---:|---|
| unmerged | 416,848 | **9** | 1.00× |
| merged | 398,336 | **3** | **1.54×** |

**ANSWER: merged is faster, and it is not the arithmetic.** A 4.6% arithmetic
saving cannot buy a 54% speedup. What drops is the per-operation dispatch: each
`LinearWithLoRA` runs one `nn.Linear` plus the two matmuls inside
`LoRALayer.forward`, so it is **one instead of three**, not "one instead of two".

**FINDING: the overhead is concentrated in the narrowest layer.**

| layer | shape | adapter MACs as % of the layer |
|---|---|---:|
| 0 | 256 → 512 | 4.7% |
| 2 | 512 → 512 | 3.1% |
| 4 | 512 → 10 | **81.6%** |

The adapter's cost depends on the rank, not on the output width, so a narrow
output projection nearly doubles its own cost.

**FINDING: merging is not reversible, and that is the trade.**
`merge_lora_weights` writes into `linear.weight.data` in place and replaces the
wrapper with the bare linear, leaving **0** adapters in the model. The merged
model cannot be swapped to another adapter — which is exactly what exercise 4's
multi-adapter serving needs. Speed and swapability are exclusive here, and the
lesson asks for both.
