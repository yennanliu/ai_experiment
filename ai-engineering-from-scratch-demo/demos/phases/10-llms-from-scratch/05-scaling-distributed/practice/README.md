<!-- generated:start -->
# 10-llms-from-scratch / 05-scaling-distributed

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/05-scaling-distributed/) · upstream spec
`phases/10-llms-from-scratch/05-scaling-distributed/docs/en.md`

```bash
uv run demo practice run 05-scaling-distributed --ex 1
uv run demo explain 05-scaling-distributed --ex 1
uv run pytest demos/phases/10-llms-from-scratch/05-scaling-distributed
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Modify the memory calculator to include activation checkpointing. With checkpointing, only st… | code | T1 | `ex01_checkpointing_saves_the_smallest_term.py` |
| 2 | Extend the pipeline parallelism simulation to implement the 1F1B (one forward, one backward)… | code | T1 | `ex02_the_bubble_is_identical.py` |
| 3 | Implement a gradient accumulation simulator. Instead of all-reducing after every micro-batch,… | code | T1 | `ex03_identical_to_the_wrong_baseline.py` |
| 4 | Build a cost estimator. Given a model size, target token count, GPU type (A100 at $2/hr, H100… | code | T1 | `ex04_the_two_known_costs_use_different_rates.py` |
| 5 | Add ZeRO-Offload to the memory calculator. Assume CPU RAM is 512GB per node and NVMe is 2TB.… | code | T1 | `ex05_the_optimizer_state_does_not_fit_the_node.py` |
<!-- generated:end -->

## Answers

`code/main.py` is a calculator, not a trainer: a memory model, a pipeline
scheduler, a communication-volume formula and a cost estimator. That makes every
exercise here checkable exactly — and four of the five turn out to name a number
that the lesson's own arithmetic does not produce. The pattern is the same each
time: the mechanism is right and the constant beside it is round.

All five are **T1** on the `math` group (`uv sync --extra math`) — `main.py`
imports numpy, so loading the reference needs it even though the arithmetic
below does not.

### 1 — checkpointing saves the smallest term

70B at `hidden_dim=8192`, `num_layers=80`, `sequence_length=2048`, so the
calculator takes its real activation branch.

| batch/GPU | activations | per-GPU total | full checkpointing saves |
|---:|---:|---:|---:|
| **1** *(the default)* | **10.7 GB** | 850.7 GB | **1.2%** |
| 2 | 21.5 | 861.5 | 2.5% |
| 4 | 42.9 | 882.9 | 4.8% |
| 8 | 85.9 | 925.9 | 9.2% |
| 16 | 171.8 | 1011.8 | 16.8% |
| 32 | 343.6 | 1183.6 | **28.7%** |

**FINDING first, because it changes what is being asked:** "only store
activations at every K-th layer (typical K=1, meaning recompute all)" — K=1
stores *every* layer and recomputes nothing. It is the no-checkpointing case.
"Recompute all" is K = `num_layers` = 80.

**ANSWER: at the calculator's own defaults, 1.2% of per-GPU memory for 33% more
compute.** The 70B still does not fit an 80 GB card and never could — weights,
gradients and Adam states are 840 GB of the 850.7.

**FINDING: the answer is a statement about batch size.** Activations are the
only term that scales with the batch, so the saving runs from 1.2% to 28.7% over
the table above, and does not equal the fixed 840 GB until **batch size 78**.
The exercise asks the question as though it had one answer, and the default of 1
is where that answer is smallest.

**FINDING: the calculator's fallback overstates the term 6.5×.** Called without
`hidden_dim` and `num_layers`, activations become `params × 2 × 0.5` = **70 GB**
instead of 10.7 — the difference between activations being 8% of memory and 1.3%
of it, on the one term the exercise is about.

### 2 — the bubble is identical

| micro-batches | GPipe total | GPipe bubble | 1F1B total | 1F1B bubble | `(n-1)/(m+n-1)` |
|---:|---:|---:|---:|---:|---:|
| 4 | 56 | 0.4286 | 56 | 0.4286 | 0.4286 |
| **8** | **88** | **0.2727** | **88** | **0.2727** | 0.2727 |
| 16 | 152 | 0.1579 | 152 | 0.1579 | 0.1579 |

**ANSWER: there is no difference.** 1F1B reorders the same work without removing
any of the pipeline fill and drain. Both arms sit exactly on the GPipe formula,
which also means **the "naive" baseline is already GPipe** — not something worse
for 1F1B to beat.

**ANSWER to the sentence the exercise adds — and this is the real result:**

```text
GPipe  stash per stage:  8, 8, 8, 8      (all micro-batches, before any backward)
1F1B   stash per stage:  4, 3, 2, 1      (num_stages - stage)
```

2× at 8 micro-batches, 4× at 16. The advantage grows with the micro-batch count,
which is the same knob the bubble comparison was pointed at — and on which the
bubble comparison finds nothing.

**FINDING: the exercise's two sentences ask for opposite things.** "Compare the
bubble fraction" has the answer *no difference*; "should have a smaller peak
memory" is what actually separates the schedules. Both waste the same 27.3% of
the pipeline; only one holds eight activation sets while doing it.

### 3 — identical to the wrong baseline

**ANSWER: communication does fall by exactly K.** One all-reduce per 8
micro-batches instead of 8, each moving the same 245.0 GB by the lesson's own
ring-all-reduce formula for a 70B model on 8 GPUs: **1960.0 GB → 245.0 GB**.

**FINDING: the gradients are not identical.**

| | value |
|---|---:|
| max absolute difference | 1.601e-10 |
| max relative difference | **1.097e-07** |
| float32 epsilon | 1.192e-07 |

Addition is not associative in floating point, and the two schedules add in
different orders. "Identical" holds in exact arithmetic and not in the
arithmetic the code runs on.

**FINDING: the baseline it *is* identical to is not the one described.**
Accumulating K micro-batches and stepping once is arithmetically one step at K×
the batch size — not K steps at batch size B. The optimizer takes 1 step where
the other schedule takes 8, and the learning-rate schedule sees 8× fewer points.
"And thus identical training" compares gradient accumulation against a schedule
nobody was running.

**MECHANISM: the saving is per step, and there are K× fewer steps.** Bytes per
micro-batch of data are unchanged. Gradient accumulation buys a larger effective
batch at fixed memory; it does not buy free bandwidth.

### 4 — the two known costs use different rates

| | published GPU-hours | published cost | implied $/GPU-hour | implied MFU |
|---|---:|---:|---:|---:|
| Llama 3 405B | 30.84M | ~$100M | **$3.24** | 34.5% |
| DeepSeek V3 (37B active) | 2.788M | $5.576M | **$2.00** | 33.1% |

**ANSWER: the compute model is exact, once utilisation is not guessed.** Feed
those MFU figures back into 6ND and the estimator returns 3.0830e7 and 2.7852e6
GPU-hours — **within 0.1% of both**.

**FINDING: the lesson's default `utilization=0.4` is the entire error.** At 0.4
the estimator is 13.8% and 17.3% low on hours, in the same direction on both
runs, by about the ratio of 0.4 to the 33–34.5% achieved. An optimistic
constant, not a modelling failure.

**FINDING: no single estimator validates against both dollar figures.** The
hardcoded $3.50/hr sits above both implied rates, so at the matched utilisation
it lands **+7.9%** on Llama 3 and **+74.8%** on DeepSeek. DeepSeek's $2.00 is
their paper's stated rental assumption for the *final run only*; the ~$100M is a
much broader estimate for a company that owns its GPUs. The two numbers answer
different questions.

**FINDING: the 17.9× headline is sparsity, not thrift.** It factors exactly as
**11.1× GPU-hours × 1.62× price**, and those hours are 11.5× FLOPs with the two
runs within 4% of each other on MFU. Costed at its full 671B rather than its 37B
active, DeepSeek V3 comes to **$146.3M** against Llama 3's $93.1M — *more*.

### 5 — the optimizer state does not fit the node

**FINDING: the premise fails its own constraint.** Adam keeps two fp32 moments,
8 bytes a parameter, so 70B is **560 GB** — 48 GB more than the 512 GB of CPU
RAM the exercise assumes per node. The 2 TB of NVMe does fit it, but paging
optimizer state to NVMe is ZeRO-Infinity, not ZeRO-Offload, and an order of
magnitude slower again.

**ANSWER: 5 GPUs, not 4 — and the baseline is 13, not 16.**

| | requirement | first N that fits 80 GB | per card |
|---|---|---:|---:|
| ZeRO-3 | `840/N + 10.7 ≤ 80` | **13** | 75.4 GB |
| ZeRO-3 + optimizer offload | `280/N + 10.7 ≤ 80` | **5** | 66.7 GB |

**FINDING: the cost is 2× the whole step, not 30–50% of the optimizer step.** At
5 GPUs, offload moves gradients down and updated parameters back every step:

```text
transfer  2 × 140 GB / 5 = 56 GB per GPU   →  2.24 s at 25 GB/s (PCIe 4.0 x16)
compute   6ND at 40% MFU, 5 × 2048 tokens  →  2.17 s
```

Transfer and compute are the same size, so the step goes to **2.03×** unless
they overlap perfectly. The quoted 30–50% is a figure for the optimizer *step* —
a small part of the step a user waits on.

**FINDING: what is bought is fewer cards, at 2.03× the card-seconds per token.**
13 cards clear 26,624 tokens in one step time; 5 clear 10,240 in 2.03 of one —
1,008 tokens per card-second against 2,048. Offload buys the ability to run at
all on the hardware you have. "Allows a 70B model to train on 4 GPUs instead of
16" reads as though the GPU count were the only thing that changed.
