<!-- generated:start -->
# 19-capstone-projects / 79-pipeline-parallel

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/79-pipeline-parallel/) · upstream spec
`phases/19-capstone-projects/79-pipeline-parallel/docs/en.md`

```bash
uv run demo practice run 79-pipeline-parallel --ex 1
uv run demo explain 79-pipeline-parallel --ex 1
uv run pytest demos/phases/19-capstone-projects/79-pipeline-parallel
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement 1F1B and verify the bubble fraction matches GPipe but activation memory is bounded. | code | T1 | `ex01_1f1b_matches_gpipes_bubble_on_42_grids_and_holds_min_n_m_microbatches_instead_of_m.py` |
| 2 | Profile real per-stage time on a deeper model and rebalance stages by measured wall-clock. | code | T1 | `ex02_rebalancing_by_measured_time_halves_the_slowest_stage_yet_idle_stays_above_the_closed_form.py` |
| 3 | Add gradient accumulation across pipeline microbatches and check the gradient equals the grad… | code | T1 | `ex03_the_lessons_gloo_pipeline_steps_on_the_sum_of_microbatch_gradients_4x_the_full_batch.py` |
| 4 | Pair the pipeline with activation checkpointing and measure the memory drop versus compute cost. | code | T1 | `ex04_checkpointing_cuts_the_stash_6x_but_spent_on_more_microbatches_throughput_drops_3_7pct.py` |
| 5 | Combine pipeline with DDP (each pipeline rank is replicated across a data-parallel group) and… | code | T1 | `ex05_a_2x2_pipeline_ddp_grid_matches_the_full_batch_and_send_before_recv_deadlocks_gloo.py` |
<!-- generated:end -->

## Answers

Every exercise loads the lesson's `code/main.py`: a unit-time GPipe
schedule, the closed form (N-1)/(M+N-1), a two-layer `StageMLP`, and a
2-rank gloo pipeline. Files that need several ranks start themselves once
per rank as subprocesses on localhost, with timeouts, and kill any rank
still alive at the end. Nothing needs a GPU. Timings depend on machine
load, so only wide margins are asserted and the measured numbers go in the
check details. The Megatron-LM formulas come from Narayanan et al. 2021
(https://arxiv.org/html/2104.04473, read 2026-09-29).

### 1 — 1F1B matches GPipe's bubble on 42 grids and holds min(N, M) microbatches instead of M

**1F1B's bubble is exactly GPipe's and the closed form on all 42 (N, M)
grids, and its activation memory is bounded by the pipeline depth.** Both
schedules run through one dependency-driven simulator, which reproduces the
lesson's `gpipe_schedule` cycle for cycle. Peak in-flight microbatches at N=4:

| M | 4 | 7 | 8 | 16 | 64 |
|---|---:|---:|---:|---:|---:|
| GPipe | 4 | 7 | 8 | 16 | 64 |
| 1F1B | 4 | 4 | 4 | 4 | 4 |

**The lesson's backward-costs-2x constants are dead code.** `FORWARD_UNITS`
and `BACKWARD_UNITS` are defined and never read. With backward set to 2
units both schedules still give exactly (N-1)/(M+N-1), so the omission is
harmless.

**The lesson's bubble is not the Megatron bubble.** The lesson divides idle
time by total time: 27.3% at N=4, M=8. Megatron divides by ideal time,
(p-1)/m = 37.5%. The "interleaved 1F1B" the objectives mention is a third
schedule, which cuts that by v (18.75% at v=2), so it does not match GPipe.
The doc's `PipelineStage` and `Pipeline(stages, num_microbatches)` do not
exist in `main.py`.

### 2 — rebalancing by measured time halves the slowest stage, yet idle stays above the closed form

**Cutting by measured time makes the slowest stage 2.0x faster than 4 layers
per stage.** Timing the rebuilt stages again gives about 2.1x. The model is
an embedding, 11 narrow and 4 wide `StageMLP` blocks. Idle time at M=8:

| partition | slowest stage vs time-balanced | GPipe idle at M=8 |
|---|---:|---:|
| 4 layers per stage | 2.0x | ~64% |
| balanced parameters | ~1.2x | — |
| balanced measured time | 1.0x | ~39% |
| closed form | — | 27.3% |

Even the best cut idles above the closed form, because one wide block is a
quarter of the model and cannot be split. On a heavily loaded machine one run measured
1.7x and 50% idle. The check asserts only 1.4x (1.3x re-timed).

**An embedding costs more than a FLOP count says.** On CPU its backward
writes a dense gradient for the whole table. `Embedding(16000, 64)` takes
5-6x as long as `Embedding(4000, 64)` on the same 64 tokens. The doc's rule
"equalise FLOPs per stage" would miss that, and a profile does not.

### 3 — the lesson's gloo pipeline steps on the sum of microbatch gradients, 4x the full batch

**Scale each microbatch loss by 1/M and the accumulated gradient equals the
full-batch gradient** (3.7e-9 max difference on both stages). Leave the
scale out and it is exactly M = 4 times the full-batch gradient.

**The lesson's real pipeline leaves it out.** A global optimizer pre-step
hook records the gradient `_pipe_worker` hands `optim.step()` on both gloo
ranks. All 8 tensors are 4.0x the full-batch gradient (relative error
1.8e-7). The doc says the step gradient "is the gradient on the combined
M*B examples". In the code the effective learning rate is 0.05 x M.

### 4 — checkpointing cuts the stash 6x, but spent on more microbatches throughput drops 3.7%

**Checkpointing cuts what each stage stashes from 48 KiB to 8 KiB per
microbatch (6.0x, or 3.4x counting the one microbatch rebuilt during
backward). It costs one extra forward per microbatch.** That is 16 forward
calls per stage instead of 8, priced at 1.33x with the lesson's units.
Gradients are bit-identical. Wall-clock went up 1.6-3.0x across runs here,
more than the unit price. Without checkpointing the stash is exactly linear
in M, as the doc says.

**Spending the saved memory on more microbatches loses throughput:**

| | M | bubble | throughput (microbatches per unit) |
|---|---:|---:|---:|
| GPipe, no checkpointing | 8 | 27.3% | 0.242 |
| same memory, checkpointed | 42 | 6.7% | 0.233 |

The recompute makes each backward 3 units instead of 2, which costs more
than the smaller bubble saves. Checkpointing buys memory here, not speed.

### 5 — a 2x2 pipeline x DDP grid matches the full batch, and send-before-recv deadlocks gloo

**On 4 gloo ranks (2 stages x 2 replicas) the averaged gradients match a
single process's 64-row full batch** (relative error 1.9e-7). The two
replicas of each stage are bit-identical after the all-reduce and differ
before it. Each rank makes 8 point-to-point calls.

In the 2D schedule the two pipelines never talk to each other. They meet
only at one DP all-reduce per stage. In the lesson's GPipe schedule (N=4,
M=8), stage s issues its last backward s cycles before the step ends (0, 1,
2, 3). So stage 3 can hide its all-reduce behind the drain, and stage 0,
which finishes last, cannot hide any of it.

**The doc's deadlock warning holds on gloo, even for 16 floats.** An ordered
pair swaps its tensors, but a pair where both ranks send first is still
stuck after 12 s. Gloo's `send` does not return until the peer receives.
