<!-- generated:start -->
# 19-capstone-projects / 81-end-to-end-distributed-train

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/81-end-to-end-distributed-train/) · upstream spec
`phases/19-capstone-projects/81-end-to-end-distributed-train/docs/en.md`

```bash
uv run demo practice run 81-end-to-end-distributed-train --ex 1
uv run demo explain 81-end-to-end-distributed-train --ex 1
uv run pytest demos/phases/19-capstone-projects/81-end-to-end-distributed-train
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a tensor-parallel split of the attention head and verify the loss matches the single-rank… | code | T1 | `ex01_allreducing_the_attention_output_matches_the_loss_but_only_megatrons_f_and_g_train_the_same_model.py` |
| 2 | Add gradient accumulation across 4 microbatches and prove the gradient equals the gradient of… | code | T1 | `ex02_4_microbatches_match_one_batch_to_5e_7_and_replay_the_lessons_4_rank_run_on_1_rank.py` |
| 3 | Add a resume-from-step-10 path that actually continues training to step 20 and produces the s… | code | T1 | `ex03_the_step_10_resume_is_bit_exact_with_adam_state_and_drifts_to_4_2399_without_it.py` |
| 4 | Add a metrics export (loss, grad norm, step time) to JSONL so the run can be visualised after… | code | T1 | `ex04_rank_0s_grad_norm_is_2x_the_runs_and_the_loss_rises_at_10_of_19_steps.py` |
| 5 | Add a NaN guard that rolls back to the previous checkpoint on a loss spike, and force a spike… | code | T1 | `ex05_rollback_replays_the_run_exactly_but_300x_lr_slips_past_the_2x_guard_and_a_spike_before_the_checkpoint_poisons_it.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py` (torch 2.14.0, CPU) on real
gloo process groups. Each file starts itself once per rank as a subprocess
on localhost, with a 120-150 s timeout, and kills any rank still alive at
the end. Checkpoints, rendezvous files and the JSONL go to temporary
directories. Where the lesson's own `_train_worker` can be used unmodified
(exercises 2, 3 and 4) it is. Its `mp.Queue` is replaced by one that prints,
because the worker ends in `os._exit`.

### 1 — allreducing the attention output matches the loss, but only Megatron's f and g train the same model

**With Megatron's f and g, the loss matches the single-rank baseline at all
20 Adam steps to 4.8e-7, and the two ranks stay bit-identical.** Each rank
keeps half the heads in both blocks: 4,096 of the 8,192 attention weights.

**The recipe as worded, which only allreduces the output, passes a step-0
check but trains a different model.** The attention input is replicated, so
its gradient is a sum over both ranks' heads. Megatron's `f` allreduces it.
Without `f`, each rank backpropagates only its own heads' share.

| backward rule | step-0 loss gap | 20-step curve gap | rank 0 vs rank 1 | step-0 embedding grad |
|---|---:|---:|---:|---:|
| `f` + `g` (Megatron) | 0 | 4.8e-7 | 0 | 1.000x |
| output allreduce only | 0 | 3.0e-3 | 3.8e-3 | 0.993x |
| `torch.distributed.nn.functional.all_reduce` | 0 | 3.8e-3 | 6.9e-3 | 1.006x |

torch's autograd-aware `all_reduce` errs the other way. It allreduces the
output gradient too, which doubles it.

### 2 — 4 microbatches match one batch to 5e-7, and replay the lesson's 4-rank run on 1 rank

**The accumulated gradient equals the big-batch gradient to 4.8e-7 relative
to its largest element, over all 29,824 elements.** The big batch is the 16
sequences of the lesson's step 0. It is not bit-equal, because the sums run
in a different order. Without the 1/4 loss scaling the norm is exactly 4.0x.

**The lesson's 4-rank run is itself gradient accumulation.** One rank running
the lesson's `ZeroOptimizer` accumulates the 4 ranks' batches as
microbatches. It reproduces the 20 rank-0 losses of the unmodified 4-rank
run to 4.8e-7 and the final norm 54.513171 to 8.7e-9.

**The equality needs equal microbatches.** Split the 16 sequences 1/3/5/7
and scale each loss by 1/4, and the gradient is off by 124% of its largest
element. Weighting each loss by n/16 makes it exact again (3.8e-7). The
lesson's division by world size makes the same equal-batch assumption.

### 3 — the step-10 resume is bit-exact with Adam state, and drifts to 4.2399 without it

**Resuming from the lesson's step-10 checkpoint reproduces steps 10-19 bit
for bit.** The final rank-0 loss is 4.222419 and the final parameter norm is
54.513171 on every rank, in both runs. Adam's step count is 20.

| resume | step 10 loss | steps 11-19 | final loss |
|---|---|---|---:|
| model + `ZeroOptimizer` state | identical | identical | 4.2224 |
| weights only, fresh Adam | identical | all differ | 4.2399 |

**The lesson never resumes, and 55% of its checkpoint is copies of the model.**
`verify_resume` only compares saved bytes with an in-memory snapshot, and
`main`, `run_e2e` and `verify_resume` never call `load_state_dict`. Each of
the 4 shard files (216,655 bytes) carries the full 119,296-byte
`model_state`, identical on every rank. The concatenated master shards
already equal it. Loading at world size 2 is refused: "world_size mismatch:
manifest=4, expected=2".

### 4 — rank 0's grad norm is 2x the run's, and the loss rises at 10 of 19 steps

**Rank 0 writes one well-formed JSON line per step, and training is
unchanged.** The export wraps the lesson's `cross_entropy` call and
`ZeroOptimizer.zero_grad`/`step` without editing the loop. Each row has
`step`, `loss`, `loss_rank0`, `grad_norm`, `grad_norm_rank0` and
`step_time_s`. The global values are averaged over the 4 ranks, and a step
takes about 5-18 ms. Rank 0's losses and the final norm are bit-identical to
a run without the export.

**The grad norm one rank sees is 1.8-2.1x the run's.** Averaging 4
independent noisy gradients divides the norm by about sqrt(4) = 2. A
dashboard fed from one rank would be off by that factor.

**The curve contradicts the doc's invariant (a), that "the loss decreases
monotonically".** The global loss rises at 10 of 19 steps and rank 0's at 8.
The corpus is uniform random tokens, so ln 64 = 4.1589 is the best loss any
model can expect. The global loss goes from 4.3373 to 4.2111 and never drops
below it. Rank 0's loss dips below it at steps 12 and 14, which is batch
noise, not learning.

### 5 — the rollback replays the run exactly, but 300x LR slips past the 2x guard, and a spike before the checkpoint poisons it

**A 1000x LR step at step 12 trips the guard at step 13, and the rollback
replays the clean run exactly.** The guard uses the doc's rule: a loss over
2x the previous step's, or a non-finite one. It checks the loss averaged
over the ranks, so all ranks agree. The loss jumps from 4.1843 to 70.4817.
Every rank reloads its shard of the step-10 checkpoint, and steps 10-19 then
match the clean run bit for bit (final loss 4.2111).

**No multiplier makes a NaN, and the 2x rule misses real damage.** Adam
normalises the step size and LayerNorm the activations:

| LR multiplier on one step | 10 | 100 | 300 | 500 | 1000 | 1e6 |
|---|---:|---:|---:|---:|---:|---:|
| next loss / previous loss | 1.03 | 1.09 | 1.37 | 3.50 | 16.8 | 2e7 |
| trips the 2x guard | no | no | no | yes | yes | yes |

Unguarded, the 300x run still ends at loss 5.38 against 4.21. The doc says
"the guard is unused but the hook stays", but `main.py` has no guard or hook.

**A spike just before the checkpoint poisons it.** With 1000x at step 9, the
step-10 checkpoint is written from the spiked weights before any loss can
show it. Every rollback reloads loss 76.933 and trips again. The cap of 3
retries is all that stops it. A safe rollback needs an older checkpoint, or
one that is only trusted after the next step's loss has been checked.
