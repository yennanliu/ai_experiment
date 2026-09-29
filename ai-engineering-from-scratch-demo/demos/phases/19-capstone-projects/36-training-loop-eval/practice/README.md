<!-- generated:start -->
# 19-capstone-projects / 36-training-loop-eval

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/36-training-loop-eval/) · upstream spec
`phases/19-capstone-projects/36-training-loop-eval/docs/en.md`

```bash
uv run demo practice run 36-training-loop-eval --ex 1
uv run demo explain 36-training-loop-eval --ex 1
uv run pytest demos/phases/19-capstone-projects/36-training-loop-eval
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add `weight_decay_groups()` unit tests that confirm scale and bias parameters land in the no… | code | T1 | `ex01_the_decay_split_passes_6_of_6_tests_and_either_half_of_its_rule_alone_passes_them_too.py` |
| 2 | Replace synthetic random tokens with bytes from a small text file so the demo trains on somet… | code | T1 | `ex02_on_real_text_0_of_192_sampled_bytes_leave_the_file_and_the_synthetic_demos_val_loss_climbs_to_7_49.py` |
| 3 | Add a `min_lr` floor of 10 percent of `max_lr` to the cosine schedule and re-plot. | code | T1 | `ex03_the_10pct_floor_is_already_the_default_and_the_80_step_run_never_reaches_it.py` |
| 4 | Save a checkpoint every `eval_every` steps in addition to the JSONL log. Add a `resume_from`… | code | T1 | `ex04_resume_is_bit_exact_only_if_the_batch_stream_is_fast_forwarded_too.py` |
| 5 | Log per step throughput (tokens per second) next to the loss and confirm it stays in a steady… | code | T1 | `ex05_throughput_holds_a_steady_band_on_76_of_80_steps_and_each_probe_step_drops_it_4x.py` |
<!-- generated:end -->

## Answers

Every exercise imports and runs the lesson's `code/main.py`: a 2-layer,
64-wide GPT trained for 80 steps by `train()` with AdamW, warmup plus cosine
LR, and a probe (held-out loss and a sample) every 20 steps. All runs are on
CPU, seeded, with torch pinned to one thread. External fact read on
2026-09-29: minGPT's `configure_optimizers`
(https://raw.githubusercontent.com/karpathy/minGPT/master/mingpt/model.py).

### 1 — the decay split passes 6 of 6 tests, and either half of its rule alone passes them too

**All six tests pass on the tied, untied and bias-free models.** `main.py`
has no `weight_decay_groups()`, so the tests target `build_param_groups`. On
the demo model, 10 matrices (116,736 values) are decayed and 18 vectors
(1,792 values) are not. To show the tests can fail, they were also run on
three broken splits:

| split | tests failed (of 6) |
|---|---:|
| lesson `build_param_groups` | 0 |
| dim test only / name test only | 0 / 0 |
| decay everything | 2 |
| decay nothing | 2 |
| swap the groups | 4 |

**The rule is redundant.** The `dim() < 2` test and the `.bias`/`.shift`/
`.scale` name test each produce the same groups on their own. The lesson's
own tests never check an embedding, so they could not tell this split from
minGPT's, which excludes `nn.Embedding` from decay. **The demo mislabels its
parameter count:** it prints 118,528 as "(untied count)", but
`parameters()` yields the shared embedding/head matrix only once, so that is
the tied count. Untied is 134,912.

### 2 — on real text 0 of 192 sampled bytes leave the file, and the synthetic demo's val loss climbs to 7.49

**The generated sample uses only characters from the file.** Trained for 80
steps on the Zen of Python (856 bytes, 45 distinct), three 64-byte samples at
the demo's sampler settings (T = 0.8, top_k = 20) contain 0 bytes that are
not in the file. The untrained model draws 46 of 64 from outside it. The
output is not yet legible: val loss is 2.973 nats against a unigram entropy
of 3.109, so the model has little more than letter frequencies.

**top_k does part of the work.** With top_k = 0 at T = 1.0, 47 of 2,000
sampled bytes (2.4%) fall outside the file.

**The lesson's synthetic demo cannot show held-out loss dropping.** Train and
val come from different seeds, so they repeat unrelated 32-token patterns.
The code comment says "the eval loss should drop visibly", but it goes the
other way:

| | untrained | step 19 | step 79 |
|---|---:|---:|---:|
| val loss, synthetic demo | 5.554 | 6.535 | 7.492 |

Train loss meanwhile falls to 1.101. On real text the probe does its job:
after 400 steps train loss is 0.960 while val loss rises from 2.981 to
3.455, which is overfitting on 770 training bytes.

### 3 — the 10% floor is already the default, and the 80-step run never reaches it

**With the floor, the LR ends at 3.01e-4 instead of 1.5e-6.**

    floor 10%  ▁▃▅▆██████████▇▇▇▇▆▆▆▅▅▅▄▄▄▃▃▃▃▂▂▂▂▂▂▁▁▁
    floor 0    ▁▃▅▆██████████▇▇▇▆▆▆▅▅▅▄▄▄▃▃▃▂▂▂▁▁▁▁▁▁▁▁

Mean LR rises by 8.4%, and the final train loss moves from 1.130 to 1.101.
**`TrainConfig` already has the floor:** `min_lr / max_lr` is exactly 0.1,
and the lesson's committed `outputs/losses.jsonl` matches that schedule on
all 80 rows. The run never reaches the floor. Step 79 is 0.45% above it,
and only step 80, which never runs, returns 3e-4 exactly; that is the step
the lesson's unit test checks. Warmup does not "ramp from zero" as the doc
says: step 0 is 3e-4, which is the floor value, and steps 9 and 10 both sit
at the peak.

### 4 — resume is bit-exact only if the batch stream is fast-forwarded too

**Checkpoint-and-resume reproduces the uninterrupted run exactly.** The loop
is rebuilt from the lesson's own pieces, because `train()` hides its optimizer
and deletes its log on entry. Before any checkpointing it matches `train()`
with a worst difference of 0. It saves `{model, optimizer, step}` at steps
19, 39, 59 and 79, each about 3.0x the parameter bytes. After a crash at
step 45, resuming from `ckpt_39.pt` into a freshly initialised model gives
steps 40-79 identical to the straight run.

| resume variant | records equal to straight (of 40) |
|---|---:|
| model + optimizer + batch stream fast-forwarded | 40 |
| model + optimizer, stream restarted | 0 (up to 0.698 off) |
| model only | 1 (up to 0.123 off) |

**The JSONL log cannot tell you where to resume.** The doc says you can
resume "by reading the last step". After the crash the log ends at step 45,
but the newest checkpoint is step 39, so the resumed log has 86 lines with
steps 40-45 written twice. The lesson's `train()` would erase the log
instead: a second call leaves 1 line.

### 5 — throughput holds a steady band on 76 of 80 steps, and each probe step drops it 4x

**Yes, outside the probe steps.** Every step trains on 128 tokens, and
`tokens_per_sec` is written beside `train_loss` in all 80 JSONL records.
Using CPU time, the 5th-95th percentile of non-probe steps stays within
about 0.9x-1.02x of the median, which was about 58,000 tokens/s on the
authoring machine. The check asserts a 0.5x-2x band, because the absolute
rate depends on the machine.

**The dips are the probe, not training.** Steps 19, 39, 59 and 79 run 21
forward passes instead of 1: the training forward, 4 eval batches and 16
generation steps. Their throughput falls about 4.4x (asserted as more than
2x). None of the extra tokens count toward the 128, so a naive per-step
rate shows four false dips in every run.
