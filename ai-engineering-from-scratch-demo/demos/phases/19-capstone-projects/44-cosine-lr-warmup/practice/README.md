<!-- generated:start -->
# 19-capstone-projects / 44-cosine-lr-warmup

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/44-cosine-lr-warmup/) · upstream spec
`phases/19-capstone-projects/44-cosine-lr-warmup/docs/en.md`

```bash
uv run demo practice run 44-cosine-lr-warmup --ex 1
uv run demo explain 44-cosine-lr-warmup --ex 1
uv run pytest demos/phases/19-capstone-projects/44-cosine-lr-warmup
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add an inverse-square-root variant of the schedule and compare it on a 200-step toy training… | code | T1 | `ex01_cosine_ends_lower_at_both_peaks_and_inverse_sqrt_was_already_in_main_py.py` |
| 2 | Add a `--restart` flag that adds a second warmup at `total_steps / 2`. Defend whether warm re… | code | T1 | `ex02_a_restart_helps_the_undertrained_run_only_by_adding_27pct_more_lr_and_hurts_the_memorised_one.py` |
| 3 | Add a unit test that the schedule is continuous: for every step in `[0, total_steps]` the dif… | code | T1 | `ex03_the_bound_as_worded_fails_the_demo_by_1_ulp_and_any_warmup_over_39pct_breaks_it.py` |
| 4 | Wire the schedule into a `torch.optim.lr_scheduler.LambdaLR` so it composes with framework co… | code | T1 | `ex04_lambdalr_reproduces_the_run_but_around_trainstates_own_optimizer_it_trains_at_lr_0.py` |
| 5 | Add a `--plot-png` flag that writes a real plot via `matplotlib`. Defend whether the lesson's… | code | T1 | `ex05_the_text_plot_is_the_ci_default_but_hides_the_floor_and_misses_2_of_18_misconfigurations.py` |
<!-- generated:end -->

## Answers

Every exercise imports the lesson's `code/main.py` and runs it: its
`CosineWithWarmup`, `InverseSqrtWarmup`, `TrainState` (AdamW plus a step
counter), `build_toy_model` (a seed-7 16-32-4 MLP on one fixed batch of 8),
`plot_schedule_ascii` and `write_schedule_csv`. The toy runs are CPU,
single-threaded and seeded. matplotlib is not a dependency of this repo, so
exercise 5 writes its PNG with a stdlib encoder.

### 1 — cosine ends lower at both peaks, and inverse-sqrt was already in main.py

**Cosine gives the lower final loss.** Both runs are 200 `TrainState` steps
with warmup 20 and lr_min = lr_max / 100:

| lr_max | cosine | inverse-sqrt |
|---|---:|---:|
| 1e-3 | 0.0956 | 0.1062 |
| 1e-2 (the demo's peak; both memorise the batch) | ~5e-10 | ~1e-6 |

The variant was already there. `main.py` ships `InverseSqrtWarmup`, and the
lesson's tests cover it. It takes no `total_steps` and no `lr_min`, so at
step 199 it still sits at 31.7% of peak, while cosine sits at 1.0%. The
answer also depends on where the run stops. The two logs are bit-identical
through step 20, since the warmups match. At 1e-2, inverse-sqrt has the
lower loss on 66 of the 200 steps, all between steps 26 and 145, and cosine
overtakes it as it anneals.

### 2 — a restart helps the under-trained run only by adding 27% more LR, and hurts the memorised one

**On this toy run, warm restarts help only when the model is under-trained,
and the help is mostly extra learning rate.** `--restart` keeps the first
100 steps and then starts a fresh `CosineWithWarmup(20, 100)` cycle.

| schedule (lr_max 1e-3) | summed LR / lr_max | final loss |
|---|---:|---:|
| no restart | 100.9 | 0.0956 |
| `--restart` | 128.4 | 0.0233 |
| no restart, peak raised to match the area | 128.4 | 0.0273 |
| SGDR, two 100-step cycles (same area) | 100.8 | 0.0915 |

The area-matched cosine recovers 95% of the gain without any restart. At
1e-2 the batch is already memorised, and the restart takes the residual
loss from 5.2e-10 to 9.9e-10. The "warm" restart also starts cold: the
second warmup ramps from 0, so the LR at step 100 falls from 5.99e-3 to 0.
In SGDR at 1e-2 the loss rises 46% within 10 steps of the restart. On a
fixed batch there is no held-out set, and that is where a restart is
supposed to pay off.

### 3 — the bound as worded fails the lesson's demo by 1 ulp, and any warmup over 39% breaks it

**The unit test (`ContinuityTest`) passes on the demo (4/20), the test
suite's configuration (10/100) and a 200-step configuration**, provided it
allows a 1e-12 relative tolerance. The largest step is exactly the warmup
slope.

Without the tolerance, the test fails on all three, by 1, 6 and 4 ulps,
because `lr(k+1) - lr(k)` is a difference of two rounded products. A sweep
covered every configuration the lesson accepts for total_steps 2-200: 79,600
schedules across four (lr_max, lr_min) pairs.

| bound | schedules that fail |
|---|---:|
| exact, as worded | 76,059 (95.6%) |
| with 1e-12 tolerance | 47,637, all in the cosine region |

The failures with the tolerance are not bugs in the code. The cosine's
steepest step is larger than lr_max / warmup once warmup exceeds
total / (1 + (pi/2)(1 - lr_min/lr_max)), which is 38.9% of the run when
lr_min is 0. That closed form predicts 47,653 failures and misses none of
the measured ones. The lesson's validator accepts any warmup below
total_steps. With warmup_steps = 0, which the lesson explicitly supports,
the bound is a division by zero.

### 4 — LambdaLR reproduces the run, but wrapped around TrainState's own optimizer it trains at LR 0

**The numbers do not change; the representation and the owner of the
schedule do.** Build AdamW at lr_max and use `LambdaLR(opt, lambda k:
lr(k) / lr_max)`, stepped after `optimizer.step()`. The LR then matches
`TrainState` bit for bit on 18 of 20 demo steps and differs by 1 ulp on the
other 2. All 20 losses and the final weights are identical. The schedule is
now a multiplier on each param group's `initial_lr`. The step counter now
lives in the scheduler's `last_epoch` (20 after the run), which is part of
`state_dict()`. The lambda itself is saved as `None`.

Two pitfalls. `TrainState` builds its optimizer at `lr(0) = 0`, so a
LambdaLR wrapped around `state.optimizer` has `base_lrs = [0.0]`, and 20
steps leave the loss unchanged. And the lesson's set-then-step order,
translated literally, calls the scheduler before the optimizer. PyTorch
warns about that; the first update runs at 2.5e-3 instead of 0, the peak
moves to step 3, and the final loss is 0.1768 instead of 0.1857.
`TrainState` has no `state_dict`, although the doc says the schedule reads
`global_step` "from the trainer's checkpoint".

### 5 — the text plot is the CI default, but it hides the floor and misses 2 of 18 misconfigurations

**Keep the text plot as the CI default and put the PNG behind
`--plot-png`.** The text plot needs no dependency, lands in the log and is
byte-stable. It is 1,007 bytes at 200 steps. The PNG is valid and also
byte-stable, and it is smaller (481 bytes here), but it is opaque in a log
and in a diff. Producing it through matplotlib would add a plotting stack
to every CI image, and matplotlib is not even installed here.

The text plot is weaker than the lesson says. Its y axis always bottoms out
at 0.000000, so lr_min never shows. Each plot was diffed against six
misconfigured schedules on three runs:

| missed by | demo (40x10) | 200 steps | 2,000 steps |
|---|---|---|---|
| text plot | lr_min = 0 vs 1e-4 | none | warmup 19 vs 20 |
| PNG | none | none | none |
| `write_schedule_csv` | none | none | none |

The doc says the plot "catches the misconfigured-schedule class of bugs at
PR time". A 1-step warmup error at 2,000 steps is exactly that class. A CI
gate should diff the CSV; the plot is for people.
