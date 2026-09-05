<!-- generated:start -->
# 03-deep-learning-core / 09-learning-rate-schedules

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/03-deep-learning-core/09-learning-rate-schedules/) · upstream spec
`phases/03-deep-learning-core/09-learning-rate-schedules/docs/en.md`

```bash
uv run demo practice run 09-learning-rate-schedules --ex 1
uv run demo explain 09-learning-rate-schedules --ex 1
uv run pytest demos/phases/03-deep-learning-core/09-learning-rate-schedules
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement exponential decay: lr(t) = lr_0 * gamma^t where gamma = 0.999. Compare to cosine an… | code | T0 | `ex01_exponential_decay.py` |
| 2 | Implement the learning rate range test (Leslie Smith): train for a few hundred steps while ex… | code | T0 | `ex02_lr_range_test.py` |
| 3 | Train with warmup + cosine but vary the warmup length: 0%, 1%, 5%, 10%, 20% of total steps. F… | code | T0 | `ex03_warmup_length.py` |
| 4 | Implement cosine annealing with warm restarts (SGDR): reset the learning rate to lr_max every… | code | T0 | `ex04_warm_restarts.py` |
| 5 | Build a "schedule surgeon" that monitors training loss and automatically switches from warmup… | code | T0 | `ex05_schedule_surgeon.py` |
<!-- generated:end -->

## Answers

All five exercises ask the same kind of question — *which schedule shape is
better?* — and on this lesson's fixture the honest answer is the same every time:
**shape barely decides it, the integral of the schedule does, and a plain constant
rate wins.** Each exercise makes that visible a different way.

**1 — exponential decay vs cosine, and the `t` that decides the answer.**

`lr(t) = lr_0 · γ^t` never says what `t` counts. The lesson's trainer passes a
*step* counter, so the literal reading decays 200× per epoch:

| reading of `t` | end loss (seed 42) | vs cosine | integral of lr |
|---|---:|---:|---:|
| step (literal) | 0.133798 | **15.9× worse** | 50 |
| epoch | 0.005807 | **1.45× better** | 2593 |
| — cosine | 0.008420 | — | 1500 |
| — constant 0.05 | 0.005437 | 1.55× better | 3000 |

The exercise's comparison has **opposite answers under the two readings**, and
nothing in the text picks one. Both hold at all three data seeds.

**FINDING: the literal reading does not converge slowly, it stops.** The epoch
loss is bit-identical from **epoch 168 to epoch 299** — `0.999^(200·168)` puts the
learning rate at `1.26e-16`, so every weight update rounds away and the last 131
epochs change nothing at all.

**MECHANISM.** γ fixes an *absolute* time constant — a half-life of 693 steps,
whatever the run length — where cosine fixes a *relative* one, half-way through
`total_steps`. To reach cosine's `lr_min = 1e-5` over these 60,000 steps you would
need `γ = 0.99985806`. `0.999` is the right constant for a run of a few thousand
steps; this one is sixty thousand.

**CONTROL: it is the integral, not the shape.** A *constant* schedule matched to
cosine's integral (lr = 0.02501) ends at 0.009247 against cosine's 0.008420, and
one matched to expo-per-epoch ends at 0.006063 against 0.005807 — 10% apart at
most, while the 1.73× spread in integral moves the loss by 31%.

**2 — the LR range test returns nothing on the window it is given.**

Run exactly as specified — 1e-7 → 1, 500 steps, 25 bins — the loss falls
**monotonically**, 0.611381 → 0.123033, with **zero** bin-to-bin rises. "The
optimal max LR is just before the loss starts increasing" therefore selects the
top of whatever window you happened to choose.

Widen the identical sweep to 1e+04 and the turning point appears: the minimum is
at **lr = 1.88** and the first rise at **5.19** — both above the stated ceiling
of 1.0.

**MECHANISM: nothing here diverges.** Mean squared error on a sigmoid output is
bounded above by 1 *by construction*, and the worst value measured anywhere in
`lr ∈ [1e-7, 1e+5]` is 0.715. The lesson's own `lr_sensitivity` gates its
`DIVERGED` label on `end > 1.0` — **a condition no run can satisfy** — and prints
`CONVERGED` for `lr = 1.0`, while the lesson text says `lr = 0.1` makes "loss jump
to infinity in 3 steps". What a too-high rate does instead is pin the loss at
**0.285**, the positive-class fraction (57/200), for every bin at `lr >= 1e+01`:
the output saturates to 0 everywhere, which drives `out·(1−out)` to 0, so the
gradient *vanishes* rather than explodes.

**CONTROL: the confound is the method, not the task.** Give each learning rate a
*fresh* network and the curve does turn inside the specified window — minimum
0.059933 at lr = 1.0, rising to 0.137355 and then 0.285002 at the next two
decades. But where it turns depends on the budget:

| budget | best lr |
|---|---:|
| 3 epochs (600 steps, the sweep's order) | 1.0 |
| 100 epochs | 0.1 |
| the lesson's own schedule comparison | 0.05 |

An `lr_max` is only defined together with the number of steps it will be used for.
In the continuous range test the high rates are evaluated on a network the low
rates have already trained, so "loss at lr" is confounded with "loss after k
steps".

**3 — the warmup sweet spot, and the stability it is supposed to trade against.**

| warmup | 0% | 1% | 5% | 10% | 20% |
|---|---:|---:|---:|---:|---:|
| end loss, seed 42 | 0.008420 | 0.008807 | 0.008381 | 0.008381 | **0.008359** |
| end loss, seed 7 | 0.008470 | 0.008478 | 0.008472 | 0.008475 | **0.008467** |
| end loss, seed 99 | 0.015185 | 0.015194 | 0.015141 | 0.015122 | **0.015092** |

20% wins at all three seeds — and 1% is *worst* at all three, which is the shape
you would expect if warmup helped. But the whole grid is within **5.4%**
worst-to-best, and the 0% row is plain cosine: `warmup_steps = 0` matches
`cosine_schedule` to 1.4e-17.

**FINDING: there is nothing to trade against.** Across all 15 grid runs (4,485
epoch transitions) there are **0** epoch-over-epoch loss increases. Every curve is
monotone non-increasing, so "most stable" cannot separate the five candidates.

And there is no spend to trade either: the integral of the schedule is
1500.3 / 1500.3 / 1500.3 / 1500.3 / 1500.2 — a range of **0.006%**. The ramp gives
back about half the peak rate over its `W` steps but compresses the cosine into
`T−W`, and the two nearly cancel.

**FINDING: 20% wins because the grid stops there.** Extend it to 50% and 75% and
the three seeds produce three different winners — seed 42 at 75% (0.008291),
seed 7 at 20% (0.008467), seed 99 at 50% (0.015074) — all within 1% of each other.
The ranking is noise; a "sweet spot" read off the five-point grid is reading the
edge of the grid.

What warmup *does* buy is measurable, and it is a delay — epochs to reach loss
< 0.05 at seed 42, monotone in warmup length at every seed:

| warmup | 0% | 1% | 5% | 10% | 20% |
|---|---:|---:|---:|---:|---:|
| epochs to loss < 0.05 | 18 | 20 | 25 | 33 | 46 |

**MECHANISM: the fixture cannot exhibit the failure warmup prevents.** The lesson
motivates warmup by Adam's zero-initialised moment estimates, but
`train_with_schedule` is plain SGD — its whole update is
`w2[i] -= lr * d_out * h[i]`, and the source mentions no momentum, velocity, beta
or moment term at all. Nothing is carried between steps, so no running statistic
can be wrong at step 0.

**4 — SGDR, and three schedules that disagree about the end of the run.**

Over 600 epochs, SGDR at period 5,000 or 20,000 and plain cosine finish within
**1.6%** of each other at every seed, and the residual **flips sign** with the
data seed (SGDR/cosine = 0.9992, 1.0064, 0.9960). The reason is arithmetic, not
tuning: one cosine cycle integrates to about `0.5·(lr_max + lr_min)·T_i`, and the
cycle lengths sum to the run length whatever the period — so the number of
restarts is *absent from the total*. Measured integrals: 3000.6 (cosine), 3001.2
(period 5,000), 3000.7 (period 20,000).

What the restarts reliably produce is a **loss spike at every one of them** — at
epochs 100, 200, 300, 400, 500 the loss rises 10.5%, 15.9%, 19.2%, 22.8%, 26.1%
over the epoch before, as the rate jumps `1e-5 → 0.05` in a single step.

**FINDING: run `step` past `total_steps` and the lesson's three schedules do three
different things.** Driven to step 119,999 with `total_steps` left at 60,000:

| schedule | lr at step 119,999 | what happens |
|---|---:|---|
| `cosine_schedule` | 1.0e-05 | clamps — it guards on `step >= total_steps` |
| `warmup_cosine_schedule` | 0.048493 | **97% of peak** — no guard, the cosine argument passes 2π |
| `one_cycle_schedule` | **−0.099983** | twice the peak rate, wrong sign, unbounded |

The middle row is a **warm restart nobody asked for, and it helps** — the loss
over the second half goes 0.00838 → 0.00522. The bottom row is gradient ascent:
0.00818 → 0.71499, ending at the negative-class fraction with the network pushed
onto the constant-1 predictor.

**5 — the schedule surgeon, and the question its trigger is answering.**

Both triggers fire. Handoff at epoch 28 / 21 / 19 over seeds (42, 7, 99) — always
at the full rate, since warmup has finished by epoch 12 — and plateau cuts at
epochs 56 / 51 and 59 / 54. The surgeon beats cosine at every seed
(0.026478 vs 0.031536, 0.027366 vs 0.030134, 0.038179 vs 0.041761) and loses to a
plain constant rate at every seed (0.021816, 0.021196, 0.029803).

**FINDING: there is no handoff point to detect.** Sweeping a *fixed* handoff epoch
over 4…52 gives an end loss that is **strictly decreasing at all three seeds**:

| handoff epoch | 4 | 12 | 20 | 28 | 36 | 44 | 52 |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed 42 | 0.0324 | 0.0314 | 0.0288 | 0.0267 | 0.0249 | 0.0232 | **0.0219** |
| seed 7 | 0.0306 | 0.0302 | 0.0280 | 0.0261 | 0.0244 | 0.0230 | **0.0218** |
| seed 99 | 0.0470 | 0.0420 | 0.0377 | 0.0348 | 0.0326 | 0.0307 | **0.0292** |

The best switch point on this task is the latest one tried — "as late as possible"
— while the loss trigger fires 9 epochs apart across seeds on identical code.
*Switch when the loss stabilizes* is answering a question whose answer is *don't*.

And the plateau rule compounds it. Every cut lands in the last third of the run,
at a rate the cosine tail has already annealed to 0.00053 / 0.00409 / 0.00005 /
0.00124 — 1% to 8% of peak. The rule is detecting the schedule it supervises
having stopped, and its answer to that plateau is to lower the rate again.

**CONTROL: the whole controller is its integral.** A constant rate spending the
same total (0.03207 / 0.02906 / 0.02830) ends at 0.028597 / 0.029183 / 0.040622
against the surgeon's 0.026478 / 0.027366 / 0.038179 — within 8% at every seed,
success and failure alike.

The closed loop is only expressible at all because `train_with_schedule` is
bit-deterministic: it calls `random.seed(0)` itself, so replaying a recorded rate
history reproduces `constant_schedule`'s epoch losses bit-for-bit. And
`schedule_fn(step, lr, total_steps)` is never passed the loss it has just
computed, so a loss-driven schedule cannot *be* a `schedule_fn` — the loop has to
be closed from outside.

## A note on the reference code

Three things about `code/main.py`, each of which surfaces in more than one
exercise:

- **`train_with_schedule` is plain SGD with no optimizer state.** That makes it a
  clean fixture for schedule *shape*, and simultaneously makes it unable to
  exhibit the instability that motivates warmup (exercise 3). The lesson explains
  warmup with Adam and then measures it with SGD.
- **Only `cosine_schedule` guards `step >= total_steps`.** `warmup_cosine_schedule`
  and `one_cycle_schedule` do not, and the second returns *negative* learning
  rates past the horizon (exercise 4). Nothing in the trainer clamps them.
- **`lr_sensitivity`'s `DIVERGED` branch is unreachable.** Its condition is
  `end > 1.0` on a loss bounded above by 1 (exercise 2), so the one diagnostic in
  the lesson meant to show a learning rate failing never fires. The failure this
  fixture actually has is saturation, which looks like a *low* loss that stops
  moving.

A fourth point is about the task rather than the code: on the circle dataset a
constant `lr = 0.05` beats every schedule in the lesson, at every seed, in
exercises 1, 3 and 5. Any conclusion here about *shape* is really a conclusion
about how much learning rate the schedule spends.
