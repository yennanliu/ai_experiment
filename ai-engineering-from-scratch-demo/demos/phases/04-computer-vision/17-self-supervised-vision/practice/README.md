<!-- generated:start -->
# 04-computer-vision / 17-self-supervised-vision

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/17-self-supervised-vision/) · upstream spec
`phases/04-computer-vision/17-self-supervised-vision/docs/en.md`

```bash
uv run demo practice run 17-self-supervised-vision --ex 1
uv run demo explain 17-self-supervised-vision --ex 1
uv run pytest demos/phases/04-computer-vision/17-self-supervised-vision
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Verify that InfoNCE loss drops when you decrease temperature for well-aligned embeddin… | code | T1 | `ex01_infonce_temperature_sweep.py` |
| 2 | (Medium) Implement a DINO-style centre buffer. Show that without the centring, the student co… | code | T1 | `ex02_dino_centring_prevents_collapse.py` |
| 3 | (Hard) Train MAE on CIFAR-100 using the TinyUNet from Lesson 10 as the backbone. Report linea… | code | T1 | `ex03_mae_probe_versus_pretext_floor.py` |
<!-- generated:end -->

## Answers

The lesson's code is correct — `info_nce`, `random_mask_indices` and `DinoHead`
all do what they claim. What the three exercises find is that **two of the three
metrics they ask you to report cannot see the thing the lesson is about**.
InfoNCE reads exactly `log(2N-1)` for a fully collapsed encoder at every
temperature, and the MAE linear probe reads 1.000 at 10, 50 and 200 epochs. In
both cases the file says so and measures something that discriminates instead.

### 1 — both halves of the claim hold, and "well-aligned" is doing the work

The sweep the exercise asks for, on the lesson's own `info_nce` at `N=16`,
`D=32` (seed 0; matplotlib is not in the `vision` deps group, so it ships as a
table rather than a plot):

| tau | 0.05 | 0.1 | 0.2 | 0.5 |
|---|---:|---:|---:|---:|
| identical views | **0.0000** | 0.0056 | 0.2477 | 1.6549 |
| independent views | **7.4584** | 4.6894 | 3.7623 | 3.4793 |

**ANSWER: both directions confirmed, in 4/4 seeds.** Aligned pairs fall
monotonically as tau falls in all four seeds; independent pairs rise in all four
(monotone-down in 0/4). Across seeds the tau=0.05 aligned loss spans
**0.00001–0.00002** and the random one **7.458–8.402**. Low temperature widens
the gap between a good and a useless encoder by **4.1x** — 1.824 at tau=0.5
against 7.458 at tau=0.05.

**FINDING: the first half reverses once the positive stops winning its row.**
Replacing the identical second view with `normalize(z1 + eps*noise)`:

| eps | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 |
|---|---:|---:|---:|---:|---:|---:|
| mean positive cosine | 0.863 | 0.640 | 0.477 | 0.370 | 0.297 | 0.246 |
| seeds still monotone | 4/4 | 4/4 | **4/4** | 2/4 | **0/4** | **0/4** |

`eps<=0.3` keeps the claim in 4/4 seeds and `eps>=0.5` breaks it in 4/4, with
eps=0.4 splitting 2/4. So the monotonicity is a property of **how aligned the
views are**, not of InfoNCE: `tau -> 0` approaches a hard max, which only helps a
positive that already wins.

**MECHANISM: a collapsed encoder reads exactly `log(2N-1)` at every tau.** One
vector repeated 16 times as both views gives **3.4340** at all four temperatures
— spread **exactly 0.0** — against `log(31) = 3.4339872`. When every similarity
is equal, `masked_fill` leaves 31 tied logits per row and tau cancels out of the
softmax entirely. The number this exercise sweeps is **blind to representation
collapse**, which is exactly why DINO needs the centring of exercise 2.

**FINDING: `log(2N-1)` is the `tau -> infinity` limit, not the random-pair
loss.** Reproducing `main.py`'s own draw gives **0.006** for identical views and
**5.229** for random ones — matching what it prints, beside its comment "should
be near log(2N-1) = 3.434", which the random loss exceeds by **1.52x**. Raising
tau locates the real limit:

| tau | 1.0 | 5.0 | 20.0 | 100.0 |
|---|---:|---:|---:|---:|
| loss | 3.4973 | 3.4444 | 3.4365 | 3.4345 |
| \|loss − log(31)\| | 0.0633 | 0.0104 | 0.0025 | **0.0005** |

### 2 — collapse is real, "within a few epochs" is not, and the obvious metric cannot see it

**FINDING: the metric one reaches for first is useless here.** Across four runs
that are *all* fully collapsed, the peak teacher column mean spans
**0.51–1.00** — collapse picks an arbitrary output direction, so no single
threshold on it separates collapse from health. What discriminates is the
**effective rank** of the student-output covariance, `(Σe)² / Σe²`.

| teacher temp | centring | final effective rank (4 seeds) | argmax modes of 16 | collapsed |
|---|---|---:|---:|---:|
| 0.04 | none | **1.00–1.01** | 1–2 | 4/4 |
| 0.04 | logits (`main.py`) | **1.96–3.16** | 5–8 | 0/4 |
| 0.04 | probabilities | 2.15–4.62 | — | 0/4 |
| 0.1 | none | 1.02–1.98 | — | 1/4 |
| 0.2 | none | **10.27–10.85** | 16 | 0/4 |

**ANSWER: without centring the student collapses in 4/4 seeds**, to effective
rank 1.00–1.01 using 1–2 of 16 argmax modes; the centre stays **exactly 0.0**, so
`teacher()` degenerates to plain softmax sharpening. Calling `update_centre` as
`main.py` does holds the rank at 1.96–3.16 in 4/4 — a gap of **0.96 with no seed
overlap** — and keeps 5–8 modes alive.

**FINDING: "within a few epochs" does not survive reseeding.** The first epoch
below rank 1.05, per seed: **1, 3, 3 and 25**. Three seeds collapse by epoch 3
but the slowest needs 25, so the phrase is right for the median run and wrong by
an order of magnitude for the tail. All four are collapsed by epoch 40.

**MECHANISM: this is a temperature-ratio effect, not distillation itself.**
`DinoHead.teacher` defaults to `temp=0.04` against the student's 0.1. With
centring off and the teacher raised to **0.2**, there is no collapse at all —
rank 10.27–10.85, all 16 modes in use, no buffer. At teacher temp 0.1 it is
degenerate again (1.02–1.98). **Sharpening is what drives the collapse**, and the
centre is what makes a sharp teacher usable.

**CONTROL: the naming mismatch in `update_centre` is latent, not fatal.**
`register_buffer` puts `centre` in `state_dict()` with `requires_grad False`, so
all **1040** trainable head parameters are `proj`'s. Its parameter is named
`teacher_out` (probabilities) while `main.py` passes logits — feeding it
probabilities instead **also** prevents collapse, because `teacher()` divides by
0.04 and so scales any centre by **25x**; the smaller buffer (|centre|max
0.22–0.46 against 0.33–0.88) is still large enough.

### 3 — MAE beats the from-scratch probe overwhelmingly, and the ledger the exercise wants is saturated

Three premises do not survive contact: CIFAR-100 is **not downloaded** (the
1,000-image subset is synthetic 16×16 oriented gratings over 10 classes, not
100); lesson 10's `TinyUNet` is a **diffusion** net whose `forward(x, t)` adds a
timestep embedding to the bottleneck, so `t` is pinned to 0 and the encoder path
is run by hand; and `base=8` on 16×16 inputs (**10,763** parameters) keeps 200
epochs inside the T1 budget. Masking is the lesson's own
`random_mask_indices(64, 0.75)` — 16 patches visible of 64, one mask per batch.

**ANSWER: the MAE-pretrained probe wins by 0.90 absolute.**

| linear probe on 500 held-out images | test | train |
|---|---:|---:|
| MAE-pretrained frozen bottleneck (256-d) | **1.000** | — |
| from scratch on the same 1,000 images' raw pixels | **0.100** | 1.000 |
| frozen *untrained* `TinyUNet` | 0.120 | — |
| chance | 0.100 | — |

The raw-pixel head reaches **1.000 on its own training split** and 0.100 on
held-out data — it memorises rather than generalises, because random phase makes
the classes linearly inseparable in pixel space. The untrained backbone at 0.120
is barely above chance, so the gain is **pretraining**, not the architecture's
random features. `features` runs under `torch.no_grad()` on `net.eval()`, so all
10,763 backbone parameters are frozen and only the logistic head is fitted.

**FINDING: the 10/50/200 ledger the exercise asks for ranks nothing.** The three
numbers it wants reported are one number:

| epoch | 1 | 2 | 3 | 5 | **10** | **50** | **200** |
|---|---:|---:|---:|---:|---:|---:|---:|
| probe accuracy | 0.612 | 0.994 | 1.000 | 0.998 | **1.000** | **1.000** | **1.000** |

The probe is pinned at its ceiling, so it cannot rank 50 against 200 or see
anything in between, and the ceiling arrives **long before the first checkpoint
the exercise names**. Quoting it as a headline would claim a measurement that was
not made.

**MECHANISM: the pretext loss still discriminates, measured against its own
floor.** The masked-patch MSE cannot fall below the injected noise variance,
`0.5² = 0.250`, because 75% of patches are hidden and their noise is
unpredictable from the 16 visible ones.

| | trivial (predict zeros) | ep 10 | ep 50 | ep 200 | noise floor |
|---|---:|---:|---:|---:|---:|
| masked-patch MSE | 0.7496 | 0.4744 | 0.3522 | **0.3148** | 0.250 |

Still falling at 200 epochs, having closed **87.0%** of the reducible gap and
sitting at **1.26x** the floor. This is the metric that separates 50 epochs from
200 once the probe has saturated.
