<!-- generated:start -->
# 03-deep-learning-core / 12-intro-to-jax

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/03-deep-learning-core/12-intro-to-jax/) · upstream spec
`phases/03-deep-learning-core/12-intro-to-jax/docs/en.md`

```bash
uv run demo practice run 12-intro-to-jax --ex 1
uv run demo explain 12-intro-to-jax --ex 1
uv run pytest demos/phases/03-deep-learning-core/12-intro-to-jax
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add dropout to the MLP. In JAX, dropout requires a PRNG key -- thread a key through the forwa… | code | T0 | `ex01_dropout_keys.py` |
| 2 | Use `jax.vmap` to compute per-example gradients for a batch of 32 MNIST images. Compute the g… | code | T0 | `ex02_per_example_grads.py` |
| 3 | Replace the manual forward function with a generic `mlp_forward(params, x)` that works for an… | code | T0 | `ex03_generic_forward.py` |
| 4 | Benchmark the training step with and without `@jax.jit`. Time 100 steps of each. How large is… | code | T0 | `ex04_jit_benchmark.py` |
| 5 | Implement gradient clipping by composing `optax.chain(optax.clip_by_global_norm(1.0), optax.a… | code | T0 | `ex05_clip_by_global_norm.py` |
<!-- generated:end -->

## Answers

Every fixture here is a **seeded synthetic stand-in for MNIST** — the lesson's
`get_mnist_data` downloads from OpenML, and these solutions run offline and
bit-reproducibly. Shapes and label noise are MNIST's; the pixels are not.

The thread across the five: **JAX's transforms do exactly what they say, and the
lesson's claims about what they buy are the part that does not survive
measurement.** `jit` is 7.2× faster per step but 220 steps behind after its first
call; `vmap` is 24× faster than an eager loop and only 1.5× faster than a compiled
one; `clip_by_global_norm` in front of Adam is very nearly a no-op; and dropout
costs held-out accuracy on this fixture rather than buying it.

**1 — dropout costs accuracy here, and the eval path is the bigger hazard.**

600 steps each:

| p | train | test |
|---:|---:|---:|
| 0.0 | 0.9536 | **0.3018** |
| 0.2 | 0.9502 | 0.2988 |
| 0.5 | 0.3354 | 0.1475 |

p = 0.2 moves held-out accuracy by **−0.29 points**; p = 0.5 costs 15.4, because
it stops the net fitting at all. The train/test gap dropout is meant to close is
0.6519 at p = 0 and 0.6514 at p = 0.2 — unmoved. The gap is label noise the net
memorises, and dropout does not touch it.

**FINDING: leaving dropout on at eval costs 6.6 points — 20× the effect the
exercise asks about.** The same p = 0.2 parameters score 0.2988 with dropout off
and 0.2324 with it on. This is the JAX-specific trap: the key is a *required*
argument, so the eval path has to pass one, and passing one keeps dropout live.
There is no `model.eval()` to forget — there is a parameter you cannot omit.

**MECHANISM: one key reused is not two masks — the second is a prefix of the
first.** `bernoulli(k, .8, (256,))[:128]` and `bernoulli(k, .8, (128,))` agree on
**100%** of positions. The counter-based PRNG indexes by position, so asking for a
different shape does not decorrelate anything. That is precisely why the exercise
says *split it for each dropout layer*.

**MECHANISM: `split` is a pure function of its key.** Splitting `PRNGKey(1234)`
twice gives bit-identical subkeys; the two siblings of one split differ;
`PRNGKey(1235)` differs again. Reproducibility is a property of the value passed
in, not of when anything last called `seed()` — no interleaved draw shifts it.

**CONTROL.** Inverted dropout preserves the mean: 20,000 ones at p = 0.5 return
0.99840, against 0.49920 without the `1/(1-p)` rescale. At p = 0.0 this forward is
bit-identical to the lesson's own.

**2 — the examples with the largest gradients are the ones the model gets wrong.**

After 200 steps, mean ‖g‖ over the 32-example batch:

| | mean ‖g‖ | n |
|---|---:|---:|
| misclassified | **32.86** | 8 |
| correct | 16.46 | 24 |

**2.00×.** The three largest gradients sit at p(true class) 0.189, 0.118, 0.283;
the three smallest at 0.927, 0.920, 0.891.

**MECHANISM — and it is exact.** The per-example gradient w.r.t. `layer3.b`
matches `softmax(z) − onehot(y)` elementwise to **5.96e-08**. Every other gradient
in the tree is that vector back-propagated, so ‖gᵢ‖ is a monotone read-out of how
wrong the model is about example *i* (correlation with the cross-entropy loss
0.906).

**FINDING: at initialisation the question has almost no answer.** Untrained, the
32 norms span only **1.56×** and correlate 0.636 with ‖p − y‖; after 200 steps
they span **12.23×** at 0.953. A near-uniform softmax makes `p − y` nearly the
same vector for every example. *Which examples dominate* is a property the model
acquires, not one the data has — so asking it of a fresh net gets noise.

**FINDING: vmap's 24× is mostly dispatch, not vectorisation.** Best of 5:

| | time | vs vmap |
|---|---:|---:|
| `vmap` + `jit` | **2.84 ms** | — |
| eager Python loop | 68.8 ms | 24.2× |
| loop over a *jitted* per-example grad | 4.28 ms | **1.5×** |

The lesson's "10-100x faster than a Python loop" holds against the eager loop. Put
`jit` on the per-example function and the margin collapses to 1.5×: most of what
`vmap` removes is the interpreter, not the arithmetic.

**CONTROL: the 32 rows average to the batch gradient** to **8.94e-08**. `loss_fn`
ends in `jnp.mean`, so the grad of the mean is the mean of the grads — which means
the ordinary batch gradient throws away exactly the spread the first check
measures.

**3 — one `mlp_forward` for any depth, and two ways the pytree misleads you.**

The replacement is bit-identical to the lesson's `forward` on the lesson's own
3-layer params — *equal*, not close — and so is its gradient (max |Δ| **0.0e+00**).
Depths 1, 2, 4 and 11 all return `(64, 10)`.

`len(jax.tree.leaves(params)) // 2` recovers the depth correctly at every one of
those: 1, 2, 4, 11.

**MECHANISM, with a trap.** The flattening is by *sorted key*, so within a layer
it yields `layer1.b` before `layer1.w` — **bias first**. Pairing leaves as `(w, b)`
in flatten order silently transposes every layer.

**FINDING: past nine layers the key order stops meaning depth order.**
`sorted(params)` on an 11-layer net gives `layer1, layer10, layer11, layer2, …` —
`layer10` sorts before `layer2` as a string. `jax.tree.leaves` still counts 11
layers, so *the depth is right and the order is wrong*, and the same forward
driven by `sorted` fails on shapes:

```
dot_general requires contracting dimensions to have the same shape, got (10,) and (24,)
```

Any scheme that reads structure out of dict keys inherits this. Sorting by
`(len(name), name)` fixes it.

**CONTROL: what the replacement is actually worth.** The lesson's own `forward` is
wired to exactly three layers, and only one of its two failures is loud:

| net | lesson's `forward` | `mlp_forward` |
|---|---|---|
| 1 layer | raises `KeyError: 'layer2'` | `shape (64, 10)` |
| 11 layers | **`shape (64, 24)`, raises nothing** | `shape (64, 10)` |

On the deep net it stops after `layer3` and the remaining eight layers are never
applied — the output is a hidden activation wearing the shape of a prediction.
That is the silent one, and it is the reason to write the generic version.

**4 — `jit` is 7.2× per step and 220 steps behind after the first call.**

100 steps at batch 128: **0.500 ms/step** jitted against **3.595 ms**
undecorated. The first call takes **110 ms** — 220 compiled steps' worth — so jit
goes net ahead only at **step 36** of the 100.

**MECHANISM: the compile is one-off.** The cache holds 0 entries before the first
call and 1 after, and still 1 once several hundred more calls at that shape have
run.

**FINDING: a different batch size is a different program.** One call at batch 64
takes the cache from 1 to 2 and costs 109 ms — another 217 steps' worth. The
lesson's training loop dodges this only by dropping the ragged tail:
`len(X_train) // batch_size` discards **96 of 60,000** rows an epoch.

**FINDING: the speedup is Python dispatch, and it collapses with batch size.**

| batch | speedup |
|---:|---:|
| 16 | **12.3×** |
| 2048 (128× the arithmetic) | **2.1×** |

The jitted step slows 12.6× over that range; the eager step only 2.1×. Eager time
is Python dispatch of a fixed number of primitives, so what `jit` removes is
mostly the interpreter — and the bigger the batch, the less of the wall clock that
is.

**CONTROL: `jit` changes the clock and nothing else** — 100 identical steps down
each path from one init leave the parameters agreeing to **2.5e-07**. Not free a
priori: XLA fuses and reorders float32 arithmetic.

**5 — `chain(clip_by_global_norm(1.0), adam(1e-3))` very nearly undoes its own clip.**

300 steps, seeds (0, 1, 2), final loss / held-out accuracy:

| seed | without clip | with clip |
|---:|---|---|
| 0 | 0.535 / 0.2529 | 0.601 / **0.2900** |
| 1 | 0.408 / 0.2764 | 0.621 / **0.2969** |
| 2 | 0.517 / 0.2627 | 0.914 / **0.2734** |

Mean **+2.28** accuracy points — against a 2.34-point spread across the *unclipped*
seeds alone. It regularises; it does not stabilise.

**FINDING: plotting the gradient norm shows the clip doing nothing.** The norms a
training loop can record are the *raw* ones, and in the clipped run they still
cross 1.0 on **298 of 300** steps (2.55 at step 0, 1.70 at 10, 3.71 at 150, 3.12 at
299). The clip lives between `grad` and `apply_updates` and never enters that
curve — so the plot the exercise asks for cannot show the effect it asks about.

**MECHANISM: a clip is a pure rescale, and Adam is scale-invariant.** ‖g‖ is 2.553
at init and 6.177 after 300 Adam steps; both clip to norm 1.000000 at cosine
1.000000 to the raw gradient — direction untouched. That costs SGD exactly
`1/‖g‖` of its step (0.3917, 0.1619) and costs Adam **0.01%** and **23%**. At
`t = 1` Adam's update is `g/(|g|+ε)` elementwise; later, `√v̂` still carries most of
the same factor.

**CONTROL: the same chain in front of `optax.sgd` is decisive.** `sgd(0.5)` peaks
at ‖g‖ 125 and ends at loss 2.300, 2.298, 2.297 — chance. `chain(clip, sgd)` peaks
at 7.0 and ends at 1.845, 1.903, 1.776. The clip is no no-op; **Adam is what makes
it one.**
