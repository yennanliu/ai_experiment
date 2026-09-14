<!-- generated:start -->
# 08-generative-ai / 05-stylegan

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/08-generative-ai/05-stylegan/) · upstream spec
`phases/08-generative-ai/05-stylegan/docs/en.md`

```bash
uv run demo practice run 05-stylegan --ex 1
uv run demo explain 05-stylegan --ex 1
uv run pytest demos/phases/08-generative-ai/05-stylegan
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py` with `adain_on=True` and `adain_on=False`. Compare the spread of out… | code | T0 | `ex01_with_adain_off_the_generator_ignores_z.py` |
| 2 | Medium. Implement mixing regularization: for a training batch, compute `w_a`, `w_b`, and appl… | code | T0 | `ex02_the_last_block_overwrites_mean_and_std.py` |
| 3 | Hard. Take a pretrained StyleGAN3 FFHQ model (ffhq-1024.pkl). Find the `w` direction that con… | code | T1 | `ex03_the_direction_is_already_in_the_weights.py` |
<!-- generated:end -->

## Answers

The lesson is a 129-line StyleGAN sketch: a mapping network `z → w`, a constant
input, three synthesis blocks, and AdaIN carrying the style. Nothing is trained —
there is no loss, no gradient and no optimiser anywhere in the module — so all
three exercises are really about what the *architecture* determines before any
learning happens. Exercises 1 and 2 are **T0**; exercise 3 is **T1** because the
pretrained checkpoint it names is unbuildable and `sklearn` stands in.

One line decides most of what follows. `adain(features, scale, bias)` returns
`scale * (f − mean) / sd + bias`, and it is the **last** operation of the **last**
block. So the output's mean *is* `bias` and its spread *is* `|scale|`, both linear
forms in `w`, and nothing earlier in the network can affect either.

### 1 — with AdaIN off the generator ignores z

| | per-channel σ across 200 latents | distinct outputs | what `main()` prints |
|---|---|---:|---:|
| `adain_on=True` | up to 0.266 | 200 | 0.1656 |
| `adain_on=False` | **all 0.0** | **1** | 0.0122 |

**ANSWER: the spread is exactly zero.** `w` enters `stylegan_forward` only inside
the `if adain_on:` branch, so with the branch off the function never reads its
first argument.

**FINDING: the lesson reports a non-zero spread for that constant.** `main()`
flattens the outputs into one list and takes `mean_std` of the pool, mixing
variation across latents with variation across channels. The **0.0122** is the
shape of one frozen vector — unchanged whether the sweep runs one latent or a
million.

**FINDING: a perturbed latent is the same measurement** — movement **0.0** off,
0.0068 on. There is no fixed-versus-perturbed contrast to draw when there is no
input.

**CONTROL:** with AdaIN off, the output for a mapped latent equals the output for
a zero `w`, **bit for bit**.

### 2 — the last block overwrites mean and spread

| varying | σ of output means | σ of output spreads | distinct shapes |
|---|---:|---:|---:|
| `w_b` (last block) | 0.049 | 0.092 | **2** |
| `w_a` (first two) | **0.000000** | **0.000023** | **40** |

**ANSWER: perfectly disentangled — and nothing learned it.** `w_a` owns the
shape and cannot touch the mean or the spread; `w_b` owns the mean and the spread.

**FINDING: `adain` is the whole mechanism, as an identity.**
`mean(output) − ⟨bias2, w⟩` measures **2e-17**; `sd(output) − |⟨scale2, w⟩|`
measures **9e-08**, which is AdaIN's own epsilon.

**FINDING: there is no decoder that learns.** The module exposes none of
`backward`, `update`, `loss`, `apply_grads`. The question has no referent — the
split is present at initialisation.

**CONTROL:** driving both halves with the same `w` reproduces
`stylegan_forward` bit for bit, so this is the reference with different routing.

### 3 — the direction is already in the weights

`torch`, `dnnlib` and `legacy` — the three imports `ffhq-1024.pkl` needs to
unpickle — are all absent, and the checkpoint is ~300 MB this repo does not ship.
So the *method* runs at the scale that is available (`DESIGN D11`): the lesson's
own networks, output brightness standing in for "smile", and a real `LinearSVC`
on 600 labelled `w`.

| α | 0 | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| attribute | −0.07 | +0.95 | +1.97 | +4.01 | +8.08 | **+16.24** |
| output spread | 0.22 | 0.29 | — | — | — | **4.44** |
| shape drift | 0 | **1.58** | — | — | — | 1.38 |

**ANSWER: the SVM finds it at accuracy 0.998, and it never stops working** —
**+1.019** per unit of α, linearly, forever.

**FINDING: the direction did not have to be learned.** The attribute is an exact
linear form in `w` whose normal is `synth["bias2"]`, a vector already sitting in
the weight dict. The fitted direction aligns with it at **cos = 0.995**. The
labelled samples and the margin rediscover a row of the weights.

**FINDING: "how far before identity drifts" has no threshold.** The shape drifts
**1.58** within the first unit of α and then *stops* (1.38 at α=16) as the earlier
blocks saturate, while the spread grows without bound. Identity drifts
immediately and then cannot drift further — the regime the question presumes,
where the attribute moves and everything else holds, is empty on this generator.

**CONTROL:** the fitted direction reaches +16.24 where a random unit direction of
the same length reaches +5.61.
