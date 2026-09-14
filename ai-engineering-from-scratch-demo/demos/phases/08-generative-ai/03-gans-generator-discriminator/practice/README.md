<!-- generated:start -->
# 08-generative-ai / 03-gans-generator-discriminator

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/08-generative-ai/03-gans-generator-discriminator/) · upstream spec
`phases/08-generative-ai/03-gans-generator-discriminator/docs/en.md`

```bash
uv run demo practice run 03-gans-generator-discriminator --ex 1
uv run demo explain 03-gans-generator-discriminator --ex 1
uv run pytest demos/phases/08-generative-ai/03-gans-generator-discriminator
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py` with the stock settings. Then set `D_LR = 5 * G_LR` and rerun. How f… | code | T0 | `ex01_the_non_saturating_loss_does_not_saturate.py` |
| 2 | Medium. Replace the Goodfellow BCE loss with the WGAN loss: `loss_D = E[D(fake)] - E[D(real)]… | code | T0 | `ex02_the_prescribed_clip_switches_the_critic_off.py` |
| 3 | Hard. Extend the 1-D example to 2-D data (mixture of 8 Gaussians on a ring). Track how many o… | code | T0 | `ex03_there_is_no_collapse_left_to_fix.py` |
<!-- generated:end -->

## Answers

The lesson is a 190-line pure-Python GAN on a 1-D two-mode mixture at ±2. All
three exercises are **T0** — pure stdlib, as the lesson is — and all three turn
out to be about the gap between what the exercise expects to see and what this
code can produce.

A note that decides two of them: `update_d` and `update_g` hard-code the BCE
gradient `p - target`, so any exercise that *replaces the loss* needs a new
gradient. Where that happens the new gradient is checked against a central
difference of the lesson's own `forward_d` (worst gap **7.6e-12**), so the model,
the data and the learning rate stay the lesson's and only the loss is new.

### 1 — the non-saturating loss does not saturate

| d_lr | G loss, last 100 steps | within-run sd | D(real) | D(fake) |
|---|---:|---:|---:|---:|
| 0.5× g_lr (stock) | 0.791 | 0.025 | 0.545 | 0.451 |
| 5× g_lr | 0.787 | 0.029 | 0.549 | 0.457 |

**ANSWER: it does not collapse.** The means differ by **0.004** while the same
quantity spans **0.425** across the six seeds of a single setting — **115×** the
gap. Three seeds are not enough to see this: on seeds 0–2 alone the two settings
differ by 0.085 and look separated, which is why the file runs six.

**FINDING: the exercise asks how fast the non-saturating loss saturates.**
`update_g`'s own docstring reads "Non-saturating G loss: maximize log D(G(z))".
The form that goes flat when D wins is the *saturating* `log(1 − D(G(z)))`, whose
gradient vanishes as `D(G(z)) → 0`. Goodfellow 2014 introduced the form this
lesson implements precisely to remove that failure.

**FINDING: D never wins, at any ratio tried.** From 0.5× to **50×**, `D(fake)`
stays between 0.43 and 0.46 and `D(real)` between 0.54 and 0.60 — a critic at
chance, which is equilibrium rather than dominance. What 50× buys is *variance*:
the within-run tail deviation rises to **0.136** against **0.025**. Over-training
D makes this run jumpier, not deader.

**CONTROL: `D_LR` and `G_LR` do not exist.** They are locals `d_lr` and `g_lr`
inside `main()`, so the edit the exercise describes cannot be made as written.

### 2 — the prescribed clip switches the critic off

Both arms are scored by the **exact 1-D Wasserstein distance** — what WGAN claims
to minimise, and a yardstick neither loss can game. Two anchors make it readable:
two fresh real draws score **0.129**, and the same generators with training
switched off score **1.730**.

| arm | W1 | per-seed spread | mean \|critic score\| |
|---|---:|---:|---:|
| BCE (the lesson's) | 1.470 | 0.973 | 0.398 |
| WGAN, clip 0.01 | 1.715 | **0.204** | **0.0003** |
| WGAN, clip 0.1 | 1.421 | 0.410 | — |
| WGAN, clip 0.5 | **0.937** | **0.158** | — |
| WGAN, clip 2.0 | 3.443 | 5.141 | — |

**ANSWER: more stable — because the critic is switched off.** At the prescribed
clip WGAN is four times steadier than BCE and lands on the *untrained* 1.730,
inside its own seed spread. Clipping every weight to `[-0.01, 0.01]` through a
1→16→1 net bounds the score to about ±0.013, and mean `|D(x)|` measures
**0.0003** against BCE's 0.398 — **1,394×** smaller. A generator learns from the
critic's slope, and there is none.

**FINDING: raise the clip and WGAN wins outright.** At **0.5** it reaches
**0.937** where BCE reaches 1.470, and is **6.2×** steadier. At 2.0 it diverges.
Across 0.01 → 2.0 the clip is not a detail of the method; it is the entire
experiment.

**FINDING: wall-clock decides nothing.** One WGAN run costs **1.0×** a BCE run.
The exercise never mentions `n_critic`, and the standard recipe's five critic
steps per generator step is the term that would actually make WGAN slower.

### 3 — there is no collapse left to fix

Ring of 8 Gaussians, radius 2, σ = 0.05. A mode counts as captured when it draws
at least a quarter of its fair share of 400 probe samples — a rule under which
the real data itself scores 8/8.

| step | plain GAN | + minibatch discrimination |
|---|---|---|
| 1,000 | 4, 6, 8 | 5, 7, 8 |
| 5,000 | **8, 8, 8** | **8, 8, 8** |
| 10,000 | **8, 8, 8** | **8, 8, 8** |

**ANSWER: 8 of 8 from step 5,000 on, with or without the fix.** Minibatch
discrimination fixes mode collapse, and on this architecture, this ring and this
budget there is none to fix — so the re-measurement the exercise asks for returns
the same number twice.

**FINDING: what the fix buys is a head start, not a better endpoint — probably.**
At step 1,000 it is ahead or level on every seed, **+0.67** of a mode on average.
The direction is consistent; the size is not established, because the two arms
are **not paired** — a critic one column wider consumes different draws at
initialisation and every sample after that differs. By 5,000 the gap is zero.

**FINDING: "extend to 2-D" is one function.** `init_mlp`, `forward_g`,
`forward_d`, `update_d` and `update_g` are all written over `len(x)` and run
unmodified at any width — the plain arm above trains through the lesson's
untouched `update_d` and `update_g` in 2-D. Only `sample_real` is hard-wired to
one dimension, so the "hard" half of this exercise is replacing a nine-line
sampler.

**Cost note.** This exercise trains 6 GANs for 10,000 steps each in pure Python
and takes about **2¼ minutes**. The step counts are the ones the exercise names,
and shortening them to save time would answer a different question.
