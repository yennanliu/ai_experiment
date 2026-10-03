<!-- generated:start -->
# 08-generative-ai / 14-evaluation-fid-clip-score

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/08-generative-ai/14-evaluation-fid-clip-score/) · upstream spec
`phases/08-generative-ai/14-evaluation-fid-clip-score/docs/en.md`

```bash
uv run demo practice run 14-evaluation-fid-clip-score --ex 1
uv run demo explain 14-evaluation-fid-clip-score --ex 1
uv run pytest demos/phases/08-generative-ai/14-evaluation-fid-clip-score
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. Compare FID at N=100 vs N=1000 on the same synthetic distributions.… | code | T0 | `ex01_small_n_fid_is_biased_up_not_down.py` |
| 2 | Medium. Implement CMMD from synthetic CLIP-style features (see Jayasumana et al., 2024 for th… | code | T0 | `ex02_cmmd_at_sigma_10_only_sees_the_mean.py` |
| 3 | Hard. Replicate the HPSv2 setup: take 1000 image-prompt pairs from a subset of Pick-a-Pic, fi… | code | T0 | `ex03_agreement_is_capped_by_the_annotators.py` |
<!-- generated:end -->

## Answers

The lesson is 146 lines of stdlib: FID through a Denman-Beavers matrix
square root, a cosine `clip_like`, and an Elo update, all on 4-D synthetic
features. Exercises 1 and 2 are **T0** against that code; exercise 3 names
Pick-a-Pic and a CLIP backbone, neither of which is here, so it ships the
scaled-down runnable `DESIGN D11` requires.

### 1 — small-N FID is biased up, not down

Both pools come from the lesson's own `make_features(0.0, n, 4, rng)`, so the
true FID is 0 and every unit of the lesson's `fid` is bias. Averaged over 30 seeds:

| N | 100 | 1000 |
|---|---:|---:|
| null FID | 0.0268 | 0.00287 |
| N x FID | 2.68 | 2.87 |
| predicted `s²(2d + d(d+1)/2)/N` | 0.0288 | 0.00288 |

**ANSWER: about 0.027 at N=100 and 0.0029 at N=1000**, a ratio of 9.3 — a `1/N`
law, which N=1000 shrinks but never removes.

**FINDING: the bias is upward, and the lesson's prose says the opposite.**
`docs/en.md` says small N "under-estimates covariance, gives falsely low FID".
All **60 of 60** null draws score above zero (and the lesson's own `main()`
prints "biased up"). The covariance is not under-estimated either: `covariance`
divides by `n − 1`, and its trace averages **0.638** against the true **0.64**.

**FINDING: the bias is a closed form.** The mean term gives `2ds²/N` (0.0128
predicted, 0.0122 measured) and the square-root term `s²d(d+1)/2N`; at d=8,
N=400 the prediction is 0.0208 against a measured 0.0223. The covariance term is
`(d+1)/4` times the mean term, **512x** at Inception's d=2048 — that is what
the "N ≥ 10,000" rule is really paying for.

**CONTROL:** `jacobi_sqrt` (Denman-Beavers, despite the name) squares back to
`cov_r @ cov_g` within **6.9e-18**.

### 2 — CMMD at σ=10 only sees the mean

CMMD is `1000 x` the unbiased MMD² under an RBF kernel of bandwidth 10, on
unit-norm 8-D features, with its distance from the lesson's own `clip_like`.
Sensitivity is a z-score against a null, 12 seeds of N=100:

| degradation | FID z | CMMD z (σ=10) | CMMD z (σ=0.5) |
|---|---:|---:|---:|
| mean tilted 0.25 rad | 4.9 | **7.4** | — |
| variance squeezed from 7 directions into 2 | **64.0** | 0.79 | 29.8 |

**ANSWER: CMMD is ahead on a shifted mean and ~80x behind on a reshaped
cloud.**

**FINDING: at σ=10 CMMD is a linear-kernel MMD.** Unit vectors are at most 2
apart, so `exp(−r²/200)` never leaves `[0.98, 1]` and CMMD equals
`10 x |μ_r − μ_g|²` (unbiased) within **0.0039** on every seed. The doc's "no
Gaussian assumption" holds; "better at detecting subtle quality differences"
does not, at this bandwidth, for anything that leaves the mean alone.

**FINDING: the bandwidth is the blind spot, not MMD** — σ=0.5 recovers z=29.8.

**CONTROL:** on identical pools CMMD averages **−0.0070**, negative on 7 of 12
seeds; FID is at least 0.0154 on every seed.

### 3 — held-out agreement is capped by the annotators

`torch`, `transformers`, `open_clip` and `datasets` are absent. What runs:
1000 training and 1000 held-out prompt/image-pair items, labelled by
Bradley-Terry (sharpness 8) on a hidden map, and an HPSv2-shaped scorer
`clip_like(W x, t)` with a logit scale, fine-tuned from `W = I`.

| scorer | held-out agreement | vs noise-free order | log loss |
|---|---:|---:|---:|
| zero-shot (`W = I`) | 72.3% | — | — |
| fine-tuned, scale learned | **85.2%** | 93.8% | 0.352 |
| fine-tuned, scale fixed at 1 | 84.2% | — | 0.556 |
| the true utility | 85.8% | 100% | — |

**ANSWER: 85.2%, up from 72.3%.**

**FINDING: that is 99% of the ceiling.** The labels themselves cap agreement at
**86.1%** expected; against the noise-free ordering the same scorer is right on
**93.8%** of pairs. An agreement number without the inter-annotator ceiling
beside it cannot say how much better a scorer could get.

**FINDING: `clip_like` is bounded in [−1, 1], so it needs a logit scale.**
Fixed at 1, the most confident preference it can state is `sigmoid(2) = 0.88`;
agreement barely moves but log loss is 0.556 against 0.352. The learned scale
lands at **8.34**, recovering the annotators' sharpness of 8 — which is why CLIP
and HPSv2 carry one.
