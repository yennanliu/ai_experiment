<!-- generated:start -->
# 01-math-foundations / 11-singular-value-decomposition

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/01-math-foundations/11-singular-value-decomposition/) · upstream spec
`phases/01-math-foundations/11-singular-value-decomposition/docs/en.md`

```bash
uv run demo practice run 11-singular-value-decomposition --ex 1
uv run demo explain 11-singular-value-decomposition --ex 1
uv run pytest demos/phases/01-math-foundations/11-singular-value-decomposition
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement the full SVD from scratch without using power iteration. Instead, compute the eigen… | code | T0 | `ex01_svd_via_eigendecomposition.py` |
| 2 | Load a real grayscale image (or convert one to grayscale). Compress it at ranks 1, 5, 10, 25,… | code | T1 | `ex02_image_compression.py` |
| 3 | Build a tiny recommendation system. Create a 10x8 user-movie ratings matrix with some known e… | code | T0 | `ex03_recommendation_system.py` |
| 4 | Create a 100x50 document-term matrix with 3 synthetic topics. Each topic has 5 associated ter… | code | T0 | `ex04_latent_topics.py` |
| 5 | Generate a clean low-rank matrix (rank 3, size 50x40) and add Gaussian noise at different lev… | code | T0 | `ex05_optimal_rank_vs_noise.py` |
<!-- generated:end -->

## Answers, and three things the lesson's code does quietly

**1 — accuracy comparison, which is what the exercise asks for.** The prescribed
route (eigendecompose AᵀA, then U = AVΣ⁻¹) is the instructive one *and* the wrong
one:

| matrix | AᵀA route | numpy.linalg.svd |
|---|---:|---:|
| well-conditioned | 5.9e-16 | — |
| σ spanning 1e-9 | **8.7e-09** | 5.9e-16 |

Forming AᵀA squares the condition number, so anything below √ε·σ₁ ≈ 1e-8·σ₁ is
unrecoverable. This is exactly why LAPACK bidiagonalises A directly
(Golub–Reinsch) and never forms the Gram matrix.

**`svd_from_scratch` silently returns fewer singular values.** On the same
ill-conditioned matrix it gives **2 of 4**, with no error and no warning: it
breaks out of its loop when an AᵀA eigenvalue drops below `1e-10`, which is
σ < 1e-5. Same shape of finding as `gram_schmidt` in lesson 01 — a threshold
chosen for the common case, applied silently to the uncommon one.

**`power_iteration` seeds with an unseeded `np.random.randn`,** so results are
not reproducible run to run. It converges to the same dominant eigenvector, but
nothing in the signature says the function is stochastic.

**2 — the rank where the image becomes acceptable: 100.** Taking "acceptable" as
relative Frobenius error below 10%:

| k | error | stores |
|---:|---:|---:|
| 1 | 29.2% | 0.4% |
| 10 | 16.0% | 3.9% |
| 50 | 10.3% | 19.5% |
| **100** | **7.4%** | **39.1%** |

Two things worth noting. Rank 1 already captures 91.5% of the *spectral energy*
at 29.2% error — natural images are dominated by mean brightness, which one outer
product expresses, so energy is a misleading proxy for quality. And by the time
error is acceptable, the factors occupy 39% of the original pixels: SVD is a poor
image codec next to anything exploiting local structure. Break-even is k = 256.

A naming trap: `compression_ratio(m, n, k)` returns `compressed / original` — a
*fraction of original size*, not an "N× smaller" ratio. 0.39 means 39% of the
original, i.e. 2.6× compression.

**3 — "reasonable" needs the right baseline.** The predictions land in 1–5 and
hold 0.141 MAE on observed entries, but the check that matters is the held-out
comparison against **the row means the matrix was filled with**: 0.879 vs 0.995,
12% better. Without that, a rank-3 reconstruction that merely reproduced the fill
would pass every other test while having learned nothing.

**4 — the topics separate, in permuted order.** σ₃/σ₄ = 3.59, top 3 hold 83.4% of
energy, and the 3D projection separates classes at intra/inter 0.159 against
0.469 in the raw 50-dimensional term space. Each component loads on a *distinct*
topic block — component 0 → topic 2, component 1 → topic 1, component 2 → topic 0
— beating noise-term loadings by at least 10×. The mapping is `[2, 1, 0]`: SVD
orders by variance, not by topic index, so latent dimensions carry no labels and
have to be read off the loadings.

**5 — how does the optimal k change with noise? It doesn't.** Best k = 3 at
σ = 0.1, 0.5, 1.0 and 2.0. The exercise asks for a plot of a relationship that is
flat, and the reason is in the setup: error is measured against the *clean*
matrix, so components past the true rank can only add noise, whatever σ is.

What does move with σ is the practitioner's ability to *find* that rank. The
spectral gap σ₃/σ₄ collapses from 24.8 to 1.4 across the four levels, and
Gavish–Donoho hard thresholding recovers 3, 3, 3, **2** components — it
under-estimates at σ = 2.0. The sweep can always find k=3 only because it peeks
at the clean matrix, which is not available in any real problem.

The relative benefit of truncating is near-constant at 2.6–2.9× across all four
noise levels: both errors scale linearly in σ, so the ratio cancels it. The
absolute gain grows with noise; the proportional one does not.
