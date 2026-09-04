<!-- generated:start -->
# 01-math-foundations / 10-dimensionality-reduction

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/01-math-foundations/10-dimensionality-reduction/) · upstream spec
`phases/01-math-foundations/10-dimensionality-reduction/docs/en.md`

```bash
uv run demo practice run 10-dimensionality-reduction --ex 1
uv run demo explain 10-dimensionality-reduction --ex 1
uv run pytest demos/phases/01-math-foundations/10-dimensionality-reduction
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Modify the PCA class to support `inverse_transform`. Reconstruct MNIST digits from 10, 50, an… | code | T0 | `ex01_pca_reconstruction.py` |
| 2 | Run t-SNE on the same MNIST subset with perplexity values of 5, 30, and 100. Describe how the… | code | T1 | `ex02_tsne_perplexity.py` |
| 3 | Take a dataset with 50 features where only 5 are informative (generate one with `sklearn.data… | code | T0 | `ex03_explained_variance_curve.py` |
<!-- generated:end -->

## Answers, and where the exercises' premises do not hold

**"MNIST" here is sklearn's bundled `load_digits`** — 1797 images, 8×8, 64
features, no download. That keeps exercise 1 at T0 and is the only reading under
which the exercise runs offline.

**1 — reconstruction error, and a value that cannot exist.**

| k | MSE | variance kept |
|---:|---:|---:|
| 10 | 4.914 | 73.8% |
| 50 | 0.0085 | ~100% |
| 200 | 0.0000 | 100% |

With all 64 components the error is 3.9e-29: PCA is a rotation, so keeping every
component loses nothing, and the error at smaller k is exactly the discarded
variance. The exercise's third value, 200, is **unreachable** — the data has 64
dimensions. `PCA(n_components=200)` does not complain; `eigenvectors[:, :200]`
simply runs out and you silently get 64. Worth knowing before trusting an
`n_components` you did not check against `X.shape[1]`.

**2 — how the output changes: non-monotonically.** The exercise asks *why*
perplexity affects tightness, which presumes a direction. There isn't one:

| perplexity | intra/inter ratio (lower = tighter) |
|---:|---:|
| 5 | 0.2096 |
| **30** | **0.1734** |
| 100 | 0.2377 |

Perplexity is the effective number of neighbours each point tries to preserve.
At 5, each point sees too few and a single digit class fragments into local
shards. At 100, the neighbourhood spans other digits and the classes blur into
each other. 30 is best, and is scikit-learn's default for exactly this reason.
Note also that 100 sits at the usual (n−1)/3 ceiling for a 300-point subset, so
the exercise's third value is a boundary case rather than a midpoint.

**3 — does the explained-variance curve find the 5 informative dimensions? No.**
90% of the variance needs **42** components, and the top 5 carry only **23.6%**.
The spectrum has no cliff: the largest component is 5.1× the median one.

This is not a PCA failure, it is the premise. `make_classification(n_informative=5)`
generates 45 further features as **independent noise at comparable scale**, so
they occupy real variance. PCA is unsupervised — it sees variance, never the
labels those 5 features are informative *about*. Informative and high-variance
are different properties, and the exercise conflates them.

To show the method is fine, the solution also builds a genuinely rank-5 dataset —
5 latent factors mixed into 50 dimensions — where the same curve puts the elbow
at exactly 5, carries 99.998% in the top 5, and drops **199,450×** between
components 5 and 6. That is what an elbow looks like when there is one.
