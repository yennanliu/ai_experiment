"""Exercise 1 — small-N FID is biased up, not down, and the bias is a closed form in 1/N.

    **Easy.** Run `code/main.py`. Compare FID at N=100 vs N=1000 on the same
    synthetic distributions. Report bias magnitude.

Reading of the exercise: "the same synthetic distributions" is read as the
lesson's own null case -- real and generated pools both drawn by the lesson's
`make_features(0.0, n, 4, rng)`, so the true FID is exactly 0 and every unit of
the measured score is bias. One draw is noisy (its spread is ~40% of its mean),
so each N is averaged over 30 independent draws scored by the lesson's own
`fid`.

**ANSWER: about 0.027 at N=100 and 0.0029 at N=1000 -- 9.3 times smaller.**
`N * FID` is 2.68 and 2.87: the bias falls as `1/N`, so N=1000 removes about
90% of it and no finite N removes all of it.

**FINDING: the bias is upward, and the lesson's prose says the opposite.**
`docs/en.md` says small N "under-estimates covariance, gives falsely low FID".
All 60 of the null draws score above zero, and the lesson's own
`main()` prints "biased up at small N". The covariance is not under-estimated
either: `covariance` divides by `n - 1`, and its trace averages 0.638 at N=100
against the true `4 * 0.4^2 = 0.64`. The bias comes from the two pools' independent noise, which
the distance can only add.

**FINDING: the bias is predictable, `s^2 (2d + d(d+1)/2) / N`.** The mean term
contributes `2 d s^2 / N` (0.0128 predicted, 0.0122 measured at N=100) and the
square-root term `s^2 d (d+1) / (2N)`, together 2.88/N for the lesson's d=4,
s=0.4: predicted 0.0288 and 0.00288 against measured 0.0268 and 0.00287, and at
d=8, N=400 predicted 0.0208 against measured 0.0223. It also shows why N matters
more as d grows: the covariance term is `(d+1)/4` times the mean term -- 512x at
Inception's d=2048.

**CONTROL: the matrix square root is right.** The lesson's `jacobi_sqrt`
(Denman-Beavers, despite the name) squares back to `cov_r @ cov_g` within 6.9e-18,
so the bias is the statistic's, not the arithmetic's.

Structure: `null_fids` scores identical pools; `predicted` is the closed form.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "14-evaluation-fid-clip-score"
DIM, SCALE, SEEDS, SIZES = 4, 0.4, 30, (100, 1000)


def null_fids(ref, n, dim=DIM, seeds=SEEDS):
    """FID between two pools drawn from one distribution, over `seeds` draws."""
    scores, means, traces = [], [], []
    for seed in range(seeds):
        rng = random.Random(1000 * n + seed)
        real = ref.make_features(0.0, n, dim, rng, SCALE)
        gen = ref.make_features(0.0, n, dim, rng, SCALE)
        mu_r, mu_g = ref.mean_vec(real), ref.mean_vec(gen)
        scores.append(ref.fid(real, gen))
        means.append(sum((a - b) ** 2 for a, b in zip(mu_r, mu_g)))
        traces.append(ref.trace(ref.covariance(real, mu_r)))
    return scores, statistics.mean(means), statistics.mean(traces)


def predicted(n, dim=DIM):
    """Expected null FID: mean term 2 d s^2 / N plus sqrt term s^2 d (d+1) / 2N."""
    return SCALE**2 * (2 * dim + dim * (dim + 1) / 2) / n


def sqrt_residual(ref):
    """Worst entry of sqrt(P) @ sqrt(P) - P, for one product of sample covariances."""
    rng = random.Random(5)
    pools = [ref.make_features(0.0, 100, DIM, rng, SCALE) for _ in range(2)]
    covs = [ref.covariance(p, ref.mean_vec(p)) for p in pools]
    prod = ref.matmul(covs[0], covs[1])
    root = ref.jacobi_sqrt(prod)
    back = ref.matmul(root, root)
    return max(abs(back[i][j] - prod[i][j]) for i in range(DIM) for j in range(DIM))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = {n: null_fids(ref, n) for n in SIZES}
    wide = null_fids(ref, 400, dim=8)
    return {
        "bias": {n: statistics.mean(runs[n][0]) for n in SIZES},
        "positive": sum(s > 0 for n in SIZES for s in runs[n][0]),
        "mean_term": runs[100][1],
        "trace": runs[100][2],
        "predicted": {n: predicted(n) for n in SIZES},
        "wide": (statistics.mean(wide[0]), predicted(400, 8)),
        "doc_says_low": "falsely low FID" in parity.doc_text(PHASE, LESSON),
        "residual": sqrt_residual(ref),
    }


def verify(result):
    bias, pred = result["bias"], result["predicted"]
    ratio = bias[100] / bias[1000]
    wide, wide_pred = result["wide"]
    return [
        practice.Check(
            "ANSWER: about 0.027 at N=100 and 0.0029 at N=1000, a 1/N law",
            7 < ratio < 13 and 0.02 < bias[100] < 0.04,
            f"mean null FID over {SEEDS} draws is {bias[100]:.4f} at N=100 and "
            f"{bias[1000]:.5f} at N=1000, a ratio of {ratio:.1f}; N*FID is "
            f"{100 * bias[100]:.2f} and {1000 * bias[1000]:.2f}, so the bias falls as 1/N",
        ),
        practice.Check(
            "FINDING: the bias is upward, and the lesson's prose says it is low",
            result["doc_says_low"] and result["positive"] == 2 * SEEDS,
            f"docs/en.md says small N 'gives falsely low FID'; {result['positive']} of "
            f"{2 * SEEDS} identical-distribution draws score above zero. The covariance is "
            f"not under-estimated either: its trace averages {result['trace']:.3f} at N=100 "
            f"against a true {DIM * SCALE**2:.2f}, because covariance divides by n - 1",
        ),
        practice.Check(
            "FINDING: the bias is the closed form s^2 (2d + d(d+1)/2) / N",
            all(abs(bias[n] / pred[n] - 1) < 0.15 for n in SIZES)
            and abs(wide / wide_pred - 1) < 0.15,
            f"predicted {pred[100]:.4f} and {pred[1000]:.5f} at d=4, {wide_pred:.4f} at d=8, N=400 "
            f"(measured {wide:.4f}). The mean term alone is {result['mean_term']:.4f} at "
            f"N=100 against 2ds^2/N = {2 * DIM * SCALE**2 / 100:.4f}; the covariance term "
            "is (d+1)/4 times it, 512x at Inception's d=2048",
        ),
        practice.Check(
            "CONTROL: the lesson's matrix square root is accurate",
            result["residual"] < 1e-12,
            f"jacobi_sqrt (a Denman-Beavers iteration) squares back to cov_r @ cov_g within "
            f"{result['residual']:.1e}, so the bias belongs to the statistic, not the arithmetic",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
