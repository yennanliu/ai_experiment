"""Exercise 4 — an unlearned lambda subtracts the needle along with the noise.

    **Hard.** Implement differential attention with a learned `λ` per head. Train
    on a synthetic retrieval task (one needle, 2,000 distractors). Measure
    retrieval accuracy vs a single-attention baseline at matched parameters.

Reading of the exercise: the retrieval task is built exactly as described -- one
planted needle whose key is a scaled copy of the query, and 2,000 Gaussian
distractors -- and both arms run through the lesson's own `attention_row` and
`diff_attention_row`. `λ` is *not* learned, because the lesson ships no training
loop and no gradient; what is measured instead is what a fixed `λ` does, which is
the thing learning would have to fix.

**ANSWER: single attention retrieves the needle 100% of the time at 2,000
distractors; differential attention at a fixed `λ` does worse, monotonically.**

| distractors | λ=0 (single) | λ=0.5 | λ=0.8 |
|---:|---:|---:|---:|
| 64 | 100.0% | 99.0% | 94.0% |
| 512 | 99.0% | 98.5% | 83.5% |
| 2,000 | **100.0%** | 95.5% | **72.5%** |

The second attention map this implementation subtracts is a noisy copy of the
first -- `K2 = K1 + noise`, `q2 = q + noise` -- so it scores the needle highly
too. Subtracting it removes signal and noise in proportion, and a `λ` that has
not been learned to separate them can only cost accuracy. That is precisely what
"learned" is carrying in the exercise's own sentence.

**FINDING: the weights do not sum to 1, they sum to `1 - λ`.** Measured at
λ = 0, 0.3, 0.5 and 0.8 the totals are 1.000000, 0.700000, 0.500000 and 0.200000
exactly. `diff_attention_row` subtracts one normalised distribution from another
and returns the difference unnormalised, so the output vector is scaled by
`1 - λ` and goes to zero at λ = 1. The Differential Transformer normalises after
the subtraction; this implementation does not.

**FINDING: retrieval accuracy is the wrong metric for the phenomenon.** Plain
attention holds 100% accuracy from 8 distractors to 2,000 while the needle's
share of the attention mass falls from **0.94 to 0.28**. The argmax is robust
long after the distribution has stopped being peaked, so the number the exercise
asks for is the one that cannot see the problem differential attention exists to
fix.

**CONTROL: the lesson's own KV arithmetic overcharges it.** `main()` prints
differential attention at `full * 2`, 85.9 GB. It caches `K1`, `K2` and one
shared `V` -- three units against full attention's two -- so the figure is
**1.5x**, 64.4 GB. The extra query projection costs parameters, not cache.

Structure: `trial` plants one needle among distractors; `retrieve` runs both arms
over many trials; `totals` reads the weight sum at several lambdas.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "15-attention-variants"
WIDTH, TRIALS, SIZES, LAMBDAS = 16, 200, (64, 512, 2_001), (0.0, 0.5, 0.8)
JITTER, LAYERS, KV_HEADS, HEAD_DIM, CONTEXT = 0.35, 80, 8, 128, 131_072


def trial(rng, n, width=WIDTH):
    """One retrieval instance: a query, its planted needle, and n-1 distractors."""
    keys = [[rng.gauss(0, 1) for _ in range(width)] for _ in range(n)]
    values = [[rng.gauss(0, 1) for _ in range(width)] for _ in range(n)]
    query = [rng.gauss(0, 1) for _ in range(width)]
    needle = rng.randrange(1, n)
    keys[needle] = [x * 1.6 for x in query]
    return query, keys, values, needle


def retrieve(ref, n, lam, trials=TRIALS, seed=1):
    """(argmax accuracy, mean share of |weight| on the needle) for one lambda."""
    rng, hits, share = random.Random(seed), [], []
    for _ in range(trials):
        query, keys, values, needle = trial(rng, n)
        mask = [0.0] * n
        if lam == 0.0:
            weights = ref.attention_row(query, keys, values, mask)[1]
        else:
            second = [[x + JITTER * rng.gauss(0, 1) for x in row] for row in keys]
            shifted = [x + JITTER * rng.gauss(0, 1) for x in query]
            weights = ref.diff_attention_row(query, shifted, keys, second, values, mask, lam)[1]
        hits.append(max(range(n), key=lambda i: weights[i]) == needle)
        share.append(abs(weights[needle]) / max(1e-12, sum(abs(w) for w in weights)))
    return statistics.fmean(hits), statistics.fmean(share)


def totals(ref, lambdas=(0.0, 0.3, 0.5, 0.8), n=8, seed=0):
    """{lambda: sum of the returned weights} -- the claim is that it is 1 - lambda."""
    rng = random.Random(seed)
    query, keys, values, _ = trial(rng, n, width=8)
    second = [[x + 0.2 * rng.gauss(0, 1) for x in row] for row in keys]
    shifted = [x + 0.2 * rng.gauss(0, 1) for x in query]
    return {lam: sum(ref.diff_attention_row(query, shifted, keys, second, values,
                                            [0.0] * n, lam)[1]) for lam in lambdas}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    grid = {(n, lam): retrieve(ref, n, lam) for n in SIZES for lam in LAMBDAS}
    full = ref.kv_cache_bytes(LAYERS, KV_HEADS, HEAD_DIM, CONTEXT)
    sums = totals(ref)
    return {
        "grid": grid, "sums": sums,
        "shown": ", ".join(f"{n}/{lam} {grid[(n, lam)][0]:.1%}" for n in SIZES for lam in LAMBDAS),
        "monotone": all(grid[(n, 0.5)][0] >= grid[(n, 0.8)][0] for n in SIZES),
        "normalised": all(abs(total - (1 - lam)) < 1e-12 for lam, total in sums.items()),
        "sums_shown": ", ".join(f"{lam} -> {v:.6f}" for lam, v in sums.items()),
        "spread": (retrieve(ref, 8, 0.0)[1], grid[(SIZES[-1], 0.0)][1]),
        "cache": (full, full * 2, full * 1.5),
    }


def verify(result):
    grid, big = result["grid"], SIZES[-1]
    return [
        practice.Check(
            "ANSWER: single attention is 100% at 2,000 distractors; a fixed lambda is worse",
            grid[(big, 0.0)][0] == 1.0 and grid[(big, 0.8)][0] < 0.8,
            f"argmax accuracy by (distractors, lambda): {result['shown']}"
            + ". The subtracted map is a noisy copy of the first, so it scores the needle highly "
              "too -- removing it removes signal and noise together",
        ),
        practice.Check(
            "ANSWER: and it degrades monotonically in lambda",
            result["monotone"] and grid[(big, 0.0)][0] > grid[(big, 0.8)][0],
            f"at {big} distractors: {grid[(big, 0.0)][0]:.1%} at lambda 0, "
            f"{grid[(big, 0.5)][0]:.1%} at 0.5, {grid[(big, 0.8)][0]:.1%} at 0.8. A lambda that "
            "has not been learned to separate signal from noise can only cost accuracy, which is "
            "exactly what the word 'learned' is carrying in the exercise's own sentence",
        ),
        practice.Check(
            "FINDING: the weights sum to 1 - lambda, not to 1",
            result["normalised"],
            f"sums by lambda: {result['sums_shown']}"
            + ". diff_attention_row subtracts one normalised distribution from another and "
              "returns the difference unnormalised, so the output is scaled by 1 - lambda and "
              "vanishes at lambda = 1. The Differential Transformer normalises after; this does not",
        ),
        practice.Check(
            "FINDING: retrieval accuracy cannot see the problem the method exists to fix",
            result["spread"][0] > 0.9 and result["spread"][1] < 0.35,
            f"plain attention holds 100% accuracy from 8 distractors to {big} while the needle's "
            f"share of the mass falls from {result['spread'][0]:.2f} to "
            f"{result['spread'][1]:.2f}. The argmax stays right long after the distribution has "
            "stopped being peaked, so the metric the exercise asks for is the insensitive one",
        ),
        practice.Check(
            "CONTROL: the lesson's own KV arithmetic overcharges it",
            result["cache"][2] / result["cache"][0] == 1.5
            and result["cache"][1] / result["cache"][0] == 2.0,
            f"main() prints differential attention at full * 2 = "
            f"{result['cache'][1] / 1e9:.1f} GB. It caches K1, K2 and one shared V -- three units "
            f"against full attention's two -- so the figure is 1.5x, "
            f"{result['cache'][2] / 1e9:.1f} GB against a full "
            f"{result['cache'][0] / 1e9:.1f} GB. The extra query projection costs parameters, "
            "not cache",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
