"""Exercise 5 — MinHash for approximate Jaccard, at 50/100/200 hash functions.

    Implement MinHash for approximate Jaccard similarity. Generate 100 random
    sets, compute exact Jaccard for all pairs, and compare with MinHash
    approximation using 50, 100, and 200 hash functions. Plot the approximation
    error.

Reading of the exercise: "plot the error" is printed as a table, and the claim
worth testing is not that error falls — it is that error falls at the rate theory
predicts. A MinHash estimate is the mean of k Bernoulli trials with p = Jaccard,
so its standard error is √(J(1−J)/k): quadrupling k from 50 to 200 should halve
the error. Check 4 measures the ratio against that prediction, which a
"smaller is better" check would not catch if the implementation were subtly
biased.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "14-norms-and-distances"
N_SETS, UNIVERSE, SEED = 100, 200, 20260904
HASH_COUNTS = (50, 100, 200)
PRIME = (1 << 61) - 1


def make_hashes(rng, k):
    return [(rng.randrange(1, PRIME), rng.randrange(0, PRIME)) for _ in range(k)]


def signature(values, hashes):
    return [min((a * v + b) % PRIME for v in values) for a, b in hashes]


def minhash_similarity(sig_a, sig_b):
    return sum(1 for x, y in zip(sig_a, sig_b) if x == y) / len(sig_a)


def _errors(sets, pairs, exact, k):
    hashes = make_hashes(random.Random(SEED + k), k)
    signatures = [signature(s, hashes) for s in sets]
    return [minhash_similarity(signatures[i], signatures[j]) - exact[(i, j)]
            for i, j in pairs]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "distances")
    rng = random.Random(SEED)
    sets = [set(rng.sample(range(UNIVERSE), rng.randint(20, 80))) for _ in range(N_SETS)]
    pairs = [(i, j) for i in range(N_SETS) for j in range(i + 1, N_SETS)]
    exact = {(i, j): ref.jaccard_similarity(sets[i], sets[j]) for i, j in pairs}
    mean_j = sum(exact.values()) / len(exact)
    rows = {}
    for k in HASH_COUNTS:
        errors = _errors(sets, pairs, exact, k)
        rows[k] = {"mae": sum(abs(e) for e in errors) / len(errors),
                   "bias": sum(errors) / len(errors),
                   "predicted": math.sqrt(mean_j * (1 - mean_j) / k)}
    return {"rows": rows, "n_pairs": len(pairs), "mean_jaccard": mean_j}


def verify(result):
    rows = result["rows"]
    maes = [rows[k]["mae"] for k in HASH_COUNTS]
    observed = maes[0] / maes[-1]
    return [
        practice.Check(f"{result['n_pairs']:,} pairs over {N_SETS} sets, "
                       f"{len(HASH_COUNTS)} hash budgets",
                       len(rows) == len(HASH_COUNTS),
                       "; ".join(f"k={k}: MAE {rows[k]['mae']:.5f}" for k in HASH_COUNTS)
                       + f"; mean exact Jaccard {result['mean_jaccard']:.4f}"),
        practice.Check("error falls monotonically with the hash count",
                       maes[0] > maes[1] > maes[2],
                       f"{maes[0]:.5f} → {maes[1]:.5f} → {maes[2]:.5f}"),
        practice.Check("the estimator is unbiased at every budget",
                       all(abs(rows[k]["bias"]) < 0.01 for k in HASH_COUNTS),
                       "signed bias: " + ", ".join(f"k={k}: {rows[k]['bias']:+.5f}"
                                                   for k in HASH_COUNTS)
                       + " — MinHash estimates Jaccard, it does not merely correlate with it"),
        practice.Check("4x the hashes halves the error, as √(J(1−J)/k) predicts",
                       1.7 < observed < 2.3,
                       f"MAE ratio k=50 to k=200 is {observed:.3f}, against the predicted "
                       f"√4 = 2. This is the check a 'smaller is better' test would pass "
                       f"even with a biased implementation"),
        practice.Check("…and the absolute error tracks the predicted standard error",
                       all(0.5 < rows[k]["mae"] / rows[k]["predicted"] < 1.2
                           for k in HASH_COUNTS),
                       "; ".join(f"k={k}: MAE {rows[k]['mae']:.5f} vs σ "
                                 f"{rows[k]['predicted']:.5f}" for k in HASH_COUNTS)
                       + " — MAE is ≈0.8σ for a normal, so these ratios are what a correct "
                         "implementation should give"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
