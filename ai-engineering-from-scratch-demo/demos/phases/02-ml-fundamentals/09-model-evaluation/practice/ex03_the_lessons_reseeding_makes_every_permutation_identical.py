"""Exercise 3 — the lesson's reseeding makes all 100 label permutations the same permutation.

    Implement a permutation test for model comparison: shuffle the labels,
    retrain, and measure performance. Repeat 100 times to build a null
    distribution. Compute the p-value for the observed model performance against
    this distribution.

Reading of the exercise: "performance" is the lesson's own 5-fold
`cross_validate` accuracy of its `SimpleLogistic`, so every permutation
retrains five models. The p-value is `(1 + #{null >= observed}) / 101`. The
shuffle is written two ways: with the module-level `random.shuffle`, the
obvious call, and with a private `random.Random` instance.

**ANSWER: observed 5-fold accuracy 0.84 against a null of mean 0.503 (range
0.36-0.62), so p = 1/101 = 0.0099**, the smallest value 100 permutations can
report.

**FINDING: with `random.shuffle` all 100 permutations are the same
permutation.** The lesson's `cross_validate` calls `kfold_split`, which calls
`random.seed(42)` on the global RNG; every shuffle that follows a retrain starts
from the same state. One distinct label order in 100, one null score (0.52),
and p can only be 1/101 or 1. On this data it says 0.0099 and is right only
because the signal is strong.

**FINDING: on random labels the broken test rejects 3 times in 10.** On 10
coin-flip label sets of 50 rows the global-RNG test returns p = 0.0099 three
times and 1.0 seven times, a 30% false-positive rate at the 5% level; the
private RNG rejects 0 times, with p from 0.158 to 0.802.

**CONTROL:** a private `random.Random` gives 100 distinct label orders and 31
distinct null scores; the lesson's reseeding cannot reach an RNG it does not
own. (The same trap is in `train_val_test_split`, `stratified_kfold_split` and
`learning_curve`, which all call `random.seed`.)
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "09-model-evaluation"
PERMS, NULL_SETS = 100, 10


def cv_accuracy(ref, X, y):
    model_fn = lambda: ref.SimpleLogistic(lr=0.1, epochs=10)  # noqa: E731
    return statistics.mean(ref.cross_validate(X, y, model_fn, k=5, metric_fn=ref.accuracy))


def permutation_test(ref, X, y, shuffle):
    """Observed CV accuracy, its null over PERMS label shuffles, and the p-value."""
    observed = cv_accuracy(ref, X, y)
    null, seen = [], set()
    for _ in range(PERMS):
        labels = list(y)
        shuffle(labels)
        seen.add(tuple(labels))
        null.append(cv_accuracy(ref, X, labels))
    p = (1 + sum(s >= observed for s in null)) / (PERMS + 1)
    return {"observed": observed, "null": null, "p": p, "distinct": len(seen)}


def null_rejections(ref):
    """5%-level rejections on NULL_SETS datasets whose labels are pure coin flips."""
    naive, private = [], []
    for d in range(NULL_SETS):
        X, _ = ref.make_classification_data(50, seed=100 + d)
        y = random.Random(d).choices((0, 1), k=50)
        random.seed(d)
        naive.append(permutation_test(ref, X, y, random.shuffle)["p"])
        private.append(permutation_test(ref, X, y, random.Random(1000 + d).shuffle)["p"])
    return naive, private


def solve():
    ref = parity.load_reference(PHASE, LESSON, "evaluation")
    X, y = ref.make_classification_data(100, seed=3)
    naive = permutation_test(ref, X, y, random.shuffle)
    private = permutation_test(ref, X, y, random.Random(0).shuffle)
    null_naive, null_private = null_rejections(ref)
    return {"naive": naive, "private": private, "null_naive": null_naive,
            "null_private": null_private}


def verify(result):
    naive, private = result["naive"], result["private"]
    null = private["null"]
    p_naive, p_private = result["null_naive"], result["null_private"]
    rej_naive, rej_private = sum(p < 0.05 for p in p_naive), sum(p < 0.05 for p in p_private)
    return [
        practice.Check(
            "ANSWER: observed 0.84 against a null centred on 0.50, p = 1/101",
            all((private["p"] == 1 / 101, max(null) < private["observed"])),
            f"5-fold CV accuracy {private['observed']:.2f}; {PERMS} shuffles give a null of mean "
            f"{statistics.mean(null):.3f}, range {min(null):.2f}-{max(null):.2f}, so p = "
            f"{private['p']:.4f}, the smallest {PERMS} permutations can report",
        ),
        practice.Check(
            "FINDING: with random.shuffle all 100 permutations are one permutation",
            all((naive["distinct"] == 1, len(set(naive["null"])) == 1)),
            f"{naive['distinct']} distinct label order in {PERMS}: cross_validate calls "
            "kfold_split, which calls random.seed(42), so every shuffle after it starts from the "
            f"same state. The null is one number, {naive['null'][0]:.2f}, and p can only be "
            f"1/101 or 1; here it is {naive['p']:.4f}, right only because the signal is strong",
        ),
        practice.Check(
            "FINDING: on random labels the broken test rejects far above 5%",
            all((rej_naive >= 3, rej_naive > rej_private, set(p_naive) <= {1 / 101, 1.0})),
            f"on {NULL_SETS} coin-flip label sets the global-RNG test rejects at 5% "
            f"{rej_naive} times with p in {sorted({round(p, 4) for p in p_naive})}; the private "
            f"RNG rejects {rej_private} times with p from {min(p_private):.3f} to "
            f"{max(p_private):.3f}",
        ),
        practice.Check(
            "CONTROL: a private Random gives 100 distinct permutations",
            private["distinct"] == PERMS,
            f"{private['distinct']} distinct label orders and {len(set(null))} distinct null "
            "scores: the lesson's reseeding cannot reach an RNG it does not own",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
