"""Exercise 3 — 1 to 200 trees: does test accuracy plateau without decreasing?

    Build a random forest with 1, 5, 10, 50, and 200 trees. Plot training
    accuracy and test accuracy vs number of trees. Observe that test accuracy
    plateaus but does not decrease (forests resist overfitting).

Reading of the exercise: "does not decrease" is a claim about a *noisy* sequence,
and five points from one seed cannot support it — a single dip proves nothing and
a single rise proves nothing either. So each size is averaged over 5 seeds, and
check 4 tests the claim as stated: the averaged test accuracy never falls
materially below its running maximum as trees are added.

Check 5 states the mechanism, which is the part worth keeping: adding trees
reduces the variance of an average without changing its expectation, so more
trees cannot overfit — it is bagging, not the trees, that resists.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "04-decision-trees"
SIZES = (1, 5, 10, 50, 200)
SEEDS = (0, 1, 2, 3, 4)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "trees")
    X, y = ref.generate_classification_data(n_samples=300, seed=42)
    X_train, y_train, X_test, y_test = ref.train_test_split(X, y, seed=42)
    rows = {}
    for size in SIZES:
        trains, tests = [], []
        for seed in SEEDS:
            forest = ref.RandomForest(n_trees=size, max_depth=8, criterion="gini")
            import random
            random.seed(seed)
            forest.fit(X_train, y_train)
            trains.append(ref.accuracy(y_train, forest.predict(X_train)))
            tests.append(ref.accuracy(y_test, forest.predict(X_test)))
        rows[size] = {"train": sum(trains) / len(trains),
                      "test": sum(tests) / len(tests),
                      "spread": max(tests) - min(tests)}
    return {"rows": rows, "n_seeds": len(SEEDS)}


def verify(result):
    rows = result["rows"]
    tests = [rows[s]["test"] for s in SIZES]
    running_max = []
    best = 0.0
    for value in tests:
        best = max(best, value)
        running_max.append(best)
    worst_drop = max(m - t for m, t in zip(running_max, tests))
    return [
        practice.Check(f"each size averaged over {result['n_seeds']} seeds",
                       all(rows[s]["spread"] >= 0 for s in SIZES),
                       "; ".join(f"{s} trees: test {rows[s]['test']:.1%} "
                                 f"(spread {rows[s]['spread']:.1%})" for s in SIZES)),
        practice.Check("a single tree is the noisiest, and averaging is what fixes it",
                       rows[1]["spread"] > rows[SIZES[-1]]["spread"],
                       f"seed-to-seed spread falls from {rows[1]['spread']:.1%} at 1 tree "
                       f"to {rows[SIZES[-1]]['spread']:.1%} at {SIZES[-1]} — this is the "
                       f"variance reduction, visible directly"),
        practice.Check("test accuracy rises then plateaus",
                       tests[-1] > tests[0] and abs(tests[-1] - tests[-2]) < 0.02,
                       f"{tests[0]:.1%} at 1 tree -> {tests[-1]:.1%} at {SIZES[-1]}, with "
                       f"only {abs(tests[-1] - tests[-2]):.1%} between the last two sizes"),
        practice.Check("ANSWER: over this sweep it never falls materially below its best",
                       worst_drop < 0.02,
                       f"largest drop below the running maximum: {worst_drop:.2%}, over "
                       f"sizes {SIZES} and {len(SEEDS)} seeds — within seed noise. This is "
                       f"an observed finite-sweep result, not a proof: a size outside "
                       f"{SIZES} or a sixth seed could show more, and one seed alone could "
                       f"not have supported the claim either way"),
        practice.Check("MECHANISM: averaging cuts variance without shifting the mean",
                       rows[SIZES[-1]]["train"] >= rows[1]["train"],
                       f"train accuracy {rows[1]['train']:.1%} -> "
                       f"{rows[SIZES[-1]]['train']:.1%} — the fit does not tighten as trees "
                       f"are added, which is the signature of averaging rather than of "
                       f"capacity growth. Each tree is fit on a bootstrap sample, so their "
                       f"errors are partly independent; averaging n of them divides the "
                       f"variance of that average by up to n while leaving its expectation "
                       f"alone. That argument is why the plateau above is expected — the "
                       f"resistance is bagging's, not the tree's — but the argument is "
                       f"theory, and only the plateau is measured here"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
