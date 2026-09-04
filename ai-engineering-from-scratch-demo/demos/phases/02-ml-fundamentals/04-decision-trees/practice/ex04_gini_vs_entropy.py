"""Exercise 4 — Gini against entropy on 5 datasets, and why they agree.

    Compare Gini impurity vs entropy as split criteria on 5 different datasets.
    Measure accuracy and tree depth. In most cases, they produce nearly identical
    results. Explain why.

Reading of the exercise: "explain why" has a precise answer that can be measured
rather than asserted. Both criteria are concave, maximal at a uniform split and
zero at a pure one, and over the whole range of binary class balances they agree
on the *ranking* of candidate splits almost everywhere — check 5 measures that
agreement directly by sweeping p from 0 to 1 and correlating the two curves,
which is why the trees they build coincide.

The scaled comparison matters: entropy in bits peaks at exactly 1.0 where Gini
peaks at 0.5, so the two are not on the same axis at all. Compared as H/2 against
Gini the largest gap over every binary balance is 0.0545 — small, but not zero:
they are similar curves, not the same one rescaled.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "04-decision-trees"
SEEDS = (1, 2, 3, 4, 5)
DEPTH = 6


def tree_depth(node):
    if node is None or node.get("leaf"):
        return 0
    return 1 + max(tree_depth(node["left"]), tree_depth(node["right"]))


def _both_criteria(ref, seed):
    X, y = ref.generate_classification_data(n_samples=250, seed=seed)
    X_train, y_train, X_test, y_test = ref.train_test_split(X, y, seed=seed)
    entry = {}
    for criterion in ("gini", "entropy"):
        tree = ref.DecisionTree(max_depth=DEPTH, criterion=criterion)
        tree.fit(X_train, y_train)
        entry[criterion] = {"test": ref.accuracy(y_test, tree.predict(X_test)),
                            "depth": tree_depth(tree.tree)}
    return entry


def _impurity_sweep(ref, n=1000):
    """Both criteria at every binary class balance."""
    return [(i / n,
             ref.gini_impurity([0] * i + [1] * (n - i)),
             ref.entropy([0] * i + [1] * (n - i))) for i in range(1, n)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "trees")
    rows = {seed: _both_criteria(ref, seed) for seed in SEEDS}
    sweep = _impurity_sweep(ref)
    gaps = [abs(g - e / 2) for _, g, e in sweep]
    return {"rows": rows, "max_gap": max(gaps),
            "max_gini": max(g for _, g, _ in sweep),
            "max_entropy": max(e for _, _, e in sweep),
            "correlation": _correlation([g for _, g, _ in sweep],
                                        [e for _, _, e in sweep])}


def _correlation(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    da = [v - ma for v in a]
    db = [v - mb for v in b]
    return (sum(x * y for x, y in zip(da, db))
            / math.sqrt(sum(v * v for v in da) * sum(v * v for v in db)))


def verify(result):
    rows = result["rows"]
    same_accuracy = sum(1 for s in SEEDS
                        if abs(rows[s]["gini"]["test"] - rows[s]["entropy"]["test"]) < 1e-9)
    same_depth = sum(1 for s in SEEDS
                     if rows[s]["gini"]["depth"] == rows[s]["entropy"]["depth"])
    worst = max(abs(rows[s]["gini"]["test"] - rows[s]["entropy"]["test"]) for s in SEEDS)
    return [
        practice.Check(f"both criteria run on all {len(SEEDS)} datasets",
                       len(rows) == len(SEEDS),
                       "; ".join(f"seed {s}: gini {rows[s]['gini']['test']:.1%}/"
                                 f"d{rows[s]['gini']['depth']}, entropy "
                                 f"{rows[s]['entropy']['test']:.1%}/"
                                 f"d{rows[s]['entropy']['depth']}" for s in SEEDS)),
        practice.Check("accuracy is nearly identical everywhere",
                       worst < 0.05,
                       f"{same_accuracy} of {len(SEEDS)} datasets identical to the digit, "
                       f"worst difference {worst:.1%}"),
        practice.Check("…and the trees are the same depth",
                       same_depth >= len(SEEDS) - 1,
                       f"{same_depth} of {len(SEEDS)} agree on depth"),
        practice.Check("WHY (1): they are the same curve on different axes",
                       result["max_gap"] < 0.06
                       and abs(result["max_entropy"] / result["max_gini"] - 2) < 1e-9,
                       f"entropy peaks at {result['max_entropy']:.4f} bits where Gini peaks "
                       f"at {result['max_gini']:.4f} — a factor of 2. Compared on the same "
                       f"scale, H/2 against Gini, the largest gap over every binary balance "
                       f"is {result['max_gap']:.4f} — small but not zero, so these are "
                       f"similar curves rather than one rescaled into the other"),
        practice.Check("WHY (2): so they rank candidate splits the same way",
                       result["correlation"] > 0.99,
                       f"correlation between the two impurity curves over p ∈ (0,1) is "
                       f"{result['correlation']:.5f}. A split is chosen by which candidate "
                       f"lowers impurity most, and two near-identical concave functions "
                       f"order the candidates identically almost everywhere — the trees "
                       f"coincide because the *argmax* does, not because the numbers match"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
