"""Exercise 5 — tree importance buries x3 under noise; single-column shuffles understate copies.

    **Permutation importance from scratch**: implement permutation importance.
    For each feature, shuffle its values 10 times, measure the average drop in
    F1 score. Compare the ranking against tree-based importance. Find cases
    where they disagree and explain why (hint: correlated features).

Reading of the exercise: the model is the lesson's `simple_logistic_importance`
(lr 0.1, 300 epochs) on all 20 standardised features of the lesson's 400
training rows; permutation importance is its F1 drop on 2000 fresh rows from the
same generator (seed 7) when one column is shuffled, averaged over 10 shuffles.
Tree importance is the lesson's `tree_importance` with its defaults (50 trees,
depth 5) on the same 400 rows. To test the hint, each correlated group (the
three copies of x1, of x2, and the two of x3) is also shuffled as one block.

**ANSWER: the rankings agree on the x1 and x2 copies and split on x3.** info_2,
which *is* x3, ranks 1st by permutation (F1 drop 0.057) and 14th by tree
importance, below four noise columns. The mixture corr_3 goes the other way: 6th
by tree importance, 18th by permutation. Rank correlation is 0.54.

**FINDING: the disagreement is in-sample credit, not cardinality.** The doc
warns that tree importance favours high-cardinality features, but every column
here has 400 distinct values. What tree importance does is score splits on the
rows it was fitted to: the ten noise columns take 22% of its total, against a
summed permutation drop of -0.005 on unseen rows. x3 has the weakest marginal
effect, so its genuine splits are outscored by noise splits deep in the trees.

**FINDING: shuffling one copy at a time understates a correlated signal.**
Shuffling info_0, info_3 and corr_0 together drops F1 by 0.293; the three
single-column drops sum to 0.155, half of that. The same holds for x2 (0.177
against 0.120) and x3 (0.101 against 0.081): with a copy still intact, the model
loses only part of the signal, so each copy looks less important than the
signal is.

**CONTROL: permutation importance is unbiased on noise:** the ten noise columns
average a drop of -0.0005, within shuffle noise of zero.

Structure: `f1` scores the model; `shuffled_drop` permutes a block of columns.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "18-feature-selection"
REPEATS = 10
GROUPS = {"x1": [0, 3, 5], "x2": [1, 4, 6], "x3": [2, 7]}


def f1(w, b, X, y):
    pred = ((X @ w + b) >= 0).astype(int)
    return 2 * np.sum(pred & y) / (pred.sum() + y.sum())


def shuffled_drop(w, b, X, y, cols, rng):
    """Mean F1 drop when `cols` are permuted together, over REPEATS shuffles."""
    base, drops = f1(w, b, X, y), []
    for _ in range(REPEATS):
        Xp = X.copy()
        Xp[:, cols] = X[rng.permutation(len(X))][:, cols]
        drops.append(base - f1(w, b, Xp, y))
    return float(np.mean(drops))


def ranks(scores):
    return np.argsort(np.argsort(-np.asarray(scores))) + 1


def noise_summary(names, perm, tree):
    noise = [i for i, n in enumerate(names) if n.startswith("noise")]
    return float(sum(tree[i] for i in noise)), [perm[i] for i in noise]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "feature_selection")
    X, y, names = ref.make_feature_selection_data(500, seed=42)
    X, y = X[:400], y[:400]
    mu, sd = X.mean(0), X.std(0)
    fresh, y_fresh, _ = ref.make_feature_selection_data(2000, seed=7)
    fresh = (fresh - mu) / sd
    w, b = ref.simple_logistic_importance((X - mu) / sd, y, lr=0.1, epochs=300)
    rng = np.random.RandomState(0)
    perm = [shuffled_drop(w, b, fresh, y_fresh, [f], rng) for f in range(20)]
    tree = ref.tree_importance(X, y)
    return {
        "names": names, "perm": perm, "tree": [float(t) for t in tree],
        "perm_rank": ranks(perm).tolist(), "tree_rank": ranks(tree).tolist(),
        "spearman": float(np.corrcoef(ranks(perm), ranks(tree))[0, 1]),
        "groups": {g: (shuffled_drop(w, b, fresh, y_fresh, c, rng), sum(perm[i] for i in c))
                   for g, c in GROUPS.items()},
        "noise": noise_summary(names, perm, tree),
        "unique": int(min(len(np.unique(X[:, f])) for f in range(20))),
    }


def verify(result):
    pr, tr, at = result["perm_rank"], result["tree_rank"], result["names"].index
    groups, (noise_share, noise) = result["groups"], result["noise"]
    above = sum(tr[at(f"noise_{i}")] < tr[at("info_2")] for i in range(10))
    return [
        practice.Check(
            "ANSWER: permutation ranks x3 first, tree importance buries it under noise",
            pr[at("info_2")] <= 2 and above >= 3 and tr[at("corr_3")] < pr[at("corr_3")] - 5,
            f"info_2 (x3): permutation rank {pr[at('info_2')]} (drop "
            f"{result['perm'][at('info_2')]:.3f}), tree rank {tr[at('info_2')]} below {above} "
            f"noise columns; corr_3: tree {tr[at('corr_3')]}, permutation "
            f"{pr[at('corr_3')]}; rank correlation {result['spearman']:.2f}"),
        practice.Check(
            "FINDING: tree importance credits in-sample noise splits, not cardinality",
            noise_share > 0.15 and sum(noise) < 0.01,
            f"every column has at least {result['unique']} distinct values; noise takes "
            f"{noise_share:.0%} of tree importance against a summed "
            f"permutation drop of {sum(noise):.3f}"),
        practice.Check(
            "FINDING: single-column shuffles understate every correlated group",
            all(block > single for block, single in groups.values())
            and groups["x1"][0] > 1.5 * groups["x1"][1],
            "; ".join(f"{g}: block {blk:.3f} vs sum of singles {one:.3f}"
                      for g, (blk, one) in groups.items())),
        practice.Check(
            "CONTROL: permutation importance is centred on zero for noise",
            abs(np.mean(noise)) < 0.002,
            f"the ten noise columns average a drop of {np.mean(noise):.4f}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
