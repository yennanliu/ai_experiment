"""Exercise 2 — a random forest from the lesson's regression tree, and what it buys.

    Implement a random forest from scratch by adding random feature subsampling
    to the regression tree. Train 100 trees with `max_features=sqrt(n_features)`
    and average predictions. Compare variance reduction to a single tree.

Reading of the exercise: "variance" is read as the bias-variance quantity -- how
much a prediction moves when the training set is redrawn -- so 5 training sets
of 200 rows are drawn from the lesson's `make_regression_data` and every model
is scored at 100 fixed points against the known true function. The forest
subclasses the lesson's `SimpleRegressionTree`: at each node it hides all but
int(sqrt(5)) = 2 random features (as constant columns, which offer no split) and
lets the lesson's own `_build` choose the split. Depth 4 for every tree.

**ANSWER: the forest cuts prediction variance by 75%,** from 0.444 for one
depth-4 tree to 0.113 for 100 random-feature trees. Plain bagging reaches 0.150.

**FINDING: it decorrelates the trees by making each one worse.** Tree-to-tree
correlation is 0.065 against bagging's 0.240, but one random-feature tree varies
1.530 -- 3.4x a single full-feature tree.

**FINDING: the variance saved is spent on bias.** Squared bias rises to 1.016
from 0.771 (one tree) and 0.722 (bagging). x0 carries the slope-2 term but is
offered at only 40% of splits, and the forest's roots split on it in exactly 40%
of trees. Error against the true function: forest 1.129, bagging 0.872, one tree
1.215. The doc's table only says random forests "do not help with bias".

The lesson's own demo prints `bag_acc - single_acc` under the label "Variance
reduction"; it never redraws a training set, so it cannot measure one.

**CONTROL:** with all 5 features offered, the subclass reproduces
`SimpleRegressionTree` exactly (max difference 0.0).
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "11-ensemble-methods"
DEPTH, N_TRAIN, DATASETS, TREES, BAG_TREES = 4, 200, 5, 100, 25


def random_feature_tree(ref):
    """The lesson's SimpleRegressionTree, choosing among `k` random features per split."""
    class RandomFeatureTree(ref.SimpleRegressionTree):
        k, rng = 5, np.random.RandomState(0)  # set per tree by forest_predictions

        def _build(self, X, y, depth):
            if depth >= self.max_depth or len(y) < self.min_samples_split:
                return ref.TreeNode(value=np.mean(y))
            masked = np.zeros_like(X)  # a constant column offers no split
            keep = self.rng.choice(X.shape[1], self.k, replace=False)
            masked[:, keep] = X[:, keep]
            full, self.max_depth = self.max_depth, depth + 1  # one split, by the lesson
            node = super()._build(masked, y, depth)
            self.max_depth = full
            if node.value is None:
                go_left = X[:, node.feature_idx] <= node.threshold
                node.left = self._build(X[go_left], y[go_left], depth + 1)
                node.right = self._build(X[~go_left], y[~go_left], depth + 1)
            return node

    return RandomFeatureTree


def forest_predictions(Tree, X, y, X_eval, k, n_trees, seed, roots):
    """Each bootstrapped tree's predictions at X_eval; root features go to `roots`."""
    rng, rows = np.random.RandomState(seed), []
    for _ in range(n_trees):
        idx = rng.choice(len(y), len(y), replace=True)
        tree = Tree(max_depth=DEPTH)
        tree.k, tree.rng = k, rng
        tree.fit(X[idx], y[idx])
        rows.append(tree.predict(X_eval))
        roots.append(tree.root.feature_idx)
    return np.array(rows)


def decompose(per_dataset, truth, n_trees):
    """Law of total variance across training sets: within-forest + between-dataset."""
    within, means = np.mean(np.var(per_dataset, axis=1)), np.mean(per_dataset, axis=1)
    between = means.var(axis=0, ddof=1).mean() - within / n_trees
    return {"tree_var": within + between, "rho": between / (within + between),
            "forest_var": within / TREES + between,  # extrapolated to 100 trees
            "bias2": float(((means.mean(axis=0) - truth) ** 2).mean())}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "ensembles")
    Tree = random_feature_tree(ref)
    X_eval, _ = ref.make_regression_data(n_samples=100, seed=7)
    truth = 2.0 * X_eval[:, 0] + np.sin(3 * X_eval[:, 1]) - 0.5 * X_eval[:, 2] ** 2
    k = int(X_eval.shape[1] ** 0.5)
    rf, bag, singles, roots = [], [], [], []
    for r in range(DATASETS):
        X, y = ref.make_regression_data(n_samples=N_TRAIN, seed=r)
        rf.append(forest_predictions(Tree, X, y, X_eval, k, TREES, 100 + r, roots))
        bag.append(forest_predictions(Tree, X, y, X_eval, 5, BAG_TREES, 100 + r, []))
        (single := ref.SimpleRegressionTree(max_depth=DEPTH)).fit(X, y)
        singles.append(single.predict(X_eval))
    (same := Tree(max_depth=DEPTH)).fit(X, y)  # k=5, all offered: must be the lesson's tree
    source = (parity.lesson_dir(PHASE, LESSON) / "code" / "ensembles.py").read_text()
    return {
        "k": k, "rf": decompose(rf, truth, TREES), "bag": decompose(bag, truth, BAG_TREES),
        "single_var": float(np.var(singles, axis=0, ddof=1).mean()),
        "single_bias2": float(((np.mean(singles, axis=0) - truth) ** 2).mean()),
        "root_x0": roots.count(0) / len(roots), "x0_offered": k / X_eval.shape[1],
        "k5_gap": float(np.abs(same.predict(X_eval) - single.predict(X_eval)).max()),
        "demo_mislabels": "Variance reduction:   {bag_acc - single_acc" in source,
    }


def verify(result):
    rf, bag, s_var, s_b2 = (result[k] for k in ("rf", "bag", "single_var", "single_bias2"))
    cut = 1 - rf["forest_var"] / s_var
    err_rf, err_bag = (d["forest_var"] + d["bias2"] for d in (rf, bag))
    return [
        practice.Check(
            f"ANSWER: 100 trees at max_features={result['k']} cut variance by {cut:.0%}",
            cut > 0.6 and result["demo_mislabels"],
            f"variance across {DATASETS} training sets: one tree {s_var:.3f}, forest "
            f"{rf['forest_var']:.3f}, bagging {bag['forest_var']:.3f}. The lesson's demo prints "
            "bag_acc - single_acc as 'Variance reduction' and never redraws a training set",
        ),
        practice.Check(
            "FINDING: subsampling decorrelates the trees by making each one worse",
            rf["rho"] < bag["rho"] / 2 and rf["tree_var"] > 2 * s_var,
            f"tree-to-tree correlation {rf['rho']:.3f} against bagging's {bag['rho']:.3f}, but one "
            f"tree varies {rf['tree_var']:.3f}, {rf['tree_var'] / s_var:.1f}x a full-feature tree",
        ),
        practice.Check(
            "FINDING: the variance it saves is spent on bias",
            rf["bias2"] > 1.2 * s_b2 and err_rf > err_bag,
            f"bias^2 {rf['bias2']:.3f} vs {s_b2:.3f} (one tree), {bag['bias2']:.3f} (bagging); "
            f"x0 (the slope-2 term) is offered at {result['x0_offered']:.0%} of splits and roots "
            f"{result['root_x0']:.0%} of trees. Error vs truth: forest {err_rf:.3f}, bagging "
            f"{err_bag:.3f}, one tree {s_var + s_b2:.3f}",
        ),
        practice.Check(
            "CONTROL: with every feature offered, the subclass is the lesson's tree",
            result["k5_gap"] == 0,
            f"max |RandomFeatureTree(k=5) - SimpleRegressionTree| on the same data: "
            f"{result['k5_gap']:.1e}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
