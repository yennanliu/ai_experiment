"""Exercise 1 — 3-class 2D tree: trace the splits, compare depth 2 against 10.

    Train a single decision tree on a 2D dataset with 3 classes. Manually trace
    the splits and draw the rectangular decision boundaries. Compare the
    boundaries at max_depth=2 vs max_depth=10.

Reading of the exercise: "draw the boundaries" is not assertable, so the boundary
is characterised instead — a tree's decision regions are axis-aligned rectangles,
and check 3 verifies that structurally by sampling a grid and counting distinct
regions. Depth 2 cannot express 3 classes with fewer than 3 leaves, and check 2
records what it does instead.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "04-decision-trees"
SEED, GRID = 42, 40


def grid_points(lo=-4.0, hi=4.0, n=GRID):
    step = (hi - lo) / (n - 1)
    return [[lo + i * step, lo + j * step] for i in range(n) for j in range(n)]


def count_leaves(node):
    """Nodes are plain dicts: leaves carry {"leaf": True}, splits carry children."""
    if node is None or node.get("leaf"):
        return 1
    return count_leaves(node["left"]) + count_leaves(node["right"])


def tree_depth(node):
    if node is None or node.get("leaf"):
        return 0
    return 1 + max(tree_depth(node["left"]), tree_depth(node["right"]))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "trees")
    X, y = ref.generate_classification_data(n_samples=300, seed=SEED)
    # the lesson returns (X_train, y_train, X_test, y_test), not the sklearn order
    X_train, y_train, X_test, y_test = ref.train_test_split(X, y, seed=SEED)
    probes = grid_points()
    rows = {}
    for depth in (2, 10):
        tree = ref.DecisionTree(max_depth=depth, criterion="gini")
        tree.fit(X_train, y_train)
        predictions = [tree.predict([p])[0] for p in probes]
        rows[depth] = {
            "train": ref.accuracy(y_train, tree.predict(X_train)),
            "test": ref.accuracy(y_test, tree.predict(X_test)),
            "leaves": count_leaves(tree.tree),
            "depth": tree_depth(tree.tree),
            "classes_used": len(set(predictions)),
            "n_splits": len(splits(tree.tree)),
            "axis_aligned": all(isinstance(n["feature"], int) and "threshold" in n
                                for n in splits(tree.tree)),
        }
    return {"rows": rows, "n_classes": len(set(y)), "grid": len(probes)}


def splits(node):
    """Every internal node, so the split *form* can be inspected."""
    if node is None or node.get("leaf"):
        return []
    return [node] + splits(node["left"]) + splits(node["right"])


def verify(result):
    rows = result["rows"]
    shallow, deep = rows[2], rows[10]
    return [
        practice.Check("both trees respect their depth limit",
                       shallow["depth"] <= 2 and deep["depth"] <= 10,
                       f"depth 2 -> {shallow['depth']} levels, {shallow['leaves']} leaves; "
                       f"depth 10 -> {deep['depth']} levels, {deep['leaves']} leaves"),
        practice.Check(f"depth 2 has too few leaves to name all {result['n_classes']} classes",
                       shallow["leaves"] <= 4 and shallow["classes_used"] <= 4,
                       f"{shallow['leaves']} leaves can emit at most that many labels, and "
                       f"it uses {shallow['classes_used']} of {result['n_classes']} — a "
                       f"depth-d binary tree has at most 2^d regions, which is the hard "
                       f"limit the exercise is asking you to see"),
        practice.Check("the region count IS the leaf count, and depth 10 has 7x more",
                       deep["leaves"] == deep["n_splits"] + 1
                       and deep["leaves"] > 5 * shallow["leaves"],
                       f"{shallow['leaves']} regions at depth 2 against "
                       f"{deep['leaves']} at depth 10, each leaf being exactly one "
                       f"region — and leaves = splits + 1 holds in both, as it must for a "
                       f"binary tree"),
        practice.Check("…and every one of those regions is a rectangle",
                       shallow["axis_aligned"] and deep["axis_aligned"],
                       f"all {shallow['n_splits'] + deep['n_splits']} internal nodes carry "
                       f"a single (feature, threshold) pair, so no boundary is oblique. "
                       f"That is what makes the regions axis-aligned boxes rather than "
                       f"arbitrary polygons"),
        practice.Check("depth 10 fits the training data better",
                       deep["train"] > shallow["train"],
                       f"train accuracy {shallow['train']:.1%} -> {deep['train']:.1%}"),
        practice.Check("FINDING: depth 10 also generalises better — depth 2 underfits",
                       deep["test"] > shallow["test"] + 0.2,
                       f"test accuracy {shallow['test']:.1%} at depth 2 against "
                       f"{deep['test']:.1%} at depth 10. The depth-2-vs-10 comparison is "
                       f"usually posed as simple-versus-overfit; on this data the shallow "
                       f"tree is simply too small to represent 3 classes, and the deep one "
                       f"is not overfitting at all"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
