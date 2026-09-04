"""Exercise 4 — Gini against entropy on 5 datasets, and why they agree.

    Compare Gini impurity vs entropy as split criteria on 5 different datasets.
    Measure accuracy and tree depth. In most cases, they produce nearly identical
    results. Explain why.

Reading of the exercise: the tempting answer to "explain why" — near-identical
concave curves, so identical split rankings, so identical trees — is measurably
false at its middle step. Checks 5 and 6 measure that ranking with the lesson's
own `information_gain` instead of inferring it; the README carries the argument.
"""

from __future__ import annotations

from harness import parity, practice, stats

PHASE, LESSON = "02-ml-fundamentals", "04-decision-trees"
SEEDS = (1, 2, 3, 4, 5)
CRITERIA, DEPTH = ("gini", "entropy"), 6


def tree_depth(node):
    if node is None or node.get("leaf"):
        return 0
    return 1 + max(tree_depth(node["left"]), tree_depth(node["right"]))


def _fit(ref, criterion, X_train, y_train, X_test, y_test) -> tuple:
    tree = ref.DecisionTree(max_depth=DEPTH, criterion=criterion)
    tree.fit(X_train, y_train)
    return ref.accuracy(y_test, tree.predict(X_test)), tree_depth(tree.tree)


def _curves(ref, n=1000) -> dict:
    pairs = [(ref.gini_impurity([0] * i + [1] * (n - i)),
              ref.entropy([0] * i + [1] * (n - i))) for i in range(1, n)]
    return {"gap": max(abs(g - e / 2) for g, e in pairs),
            "peak_gini": max(g for g, _ in pairs), "peak_h": max(e for _, e in pairs)}


def _root_gains(ref, X, y) -> list:
    """Root candidates under both criteria, enumerated as `_best_split` does."""
    out = []
    for feature in range(len(X[0])):
        values = sorted({row[feature] for row in X})
        for low, high in zip(values, values[1:]):
            cut = (low + high) / 2.0
            side = [row[feature] <= cut for row in X]
            parts = [[v for v, s in zip(y, side) if s is w] for w in (True, False)]
            out.append([(feature, cut)]
                       + [ref.information_gain(y, *parts, c) for c in CRITERIA])
    return out


def _one_seed(ref, seed) -> dict:
    """Both trees and the root ranking, off one shared train/test split."""
    parts = ref.train_test_split(
        *ref.generate_classification_data(n_samples=250, seed=seed), seed=seed)
    gains = _root_gains(ref, parts[0], parts[1])
    ranked = [[row[i] for row in gains] for i in (1, 2)]
    picks = {gains[max(range(len(gains)), key=col.__getitem__)][0] for col in ranked}
    fits = [_fit(ref, c, *parts) for c in CRITERIA]
    return {"acc": [f[0] for f in fits], "depth": [f[1] for f in fits],
            "n": len(gains), "tau": stats.kendall_tau(*ranked),
            "same_pick": len(picks) == 1}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "trees")
    return {"seeds": {seed: _one_seed(ref, seed) for seed in SEEDS},
            "curves": _curves(ref)}


def _summary(result) -> dict:
    """Everything `verify` compares, so `verify` stays a flat list of claims."""
    got = result["seeds"]
    gaps = {s: abs(got[s]["acc"][0] - got[s]["acc"][1]) for s in SEEDS}
    return {
        "table": "; ".join(f"{s}: {got[s]['acc'][0]:.0%}/d{got[s]['depth'][0]} vs "
                           f"{got[s]['acc'][1]:.0%}/d{got[s]['depth'][1]}" for s in SEEDS),
        "taus": ", ".join(f"{s}: {got[s]['tau']:.3f}" for s in SEEDS),
        "worst": max(gaps.values()), "same_accuracy": sum(g < 1e-9 for g in gaps.values()),
        "same_depth": sum(got[s]["depth"][0] == got[s]["depth"][1] for s in SEEDS),
        "same_pick": sum(got[s]["same_pick"] for s in SEEDS),
        "ranked_apart": all(0.85 < got[s]["tau"] < 0.98 for s in SEEDS),
        "candidates": got[SEEDS[0]]["n"],
    }


def verify(result):
    got, curves, n = _summary(result), result["curves"], len(SEEDS)
    return [
        practice.Check(f"both criteria run on all {n} datasets",
                       len(result["seeds"]) == n, "acc/depth " + got["table"]),
        practice.Check("accuracy is nearly identical everywhere", got["worst"] < 0.05,
                       f"{got['same_accuracy']} of {n} identical, worst {got['worst']:.1%}"),
        practice.Check("…and the trees are the same depth",
                       got["same_depth"] >= n - 1, f"{got['same_depth']} of {n} agree"),
        practice.Check("WHY (1): near-identical curves, on axes differing by exactly 2",
                       curves["gap"] < 0.06
                       and abs(curves["peak_h"] / curves["peak_gini"] - 2) < 1e-9,
                       f"H peaks at {curves['peak_h']:.4f} bits, Gini at "
                       f"{curves['peak_gini']:.4f}; largest gap {curves['gap']:.4f}"),
        practice.Check("FINDING: they do NOT rank candidate splits the same way",
                       got["ranked_apart"],
                       f"tau over {got['candidates']} root candidates — {got['taus']} — "
                       f"about 1 pair in 20 ordered differently"),
        practice.Check("WHY (2): they can pick a different root split and still tie",
                       got["same_pick"] < n,
                       f"same root (feature, threshold) on {got['same_pick']} of {n}; "
                       f"elsewhere a different threshold on the same feature, yet accuracy "
                       f"ties on {got['same_accuracy']} of {n} — near-tied candidates cut "
                       f"near-equivalent partitions"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
