"""Exercise 5 — MDI ranks a high-cardinality noise feature highly; permutation does not.

    Implement permutation importance. Compare it with MDI importance on a dataset
    where one feature is random noise but has high cardinality. MDI will rank the
    noise feature highly. Permutation importance will not.

Reading of the exercise: the claim needs two conditions the exercise omits, both
found by measurement.

**Label noise is required.** With a label the signal determines exactly, the tree
hits 100% training accuracy on the signal features alone and MDI gives the noise
columns 0.006 and 0.000 — the premise never arises. The label here carries 25%
flips, leaving residual impurity for a spurious split to reduce.

**Cardinality is the mechanism, not noise.** ~200 candidate thresholds always
include one that splits training labels a little, and MDI pays for it; the
2-valued noise column offers one and gets **0.0**. Check 5 uses it as the control.
Permutation importance is measured on held-out rows, where such a split buys
nothing.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "04-decision-trees"
SEED, N, DEPTH, FLIP = 42, 250, 8, 0.25
NAMES = ("signal_1", "signal_2", "noise_hi_card", "noise_binary")


def make_data(rng):
    """Two informative features, a continuous noise column, a binary noise column,
    and FLIP label noise — see the module docstring for why the last is needed."""
    X, y = [], []
    for _ in range(N):
        s1, s2 = rng.gauss(0, 1), rng.gauss(0, 1)
        X.append([s1, s2, rng.random(), float(rng.randint(0, 1))])
        label = 1 if s1 + s2 > 0 else 0
        y.append(1 - label if rng.random() < FLIP else label)
    return X, y


def permutation_importance(ref, tree, X, y, rng, repeats=5):
    base = ref.accuracy(y, tree.predict(X))
    scores = []
    for feature in range(len(X[0])):
        drops = []
        for _ in range(repeats):
            column = [row[feature] for row in X]
            rng.shuffle(column)
            shuffled = [row[:feature] + [column[i]] + row[feature + 1:]
                        for i, row in enumerate(X)]
            drops.append(base - ref.accuracy(y, tree.predict(shuffled)))
        scores.append(sum(drops) / repeats)
    return scores


def solve():
    ref = parity.load_reference(PHASE, LESSON, "trees")
    rng = random.Random(SEED)
    X, y = make_data(rng)
    X_train, y_train, X_test, y_test = ref.train_test_split(X, y, seed=SEED)
    tree = ref.DecisionTree(max_depth=DEPTH, criterion="gini")
    tree.fit(X_train, y_train)
    mdi = list(tree.feature_importances_)
    perm = permutation_importance(ref, tree, X_test, y_test, random.Random(SEED))
    cardinality = [len({row[j] for row in X_train}) for j in range(len(NAMES))]
    return {"mdi": mdi, "perm": perm, "cardinality": cardinality,
            "train": ref.accuracy(y_train, tree.predict(X_train)),
            "test": ref.accuracy(y_test, tree.predict(X_test)),
            "mdi_rank": sorted(range(len(NAMES)), key=lambda j: -mdi[j]),
            "perm_rank": sorted(range(len(NAMES)), key=lambda j: -perm[j])}


def _named(values, fmt=".4f") -> str:
    return ", ".join(f"{n}: {v:{fmt}}" for n, v in zip(NAMES, values))


def verify(result):
    mdi, perm, card = result["mdi"], result["perm"], result["cardinality"]
    hi, binary = 2, 3
    return [
        practice.Check("the noise columns really are noise, and differ in cardinality",
                       card[hi] > 100 and card[binary] == 2 and result["train"] < 1.0,
                       ", ".join(f"{n}: {c} values" for n, c in zip(NAMES, card))
                       + f"; the tree reaches {result['train']:.1%} train and "
                         f"{result['test']:.1%} test"),
        practice.Check("ANSWER: MDI gives the high-cardinality noise real credit",
                       mdi[hi] > 0.15 * sum(mdi),
                       _named(mdi)
                       + f" — {100 * mdi[hi] / sum(mdi):.0f}% of total MDI goes to a column "
                         f"that is pure noise"),
        practice.Check("ANSWER: permutation importance gives it essentially none",
                       abs(perm[hi]) < 0.05 and perm[hi] < mdi[hi],
                       _named(perm, "+.4f")
                       + " — measured on held-out rows, shuffling it costs nothing"),
        practice.Check("…and permutation ranks both signal features above both noise ones",
                       set(result["perm_rank"][:2]) == {0, 1},
                       f"permutation order {[NAMES[j] for j in result['perm_rank']]} against "
                       f"MDI's {[NAMES[j] for j in result['mdi_rank']]}"),
        practice.Check("MECHANISM: cardinality, not noise — MDI ignores the binary column",
                       mdi[binary] < 0.05 * mdi[hi],
                       f"MDI gives the 2-valued noise column {mdi[binary]:.4f} against "
                       f"{mdi[hi]:.4f} for the {card[hi]}-valued one, a "
                       f"{mdi[hi] / max(mdi[binary], 1e-9):.0g}x difference between two "
                       f"columns that are *equally* uninformative. {card[hi] - 1} candidate "
                       f"thresholds are enough to find one that splits training labels by "
                       f"luck; one threshold is not"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
