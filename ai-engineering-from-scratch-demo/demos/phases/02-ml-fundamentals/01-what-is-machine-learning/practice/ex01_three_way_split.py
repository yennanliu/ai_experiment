"""Exercise 1 — a 70/15/15 train/validation/test split, and why test stays sealed.

    Take any dataset (e.g., Iris, Titanic). Split it 70/15/15 into
    train/validation/test. Explain why you should not tune hyperparameters on the
    test set.

Reading of the exercise: the lesson's `train_test_split` is two-way, so a
three-way split is two calls — and the second fraction is **not** 0.15. Splitting
off 30% and then halving *that* needs `test_fraction=0.5` on the remainder, which
is the arithmetic slip the check exists to catch (check 2).

The prose half is answered in the README. Check 5 makes the argument
mechanically rather than restating it: selecting the best of 40 hyperparameter
candidates on a 22-row validation split inflates the apparent score by 12.7
points over the same candidate's test score — pure selection bias, on candidates
that differ by nothing.

Dataset: sklearn's bundled `load_iris`, so this stays T0 and offline.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "01-what-is-machine-learning"
SEED, N_CANDIDATES = 42, 40


def three_way(ref, X, y, seed=SEED):
    """70/15/15. The second call splits the 30% remainder in half, not by 0.15."""
    X_train, X_rest, y_train, y_rest = ref.train_test_split(X, y, test_fraction=0.3,
                                                            seed=seed)
    X_val, X_test, y_val, y_test = ref.train_test_split(X_rest, y_rest,
                                                        test_fraction=0.5, seed=seed)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy and scikit-learn — uv sync --extra math")
    try:
        from sklearn.datasets import load_iris
    except ImportError:
        raise practice.Skip("needs scikit-learn — uv sync --extra math") from None
    ref = parity.load_reference(PHASE, LESSON, "ml_intro")
    data = load_iris()
    X, y = data.data, data.target
    train, val, test = three_way(ref, X, y)
    sizes = [len(part[1]) for part in (train, val, test)]

    # the naive slip: test_fraction=0.15 on the remainder
    _, rest, _, y_rest = ref.train_test_split(X, y, test_fraction=0.3, seed=SEED)
    _, _, y_naive_val, y_naive_test = ref.train_test_split(rest, y_rest,
                                                           test_fraction=0.15, seed=SEED)

    # selection on a small split inflates the score it selected on
    rng = numpy.random.default_rng(SEED)
    scores = rng.normal(0.75, 0.05, size=(N_CANDIDATES, 2))   # (val, test) per candidate
    chosen = int(numpy.argmax(scores[:, 0]))
    return {"sizes": sizes, "total": len(y),
            "fractions": [n / len(y) for n in sizes],
            "naive_sizes": [len(y_naive_val), len(y_naive_test)],
            "selected_val": float(scores[chosen, 0]),
            "selected_test": float(scores[chosen, 1]),
            "mean_test": float(scores[:, 1].mean()),
            "overlap": _overlap(train, val, test)}


def _overlap(train, val, test):
    """Rows shared between any two splits, by exact feature-row identity."""
    def rows(part):
        return {tuple(row) for row in part[0]}
    a, b, c = rows(train), rows(val), rows(test)
    return len(a & b) + len(a & c) + len(b & c)


def verify(result):
    sizes, fractions = result["sizes"], result["fractions"]
    inflation = result["selected_val"] - result["selected_test"]
    return [
        practice.Check(f"the {result['total']} rows split into {sizes}",
                       sum(sizes) == result["total"],
                       f"train/val/test = {sizes}, summing to {sum(sizes)} — no row is "
                       f"dropped, which two chained splits make easy to get wrong"),
        practice.Check("the proportions are 70/15/15 to within a row",
                       abs(fractions[0] - 0.70) < 0.01
                       and all(abs(f - 0.15) < 0.01 for f in fractions[1:]),
                       f"measured {[f'{f:.1%}' for f in fractions]}. The second split uses "
                       f"test_fraction=0.5 on the 30% remainder; using 0.15 there would "
                       f"give {result['naive_sizes']} instead of {sizes[1:]}"),
        practice.Check("no row appears in more than one split",
                       result["overlap"] == 0,
                       f"{result['overlap']} shared rows — the splits are disjoint, which "
                       f"is the property the whole exercise rests on"),
        practice.Check("ANSWER: selecting on a split inflates that split's own score",
                       inflation > 0.05,
                       f"picking the best of {N_CANDIDATES} candidates by validation score "
                       f"gives {result['selected_val']:.3f} on validation but "
                       f"{result['selected_test']:.3f} on test — {inflation:.3f} of pure "
                       f"selection bias, with no real difference between the candidates"),
        practice.Check("…which is why the test split must be touched once, at the end",
                       result["selected_val"] > result["mean_test"],
                       f"the chosen candidate's test score ({result['selected_test']:.3f}) "
                       f"is ordinary against the {N_CANDIDATES}-candidate test mean of "
                       f"{result['mean_test']:.3f}. Tune on test and the same inflation "
                       f"lands on the number you report, where nothing is left to detect "
                       f"it — see the README"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
