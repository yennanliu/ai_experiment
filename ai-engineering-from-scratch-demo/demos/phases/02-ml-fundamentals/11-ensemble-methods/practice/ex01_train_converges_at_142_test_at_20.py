"""Exercise 1 — AdaBoost's training accuracy per round, and what "converged" can mean.

    Modify the AdaBoost implementation to track training accuracy after each
    round. Plot accuracy vs. number of estimators. When does it converge?

Reading of the exercise: the lesson's own `AdaBoostScratch` is fitted once for
150 rounds on the lesson's demo data (`make_classification_data(400, 5)`, 320
train / 80 test), and the accuracy after every round is read from the cumulative
sum of its `alpha_t * h_t(x)`. Fitting is sequential and deterministic, so the
first k stumps are exactly the model `n_estimators=k` would have fitted (checked
below). "Plot" is printed as a table; "converge" is answered for both the
training and the test curve, because they converge at very different rounds.

**ANSWER: training accuracy converges to 1.0, test accuracy by round ~20.**
Training accuracy first touches 1.0000 at round 112, loses it again, and holds
it only from round 142. Test accuracy reaches 0.8250 at round 20 and afterwards
only wanders between 0.8000 and 0.8375 (one test point is 0.0125), peaking at
round 32. The 120 rounds between the two are spent memorising the training set.

**FINDING: training accuracy is not monotone; AdaBoost's own loss is.** Training
accuracy falls on 40 of 149 rounds, while the mean exponential loss rises on
none: each round minimises exp(-yF), not the 0/1 error, and the error stays
under the bound prod 2 sqrt(eps(1 - eps)) throughout.

**FINDING: round 2 changes no prediction.** alpha_1 = 0.7862 > alpha_2 = 0.5323,
so sign(a1 h1 + a2 h2) = sign(h1) everywhere and train accuracy is 0.8281 after
both rounds. Two weighted voters are a dictatorship; the curve's first step is
at round 3.

**CONTROL:** `AdaBoostScratch(n_estimators=25)` scores exactly the staged
curve's round-25 value, and the mean exponential loss equals prod sech(alpha_t)
within 3.3e-16, so the lesson's weight update is exact AdaBoost.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "11-ensemble-methods"
ROUNDS, CHECKPOINTS = 150, (1, 2, 3, 10, 20, 50, 100, 150)


def staged_accuracy(model, X, y):
    """Accuracy of the first k stumps, for every k, from one fitted model."""
    votes = np.cumsum([a * s.predict(X) for a, s in zip(model.alphas, model.stumps)], axis=0)
    return (np.sign(votes) == y).mean(axis=1), votes


def settled(curve, final):
    """First round from which the curve never leaves `final` again (1-based)."""
    off = np.where(curve != final)[0]
    return int(off[-1]) + 2 if len(off) else 1


def solve():
    ref = parity.load_reference(PHASE, LESSON, "ensembles")
    X, y = ref.make_classification_data(n_samples=400, n_features=5)
    X_tr, X_te, y_tr, y_te = ref.train_test_split(X, y)
    model = ref.AdaBoostScratch(n_estimators=ROUNDS)
    model.fit(X_tr, y_tr)
    train, votes = staged_accuracy(model, X_tr, y_tr)
    test, _ = staged_accuracy(model, X_te, y_te)
    exp_loss = np.exp(-y_tr * votes).mean(axis=1)
    bound = np.cumprod(1 / np.cosh(model.alphas))  # prod 2 sqrt(eps(1-eps)) = prod sech(alpha)
    short = ref.AdaBoostScratch(n_estimators=25)
    short.fit(X_tr, y_tr)
    for k in CHECKPOINTS:
        print(f"  rounds={k:>3}  train={train[k - 1]:.4f}  test={test[k - 1]:.4f}")
    return {
        "train": train, "test": test, "alphas": np.array(model.alphas),
        "first_perfect": int(np.argmax(train == 1.0)) + 1 if (train == 1.0).any() else None,
        "settled_train": settled(train, train[-1]),
        "drops": int((np.diff(train) < 0).sum()),
        "loss_rises": int((np.diff(exp_loss) > 0).sum()),
        "bound_gap": float(np.abs(exp_loss - bound).max()),
        "bound_holds": bool(((1 - train) <= bound + 1e-12).all()),
        "prefix_gap": abs(short.accuracy(X_tr, y_tr) - train[24]),
    }


def verify(result):
    train, test, alphas = result["train"], result["test"], result["alphas"]
    best_test = int(np.argmax(test)) + 1
    late = test[19:]
    return [
        practice.Check(
            "ANSWER: training accuracy converges to 1.0, test accuracy by round ~20",
            result["first_perfect"] is not None and train[-1] == 1.0
            and late.max() - late.min() <= 0.04,
            f"train first reaches 1.0000 at round {result['first_perfect']} and holds it from "
            f"round {result['settled_train']} to {ROUNDS}; test reaches {test[19]:.4f} at round "
            f"20 and then only moves between {late.min():.4f} and {late.max():.4f} (one test "
            f"point is 0.0125), peaking at {test.max():.4f} on round {best_test}. 'Converged' "
            "is round ~20 for what generalises and round "
            f"{result['settled_train']} for what memorises",
        ),
        practice.Check(
            "FINDING: training accuracy is not monotone; the loss AdaBoost minimises is",
            result["drops"] >= 10 and result["loss_rises"] == 0 and result["bound_holds"],
            f"training accuracy falls on {result['drops']} of {ROUNDS - 1} rounds, while the "
            f"mean exponential loss rises on {result['loss_rises']}: each round minimises "
            "exp(-y F), not the 0/1 error, and the training error sits under the bound "
            "prod 2 sqrt(eps (1 - eps)) on every round",
        ),
        practice.Check(
            "FINDING: round 2 changes no prediction",
            alphas[1] < alphas[0] and train[1] == train[0],
            f"alpha_1 = {alphas[0]:.4f} > alpha_2 = {alphas[1]:.4f}, so sign(a1 h1 + a2 h2) is "
            f"sign(h1) everywhere: train accuracy is {train[0]:.4f} after both rounds 1 and 2. "
            "Two weighted voters are a dictatorship; the curve's first step is at round 3",
        ),
        practice.Check(
            "CONTROL: the tracked curve is the lesson's model, and its loss is exact",
            result["prefix_gap"] == 0 and result["bound_gap"] < 1e-12,
            f"AdaBoostScratch(n_estimators=25).accuracy matches the staged curve at round 25 "
            f"exactly; mean exp-loss equals prod sech(alpha_t) within {result['bound_gap']:.1e}, "
            "so the lesson's weight update is exact AdaBoost",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
