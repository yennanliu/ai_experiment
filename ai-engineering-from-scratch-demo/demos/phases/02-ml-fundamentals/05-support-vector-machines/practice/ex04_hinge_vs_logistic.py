"""Exercise 4 — hinge against logistic loss: how many points shape the boundary?

    Compare hinge loss vs logistic loss on the same dataset. Train a linear SVM
    and logistic regression. Count how many training points contribute to each
    model's decision boundary (support vectors vs all points).

Reading of the exercise: "contribute" has to be defined by the gradient, since
that is the only channel through which a point moves the boundary. For hinge the
answer is exact — the update branches on y·f(x) >= 1, so points outside the margin
contribute *only* weight decay and nothing data-dependent: 35 of 200 here. For
logistic the per-point gradient is (p − y)·x, which is never exactly zero, so all
200 contribute. Check 4 measures the smallest |p − y| over the dataset to show
the difference is exact rather than a matter of degree.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "05-support-vector-machines"
LOGIT_LESSON = "03-logistic-regression"
SEED, N, EPOCHS = 42, 200, 800


def _logistic(logit, X, y):
    """Fit logistic regression and return its per-point gradient magnitudes |p − y|."""
    y01 = [1 if v == 1 else 0 for v in y]
    model = logit.LogisticRegression(n_features=2, learning_rate=0.1)
    model.fit(X, y01, epochs=2000, print_every=10 ** 9)
    return model, y01, [abs(model.predict_proba(x) - t) for x, t in zip(X, y01)]


def solve():
    svm = parity.load_reference(PHASE, LESSON, "svm")
    logit = parity.load_reference(PHASE, LOGIT_LESSON, "logistic_regression")
    X, y = svm.generate_noisy_data(n_samples=N, noise=0.5, seed=SEED)
    model = svm.LinearSVM(lr=0.01, lambda_param=0.01, n_epochs=EPOCHS)
    model.fit(X, y)
    margins = [yi * (svm.dot(model.w, x) + model.b) for yi, x in zip(y, X)]

    lg, y01, residuals = _logistic(logit, X, y)
    return {
        "n": N,
        "svm_accuracy": svm.accuracy(y, model.predict(X)),
        "logit_accuracy": lg.accuracy(X, y01),
        "support": sum(1 for m in margins if m < 1.0),
        "outside_margin": sum(1 for m in margins if m >= 1.0),
        "exact_zero_residuals": sum(1 for r in residuals if r == 0.0),
        "min_residual": min(residuals),
        "under_1e3": sum(1 for r in residuals if r < 1e-3),
    }


def verify(result):
    return [
        practice.Check("both models reach the same accuracy on the same data",
                       abs(result["svm_accuracy"] - result["logit_accuracy"]) < 0.02,
                       f"SVM {result['svm_accuracy']:.1%}, logistic "
                       f"{result['logit_accuracy']:.1%} on {result['n']} points — so the "
                       f"comparison is about *which* points matter, not which model wins"),
        practice.Check(f"ANSWER: the hinge boundary depends on {result['support']} points",
                       result["support"] + result["outside_margin"] == result["n"]
                       and result["support"] < result["n"] // 4,
                       f"{result['support']} of {result['n']} have y·f(x) < 1; the other "
                       f"{result['outside_margin']} contribute only weight decay, with no "
                       f"data term at all"),
        practice.Check(f"ANSWER: the logistic boundary depends on all {result['n']}",
                       result["exact_zero_residuals"] == 0,
                       f"the per-point gradient is (p − y)·x and exactly "
                       f"{result['exact_zero_residuals']} points have p − y = 0. Every "
                       f"point pulls, however faintly"),
        practice.Check("…and 'faintly' is not 'not at all'",
                       0 < result["min_residual"] < 1e-5,
                       f"smallest |p − y| = {result['min_residual']:.3e}, and "
                       f"{result['under_1e3']} of {result['n']} points are under 1e-3. "
                       f"Small, but the sigmoid never saturates exactly, so the sum over "
                       f"200 faint pulls is what places the boundary"),
        practice.Check("MECHANISM: one loss has a flat region, the other does not",
                       result["outside_margin"] > 0,
                       f"hinge is max(0, 1 − y·f(x)): identically zero once the margin is "
                       f"met, so its derivative there is zero and the point drops out. "
                       f"Logistic is log(1 + exp(−y·f(x))): positive everywhere, so its "
                       f"derivative is never zero. Sparsity of support is a property of "
                       f"the *loss*, not of the optimiser or the data"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
