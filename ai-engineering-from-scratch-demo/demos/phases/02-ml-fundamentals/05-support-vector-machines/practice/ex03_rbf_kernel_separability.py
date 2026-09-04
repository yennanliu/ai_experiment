"""Exercise 3 — circular classes: linear SVM fails, RBF makes them separable.

    Create a dataset where class boundaries are circular (not linear). Show that
    a linear SVM fails. Compute the RBF kernel matrix and show that the classes
    become separable in the kernel-induced feature space.

Reading of the exercise: separability in the lifted space is testable without
forming that space — the kernel matrix carries everything needed, and a linear
SVM trained on its rows is a linear model in exactly that feature space.

Two things the exercise leaves out. Within-class kernel similarity exceeding
between-class is necessary but not sufficient: at γ=50 that ratio is 36x while
both values are near zero, because the kernel has become the identity matrix and
every point resembles only itself. So the criterion has to be **held-out
accuracy**, and check 5 sweeps γ to show it is U-shaped — 35.0% / 92.5% / 80.0%
at γ = 0.01 / 5 / 50, the last with a 0.19 train-test gap.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "05-support-vector-machines"
SEED, N, EPOCHS = 42, 200, 600
GAMMAS = (0.01, 5.0, 50.0)


def fit_lifted(ref, X_train, y_train, X_test, gamma):
    K_train = ref.compute_kernel_matrix(X_train, ref.rbf_kernel, gamma=gamma)
    K_test = [[ref.rbf_kernel(x, z, gamma=gamma) for z in X_train] for x in X_test]
    model = ref.LinearSVM(lr=0.01, lambda_param=0.01, n_epochs=EPOCHS)
    model.fit(K_train, y_train)
    return model, K_train, K_test


def within_similarity(K, y):
    pairs = [K[i][j] for i in range(len(y)) for j in range(i + 1, len(y))
             if y[i] == y[j]]
    return sum(pairs) / len(pairs)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "svm")
    X, y = ref.generate_circular_data(n_samples=N, seed=SEED)
    X_train, y_train, X_test, y_test = ref.train_test_split(X, y, seed=SEED)
    raw = ref.LinearSVM(lr=0.01, lambda_param=0.01, n_epochs=EPOCHS)
    raw.fit(X_train, y_train)
    rows = {}
    for gamma in GAMMAS:
        model, K_train, K_test = fit_lifted(ref, X_train, y_train, X_test, gamma)
        rows[gamma] = {
            "train": ref.accuracy(y_train, model.predict(K_train)),
            "test": ref.accuracy(y_test, model.predict(K_test)),
            "within": within_similarity(K_train, y_train),
            "diagonal": K_train[0][0],
        }
    best = max(GAMMAS, key=lambda g: rows[g]["test"])
    return {"rows": rows, "best": best,
            "raw_test": ref.accuracy(y_test, raw.predict(X_test)),
            "baseline": max(y_test.count(1), y_test.count(-1)) / len(y_test),
            "n": len(y)}


def verify(result):
    rows, best = result["rows"], result["best"]
    hi = rows[GAMMAS[-1]]
    tests = [rows[g]["test"] for g in GAMMAS]
    return [
        practice.Check("ANSWER: the linear SVM cannot separate circular classes",
                       result["raw_test"] < result["baseline"] + 0.15,
                       f"test accuracy {result['raw_test']:.1%} against a majority baseline "
                       f"of {result['baseline']:.1%} — no line separates an inner disc from "
                       f"an outer ring"),
        practice.Check("the kernel matrix is a valid similarity: 1 on the diagonal",
                       all(abs(rows[g]["diagonal"] - 1.0) < 1e-12 for g in GAMMAS),
                       "RBF is exp(−γ‖x−z‖²) ∈ (0, 1], so k(x,x) = 1 exactly at every γ"),
        practice.Check(f"ANSWER: a linear SVM on the kernel rows separates them ({best:g})",
                       rows[best]["test"] > result["raw_test"] + 0.2,
                       f"γ={best:g} reaches {rows[best]['test']:.1%} test against the raw "
                       f"model's {result['raw_test']:.1%}. Same model, same optimiser — "
                       f"only the representation changed"),
        practice.Check("FINDING: held-out accuracy is U-shaped in γ, not increasing",
                       tests[0] < tests[1] > tests[-1],
                       "test accuracy by γ: " + ", ".join(
                           f"{g:g}: {rows[g]['test']:.1%}" for g in GAMMAS)
                       + f" — so 'use RBF' is not the recipe; γ is"),
        practice.Check("…and the high-γ failure is overfitting, with the kernel near identity",
                       hi["within"] < 0.02 and hi["train"] - hi["test"] > 0.15,
                       f"at γ={GAMMAS[-1]:g}, mean within-class similarity is "
                       f"{hi['within']:.4f} — every point resembles only itself — and train "
                       f"{hi['train']:.1%} against test {hi['test']:.1%}, a "
                       f"{hi['train'] - hi['test']:+.1%} gap. A within/between *ratio* looks "
                       f"excellent here precisely because both terms are near zero, which "
                       f"is why it cannot be the criterion"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
