"""Exercise 2 — sweep C on noisy data: wide margin to narrow margin.

    Vary C from 0.001 to 1000 on a noisy dataset. Plot the decision boundary for
    each C value. Observe the transition from wide margin (underfitting) to
    narrow margin (overfitting).

Reading of the exercise: the lesson's LinearSVM takes `lambda_param`, not C, and
they are reciprocal, so the sweep runs λ = 1/C.

Two things the exercise's suggested range does not survive. **C = 0.001 does not
underfit, it diverges**: λ = 1000 with the default lr = 0.01 makes the weight
decay factor (1 − lr·λ) = −9, so w flips sign and grows every step until it is
nan. Stability needs lr·λ < 1, i.e. C > lr — check 2 records that, and the
remaining checks run over the stable range.

And **the largest C does not overfit** (check 5). A linear model in 2D has too
little capacity, whatever C does; the overfitting half of the exercise needs a
kernel.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "05-support-vector-machines"
SEED, EPOCHS = 42, 600
C_VALUES = (0.001, 0.1, 1.0, 100.0, 1000.0)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "svm")
    X, y = ref.generate_noisy_data(n_samples=250, noise=0.6, seed=SEED)
    X_train, y_train, X_test, y_test = ref.train_test_split(X, y, seed=SEED)
    rows = {}
    for c in C_VALUES:
        model = ref.LinearSVM(lr=0.01, lambda_param=1.0 / c, n_epochs=EPOCHS)
        model.fit(X_train, y_train)
        norm = ref.vec_norm(model.w)
        scores = [ref.dot(model.w, x) + model.b for x in X_train]
        rows[c] = {"margin": 1.0 / norm if norm else float("inf"),
                   "norm": norm,
                   "train": ref.accuracy(y_train, model.predict(X_train)),
                   "test": ref.accuracy(y_test, model.predict(X_test)),
                   "n_support": sum(1 for yi, s in zip(y_train, scores) if yi * s < 1)}
    stable = [c for c in C_VALUES if rows[c]["margin"] == rows[c]["margin"]
              and rows[c]["margin"] != float("inf")]
    best = max(stable, key=lambda c: rows[c]["test"])
    return {"rows": rows, "best": best, "n_train": len(y_train),
            "stable": stable, "lr": 0.01}


def verify(result):
    rows, best, stable = result["rows"], result["best"], result["stable"]
    diverged = [c for c in C_VALUES if c not in stable]
    lo, hi = rows[stable[0]], rows[stable[-1]]
    margins = [rows[c]["margin"] for c in stable]
    supports = [rows[c]["n_support"] for c in stable]
    return [
        practice.Check(f"all {len(C_VALUES)} values of C trained",
                       len(rows) == len(C_VALUES),
                       "; ".join(f"C={c:g}: margin {rows[c]['margin']:.3f}, "
                                 f"train {rows[c]['train']:.1%}, test {rows[c]['test']:.1%}"
                                 for c in C_VALUES)),
        practice.Check(f"FINDING: C={C_VALUES[0]:g} diverges — it does not underfit",
                       diverged == [C_VALUES[0]],
                       f"λ = 1/C = {1 / C_VALUES[0]:g} with lr = {result['lr']:g} gives a "
                       f"decay factor (1 − lr·λ) = {1 - result['lr'] / C_VALUES[0]:g}, so w "
                       f"flips sign and grows each step until ‖w‖ is nan. Stability needs "
                       f"lr·λ < 1, i.e. C > {result['lr']:g}"),
        practice.Check("ANSWER: over the stable range the margin narrows monotonically",
                       all(a > b for a, b in zip(margins, margins[1:])),
                       f"1/‖w‖ falls {margins[0]:.3f} -> {margins[-1]:.4f} across "
                       f"C = {stable[0]:g}..{stable[-1]:g}, a factor of "
                       f"{margins[0] / margins[-1]:.0f}"),
        practice.Check("…and the number of support vectors falls with it",
                       supports[0] > supports[-1],
                       f"{supports[0]} of {result['n_train']} points inside the margin at "
                       f"C={stable[0]:g}, {supports[-1]} at C={stable[-1]:g}. A wide "
                       f"margin catches most of the data; a narrow one catches almost none"),
        practice.Check(f"the smallest stable C underfits — it barely beats chance",
                       lo["train"] < 0.75,
                       f"C={stable[0]:g}: train {lo['train']:.1%}, test {lo['test']:.1%} "
                       f"with a margin of {lo['margin']:.2f}. λ = {1 / stable[0]:g} "
                       f"dominates the hinge term, so w is pulled toward zero regardless "
                       f"of the data"),
        practice.Check("FINDING: the largest C does not overfit here",
                       hi["test"] >= lo["test"] and abs(hi["train"] - hi["test"]) < 0.15,
                       f"C={stable[-1]:g}: train {hi['train']:.1%}, test "
                       f"{hi['test']:.1%}, gap {hi['train'] - hi['test']:+.1%}. Best test "
                       f"accuracy is at C={best:g}. A *linear* model has too little "
                       f"capacity to overfit 2D data however hard C pushes — the "
                       f"underfitting half of the exercise is visible, the overfitting "
                       f"half needs a kernel"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
