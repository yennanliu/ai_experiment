"""Exercise 1 — find the support vectors and check they are the closest points.

    Generate a 2D linearly separable dataset. Train your LinearSVM and identify
    the support vectors. Verify that the support vectors are the points closest
    to the decision boundary.

Reading of the exercise: "the points closest to the decision boundary" is exactly
right on separable data, and for a reason worth stating — distance is
|f(x)|/‖w‖ with ‖w‖ constant, so ranking by distance and ranking by |f(x)| are the
same ranking. The margin set {y·f(x) < 1} is therefore precisely a prefix of the
distance order, and check 4 confirms all 5 coincide.

Check 5 finds where the two part company: on non-separable data 12 points are
misclassified, and a misclassified point has y·f(x) < 0 while its |f(x)| — its
distance — can be large. There the margin set and the nearest-k set differ.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "05-support-vector-machines"
SEED, N, EPOCHS = 42, 120, 800


def _geometry(ref, model, X, y):
    """Margins y·f(x), distances |f(x)|/‖w‖, and ‖w‖."""
    scores = [ref.dot(model.w, x) + model.b for x in X]
    norm = ref.vec_norm(model.w)
    return ([yi * s for yi, s in zip(y, scores)],
            [abs(s) / norm for s in scores], norm)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "svm")
    X, y = ref.generate_linear_data(n_samples=N, margin=1.0, seed=SEED)
    model = ref.LinearSVM(lr=0.01, lambda_param=0.01, n_epochs=EPOCHS)
    model.fit(X, y)
    margins, distances, norm = _geometry(ref, model, X, y)
    on_margin = [i for i, m in enumerate(margins) if m < 1.0]
    ranked = sorted(range(len(X)), key=lambda i: distances[i])
    return {"accuracy": ref.accuracy(y, model.predict(X)),
            "n": len(X), "on_margin": len(on_margin),
            "overlap": len(set(on_margin) & set(ranked[:len(on_margin)])),
            "max_margin_distance": max(distances[i] for i in on_margin),
            "min_other_distance": min(distances[i] for i in range(len(X))
                                      if i not in set(on_margin)),
            "all_correct": all(m > 0 for m in margins),
            "norm": norm, "noisy": _noisy_case(ref)}


def _noisy_case(ref):
    """The same comparison where the data is not separable."""
    X, y = ref.generate_noisy_data(n_samples=200, noise=0.5, seed=SEED)
    model = ref.LinearSVM(lr=0.01, lambda_param=0.01, n_epochs=EPOCHS)
    model.fit(X, y)
    margins, distances, _ = _geometry(ref, model, X, y)
    on_margin = [i for i, m in enumerate(margins) if m < 1.0]
    ranked = sorted(range(len(X)), key=lambda i: distances[i])
    return {"on_margin": len(on_margin),
            "misclassified": sum(1 for m in margins if m < 0),
            "overlap": len(set(on_margin) & set(ranked[:len(on_margin)]))}


def verify(result):
    return [
        practice.Check("the data is separable and the SVM separates it",
                       result["accuracy"] == 1.0 and result["all_correct"],
                       f"{result['accuracy']:.0%} accuracy on all {result['n']} points, "
                       f"every margin y·f(x) > 0"),
        practice.Check(f"{result['on_margin']} of {result['n']} points sit inside the margin",
                       0 < result["on_margin"] < result["n"] // 3,
                       f"y·f(x) < 1 for {result['on_margin']} points — these are the ones "
                       f"the hinge loss is still paying for, and the only ones whose "
                       f"position affects w"),
        practice.Check("every margin point is nearer the boundary than every other point",
                       result["max_margin_distance"] < result["min_other_distance"],
                       f"the furthest margin point is at {result['max_margin_distance']:.4f} "
                       f"in |f(x)|/‖w‖ and the nearest non-margin point at "
                       f"{result['min_other_distance']:.4f} — the two groups do not "
                       f"interleave at all"),
        practice.Check("ANSWER: the two sets coincide exactly, and necessarily",
                       result["overlap"] == result["on_margin"],
                       f"all {result['overlap']} of {result['on_margin']} margin points are "
                       f"also the {result['on_margin']} nearest. Distance is |f(x)|/‖w‖ with "
                       f"‖w‖ = {result['norm']:.3f} constant, so the two orderings are the "
                       f"same ordering — this is an identity, not a coincidence"),
        practice.Check("FINDING: they part company once the data is not separable",
                       result["noisy"]["overlap"] < result["noisy"]["on_margin"],
                       f"on the noisy set, {result['noisy']['overlap']} of "
                       f"{result['noisy']['on_margin']} margin points are among the nearest, "
                       f"with {result['noisy']['misclassified']} points misclassified. A "
                       f"misclassified point has y·f(x) < 0 while its *distance* |f(x)| can "
                       f"be large, so 'closest to the boundary' stops describing the "
                       f"support set"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
