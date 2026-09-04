"""Exercise 5 — weighted against unweighted KNN regression on sin(x) + noise.

    Build a weighted KNN regressor for y = sin(x) + noise. Compare it with
    unweighted KNN for K=3, 10, 30. Show that weighting produces smoother
    predictions, especially for large K.

Reading of the exercise: the claim is **false as stated**, and two measurement
choices are needed before that can be said honestly.

Queries must be **held out**. The weight is 1/(d + 1e-10), so a query that is
also a training point gets weight 1e10 and the prediction collapses to that
point's own label — identical output at every K, and an RMSE equal to the noise
level. Predicting on the training set makes the comparison meaningless.

And "smoother" needs a metric: mean |Δ prediction| between neighbouring x. By
that measure weighting is **rougher at every K** (check 3). What weighting
actually buys is accuracy at large K (check 4), where unweighted averaging drags
in distant points at full strength.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "06-knn-and-distances"
SEED, N, NOISE = 42, 200, 0.2
KS = (3, 10, 30)


def _rmse(predictions, truth):
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(predictions, truth)) / len(truth))


def _score(ref, k, weighted, X_train, y_train, X_test, truth):
    model = ref.KNN(k=k, weighted=weighted, task="regression")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {"rmse": _rmse(predictions, truth),
            "roughness": sum(abs(b - a) for a, b in zip(predictions, predictions[1:]))
                         / (len(predictions) - 1)}


def _self_query(ref, k, X_train, y_train):
    """Weighted KNN asked about its own training points — the degenerate case."""
    model = ref.KNN(k=k, weighted=True, task="regression")
    model.fit(X_train, y_train)
    return _rmse(model.predict(X_train), y_train)


def _split(rng):
    """A held-out split: querying the training set degenerates (see check 2)."""
    xs = sorted(rng.uniform(-math.pi, math.pi) for _ in range(N))
    y = [math.sin(x) + rng.gauss(0, NOISE) for x in xs]
    order = list(range(N))
    rng.shuffle(order)
    train, test = sorted(order[:int(0.7 * N)]), sorted(order[int(0.7 * N):])
    return ([[xs[i]] for i in train], [y[i] for i in train],
            [[xs[i]] for i in test], [math.sin(xs[i]) for i in test], test)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "knn")
    X_train, y_train, X_test, truth, test = _split(random.Random(SEED))
    rows = {k: {"plain": _score(ref, k, False, X_train, y_train, X_test, truth),
                "weighted": _score(ref, k, True, X_train, y_train, X_test, truth)}
            for k in KS}
    degenerate = {k: _self_query(ref, k, X_train, y_train) for k in KS}
    return {"rows": rows, "degenerate": degenerate, "n_test": len(test)}


def verify(result):
    rows, deg = result["rows"], result["degenerate"]
    rougher = all(rows[k]["weighted"]["roughness"] > rows[k]["plain"]["roughness"]
                  for k in KS)
    big = KS[-1]
    return [
        practice.Check(f"both variants run at K={KS} on {result['n_test']} held-out points",
                       len(rows) == len(KS),
                       "; ".join(f"K={k}: plain rmse {rows[k]['plain']['rmse']:.4f}, "
                                 f"weighted {rows[k]['weighted']['rmse']:.4f}" for k in KS)),
        practice.Check("querying the training set makes weighted KNN degenerate",
                       max(deg.values()) < 1e-6,
                       f"RMSE against the *training labels* is "
                       f"{max(deg.values()):.2e} at every K — the weight 1/(d + 1e-10) is "
                       f"1e10 at distance zero, so the prediction is the query point's own "
                       f"label and K stops mattering. That is why the comparison uses a "
                       f"held-out split"),
        practice.Check("ANSWER: weighting is ROUGHER at every K, not smoother",
                       rougher,
                       "mean |Δ prediction|: " + "; ".join(
                           f"K={k}: plain {rows[k]['plain']['roughness']:.4f} vs weighted "
                           f"{rows[k]['weighted']['roughness']:.4f}" for k in KS)
                       + " — the nearest neighbour dominates the weighted average, so the "
                         "prediction tracks it and jumps when it changes"),
        practice.Check(f"what weighting buys is accuracy at large K",
                       rows[big]["weighted"]["rmse"] < rows[big]["plain"]["rmse"],
                       f"at K={big}, RMSE {rows[big]['weighted']['rmse']:.4f} weighted "
                       f"against {rows[big]['plain']['rmse']:.4f} plain — unweighted "
                       f"averaging pulls in all {big} neighbours at full strength, including "
                       f"ones far enough away that sin(x) has moved"),
        practice.Check("…and it costs accuracy at small K, where all neighbours are close",
                       rows[KS[0]]["weighted"]["rmse"] > rows[KS[0]]["plain"]["rmse"],
                       f"at K={KS[0]}, RMSE {rows[KS[0]]['weighted']['rmse']:.4f} weighted "
                       f"against {rows[KS[0]]['plain']['rmse']:.4f} plain. With 3 nearby "
                       f"neighbours the plain mean averages away noise; weighting refuses "
                       f"to, and inherits the nearest point's noise instead"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
