"""Exercise 1 — batch vs stochastic vs mini-batch GD: speed and smoothness.

    Implement batch gradient descent, stochastic gradient descent (SGD), and
    mini-batch gradient descent. Compare convergence speed on the same dataset.
    Which converges fastest? Which has the smoothest cost curve?

Reading of the exercise: "fastest" is ambiguous and the readings disagree, so both
are measured — per **epoch** SGD wins (1 against 38), per **update** batch does
(60 against 12,000). "Smoothest" is likewise scored twice: mean |Δcost| barely
separates them (0.172/0.173/0.187) since within-epoch noise averages out by the
epoch-end reading, while counting cost *increases* is decisive (0/26/28). Check 2
records the cost of SGD's speed — furthest from the true parameters. See README.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "02-linear-regression"
SEED, N, EPOCHS, LR = 42, 200, 60, 0.05
TRUE_W, TRUE_B = 2.5, -1.0


def make_data(rng):
    xs = [rng.uniform(-3, 3) for _ in range(N)]
    return xs, [TRUE_W * x + TRUE_B + rng.gauss(0, 0.5) for x in xs]


def cost(xs, ys, w, b):
    return sum((w * x + b - y) ** 2 for x, y in zip(xs, ys)) / (2 * len(xs))


def descend(xs, ys, batch_size, rng):
    """One implementation; batch_size = n is batch GD, 1 is SGD, else mini-batch."""
    w, b, curve, updates = 0.0, 0.0, [cost(xs, ys, 0.0, 0.0)], 0
    order = list(range(len(xs)))
    for _ in range(EPOCHS):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            errors = [(w * xs[i] + b - ys[i]) for i in chunk]
            w -= LR * sum(e * xs[i] for e, i in zip(errors, chunk)) / len(chunk)
            b -= LR * sum(errors) / len(chunk)
            updates += 1
        curve.append(cost(xs, ys, w, b))
    return {"w": w, "b": b, "curve": curve, "updates": updates}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "linear_regression")
    rng = random.Random(SEED)
    xs, ys = make_data(rng)
    variants = {"batch": N, "mini-batch (32)": 32, "SGD": 1}
    rows = {}
    for label, size in variants.items():
        run = descend(xs, ys, size, random.Random(SEED))
        curve = run["curve"]
        rises = sum(1 for a, b in zip(curve, curve[1:]) if b > a)
        rows[label] = {
            "final_cost": curve[-1], "updates": run["updates"],
            "roughness": sum(abs(b - a) for a, b in zip(curve, curve[1:])) / (len(curve) - 1),
            "rises": rises,
            "epochs_to_target": next((i for i, c in enumerate(curve)
                                      if c < 1.05 * min(curve)), len(curve)),
            "w": run["w"], "b": run["b"],
        }
    # the lesson's own batch implementation, as a cross-check
    theirs = ref.LinearRegression(learning_rate=LR)
    theirs.fit(xs, ys, epochs=EPOCHS, print_every=EPOCHS + 1)
    return {"rows": rows, "reference": {"w": theirs.w, "b": theirs.b},
            "true": (TRUE_W, TRUE_B)}


def _distance(row) -> float:
    return abs(row["w"] - TRUE_W) + abs(row["b"] - TRUE_B)


def verify(result):
    rows = result["rows"]
    batch, mini, sgd = rows["batch"], rows["mini-batch (32)"], rows["SGD"]
    return [
        practice.Check("all three recover the true slope",
                       all(abs(r["w"] - TRUE_W) < 0.1 for r in rows.values()),
                       ", ".join(f"{k}: w={v['w']:.3f}, b={v['b']:.3f}"
                                 for k, v in rows.items())
                       + f" against the true ({TRUE_W}, {TRUE_B})"),
        practice.Check("…but SGD lands furthest from the true parameters",
                       _distance(sgd) > _distance(batch) and _distance(sgd) > _distance(mini),
                       f"|Δ| to ({TRUE_W}, {TRUE_B}): batch {_distance(batch):.4f}, "
                       f"mini-batch {_distance(mini):.4f}, SGD {_distance(sgd):.4f}"),
        practice.Check("and batch agrees with the lesson's own implementation",
                       abs(batch["w"] - result["reference"]["w"]) < 0.05,
                       f"lesson's LinearRegression -> w={result['reference']['w']:.4f}, "
                       f"b={result['reference']['b']:.4f}"),
        practice.Check("ANSWER: 'fastest' has two answers, per epoch and per update",
                       sgd["epochs_to_target"] < batch["epochs_to_target"]
                       and sgd["updates"] > batch["updates"],
                       f"epochs to within 5% of best: SGD {sgd['epochs_to_target']}, "
                       f"mini-batch {mini['epochs_to_target']}, batch "
                       f"{batch['epochs_to_target']} — but SGD used {sgd['updates']:,} "
                       f"updates against {batch['updates']}"),
        practice.Check("ANSWER: batch is smoothest — and only batch is monotone",
                       batch["rises"] == 0 and sgd["rises"] > 0
                       and batch["roughness"] < sgd["roughness"],
                       f"cost increases: batch {batch['rises']} of {EPOCHS}, mini-batch "
                       f"{mini['rises']}, SGD {sgd['rises']}; mean |Δcost| "
                       f"{batch['roughness']:.5f} / {mini['roughness']:.5f} / "
                       f"{sgd['roughness']:.5f} — see the README"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
