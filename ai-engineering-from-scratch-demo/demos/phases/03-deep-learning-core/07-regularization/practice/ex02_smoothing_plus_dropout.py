"""Exercise 2 — label smoothing crossed with dropout, four ways.

    Implement label smoothing from lesson 05 combined with dropout from this
    lesson. Train with four configurations: neither, dropout only, label
    smoothing only, both. Measure the final train-test accuracy gap for each.
    Which combination gives the smallest gap?

Reading of the exercise: for two classes, label smoothing is a softened target fed to the
same binary cross-entropy, so no code is forked — check 1 shows the loss equals lesson
05's `label_smoothed_cce` and that lesson 07's `backward` already differentiates it. "The
final train-test accuracy gap" then has two readings that give opposite answers: check 2
measures both halves in eval mode, check 4 reads it off the lesson's own printout.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "07-regularization"
ALPHA, DROP, SEEDS = 0.1, 0.3, (1, 2, 3, 4, 5, 6)
CONFIGS = (("neither", 0.0, 0.0), ("dropout", DROP, 0.0),
           ("smoothing", 0.0, ALPHA), ("both", DROP, ALPHA))


def bce(p, target):
    return -(target * math.log(p) + (1 - target) * math.log(1 - p))


def _table(rows, key, unit="pp"):
    return ", ".join(f"{n} {rows[n][key]:+.2f} {unit}" for n, _p, _a in CONFIGS)


def run(ref, data, drop_p, alpha, seed):
    """Train one configuration; report eval-mode accuracy on both halves."""
    net = ref.RegularizedNetwork(hidden_size=16, lr=0.05, dropout_p=drop_p)
    train = [(x, y * (1 - alpha) + alpha / 2) for x, y in data[:150]]
    random.seed(seed)
    with parity.quiet():
        _, shown_train, _, shown_test = net.train_model(train, data[150:], epochs=300)[-1]
    train_acc, test_acc = net.evaluate(data[:150])[1], net.evaluate(data[150:])
    return {"gap": train_acc - test_acc[1], "test": test_acc[1], "test_loss": test_acc[0],
            "shown_gap": shown_train - shown_test,
            "top": max(net.forward(x, training=False) for x, _ in data[150:])}


def average(rows):
    return {key: statistics.mean(r[key] for r in rows) for key in rows[0]}


def one_gradient(ref, target, step=1e-6, x=(0.7, -0.3)):
    """The lesson's own backward against a central difference, for a fractional target."""
    net = ref.RegularizedNetwork(hidden_size=16, lr=1.0)
    base = net.b2
    net.b2 = base + step
    high = bce(net.forward(list(x)), target)
    net.b2 = base - step
    finite = (high - bce(net.forward(list(x)), target)) / (2 * step)
    net.b2 = base
    net.forward(list(x))
    net.backward(target)
    return abs((base - net.b2) - finite) / abs(finite)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    l05 = parity.load_reference(PHASE, "05-loss-functions", "main")
    data = ref.make_circle_data(n=300, seed=42)
    raw = {name: [run(ref, data, p, a, s) for s in (SEEDS if p else (1,))]
           for name, p, a in CONFIGS}
    return {"rows": {name: average(runs) for name, runs in raw.items()},
            "spread": {name: statistics.pstdev([r["gap"] for r in runs])
                       for name, runs in raw.items() if len(runs) > 1},
            "grad": max(one_gradient(ref, t) for t in (0.05, 0.5, 0.95)),
            "loss_gap": max(abs(l05.label_smoothed_cce([0.0, z], 1, 2, alpha=ALPHA)
                                - bce(1 / (1 + math.exp(-z)), 1 - ALPHA / 2))
                            for z in (-2.0, 0.7, 3.0))}


def verify(result):
    rows, spread = result["rows"], result["spread"]
    best = min(rows, key=lambda n: rows[n]["gap"])
    shown_best = min(rows, key=lambda n: rows[n]["shown_gap"])
    return [
        practice.Check("CONTROL: two-class smoothing is BCE on a softened target, and the lesson "
                       "already differentiates it",
                       result["loss_gap"] < 1e-12 and result["grad"] < 1e-8,
                       f"lesson 05's label_smoothed_cce(alpha={ALPHA}) equals BCE at target "
                       f"{1 - ALPHA / 2} to {result['loss_gap']:.1e} over three logits, and its "
                       f"derivative matches a central difference in b2 to {result['grad']:.1e} at "
                       f"t = 0.05, 0.5, 0.95 — so no code is forked here"),
        practice.Check("ANSWER: 'neither' gives the smallest gap — every regularizer widens it",
                       best == "neither",
                       f"eval-mode train minus test accuracy — {_table(rows, 'gap')}; the dropout rows "
                       "vary by " + ", ".join(f"{n} ±{s:.2f}" for n, s in spread.items())
                       + " over 6 mask seeds"),
        practice.Check("FINDING: regularization does not close the gap badly — it loses test accuracy",
                       rows["neither"]["test"] > max(rows[n]["test"] for n in rows if n != "neither"),
                       f"test accuracy {_table(rows, 'test', '%')}. The unregularized net is already "
                       f"at {rows['neither']['gap']:+.2f} pp on 150 test points — there is no "
                       f"overfitting to remove, so every constraint added is capacity lost"),
        practice.Check("FINDING: read off the lesson's own printout the answer reverses",
                       shown_best == "both" and rows["both"]["shown_gap"] < 0,
                       f"train_model grades the training set through the dropout mask, so its printed "
                       f"gap= line reads {_table(rows, 'shown_gap')} and picks {shown_best} at "
                       f"{rows['both']['shown_gap']:+.2f} pp — a negative gap that is an artefact"),
        practice.Check("MECHANISM: smoothing pays for the gap in confidence, and log-loss bills it",
                       rows["smoothing"]["test_loss"] > 2 * rows["neither"]["test_loss"]
                       and 1 - rows["smoothing"]["top"] > 1e5 * (1 - rows["neither"]["top"]),
                       f"top test prediction {rows['neither']['top']:.10f} unsmoothed against "
                       f"{rows['smoothing']['top']:.4f} smoothed — a ceiling of {1 - ALPHA / 2} it "
                       f"still overshoots; against hard labels test loss is "
                       f"{rows['smoothing']['test_loss']:.4f} vs {rows['neither']['test_loss']:.4f}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
