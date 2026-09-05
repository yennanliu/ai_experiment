"""Exercise 3 — BatchNorm between the hidden layer and the activation.

    Add a BatchNorm layer between the hidden layer and the activation in your
    circle-dataset network. Train with and without BatchNorm at learning rates
    0.01, 0.05, and 0.1. BatchNorm should allow stable training at higher learning
    rates where the vanilla network diverges.

Reading of the exercise: the lesson's `RegularizedNetwork` trains one sample at a time, so
checks 1-2 ask what its own `BatchNorm` does when inserted there — the answer is why this
solution has to write a batched trainer and a BatchNorm backward of its own. Check 3 runs the
lesson's network unmodified at the three rates named; checks 4-5 run the batched comparison
the exercise intends, at those rates and past them.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "07-regularization"
H, EPS, BATCH, EPOCHS, SPLIT = 16, 1e-5, 16, 200, 150
RATES, HIGH = (0.01, 0.05, 0.1), (1.0, 2.0, 3.0, 4.0, 8.0)


def init(seed=0) -> dict:
    """The lesson's own initialisation, plus BatchNorm's gamma and beta."""
    random.seed(seed)
    return {"w1": [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(H)], "b1": [0.0] * H,
            "w2": [random.gauss(0, 0.5) for _ in range(H)], "b2": 0.0, "g": [1.0] * H,
            "be": [0.0] * H}


def forward(ref, p, xs, bn) -> tuple:
    """x -> z1 -> [the lesson's own BatchNorm] -> ReLU -> z2 -> sigmoid."""
    z1 = [[sum(p["w1"][i][j] * x[j] for j in range(2)) + p["b1"][i] for i in range(H)] for x in xs]
    a = bn.forward(z1) if bn else z1
    h = [list(map(lambda v: max(0.0, v), row)) for row in a]
    z2 = [sum(p["w2"][i] * row[i] for i in range(H)) + p["b2"] for row in h]
    return a, h, list(map(ref.sigmoid, z2))


def step(ref, p, xs, ys, bn, lr) -> None:
    """One mini-batch. The gradient passes straight through the normalisation — see check 1."""
    a, h, out = forward(ref, p, xs, bn)
    n = len(xs)
    dz2 = [(o - y) / n for o, y in zip(out, ys)]
    dz1 = [[d * p["w2"][i] * (1.0 if row[i] > 0 else 0.0) for i in range(H)]
           for d, row in zip(dz2, a)]
    descend(p, xs, h, dz1, dz2, lr)


def descend(p, xs, h, dz1, dz2, lr) -> None:
    """The lesson's own update rule, summed over the batch."""
    for i in range(H):
        p["w2"][i] -= lr * sum(d * row[i] for d, row in zip(dz2, h))
        p["b1"][i] -= lr * sum(row[i] for row in dz1)
        for j in range(2):
            p["w1"][i][j] -= lr * sum(row[i] * x[j] for row, x in zip(dz1, xs))
    p["b2"] -= lr * sum(dz2)


def evaluate(ref, p, test, bn) -> tuple:
    out = forward(ref, p, [x for x, _y in test], bn)[2]
    pairs = list(zip(out, [y for _x, y in test]))
    loss = sum(-(y * math.log(max(1e-15, o)) + (1 - y) * math.log(max(1e-15, 1 - o)))
               for o, y in pairs) / len(pairs)
    return loss, 100.0 * sum((o >= 0.5) == (y >= 0.5) for o, y in pairs) / len(pairs)


def batched(ref, lr, use_bn, data, epochs=EPOCHS) -> tuple:
    """The comparison the exercise intends: mini-batches, and a BatchNorm that trains."""
    train, p, rng = data[:SPLIT], init(), random.Random(1)
    bn = ref.BatchNorm(H) if use_bn else None
    for _epoch in range(epochs):
        order = rng.sample(range(len(train)), len(train))
        for s in range(0, len(order), BATCH):
            rows = [train[i] for i in order[s:s + BATCH]]
            step(ref, p, [x for x, _y in rows], [y for _x, y in rows], bn, lr)
    return evaluate(ref, p, data[SPLIT:], bn)


def online(ref, lr, data) -> tuple:
    """The lesson's own network and its own loop, untouched, at one learning rate."""
    net = ref.RegularizedNetwork(H, lr)
    with parity.quiet():
        history = net.train_model(data[:SPLIT], data[SPLIT:], epochs=EPOCHS)
    return history[-1][2], history[-1][3]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    data = ref.make_circle_data(200, 42)
    pair = {lr: (batched(ref, lr, False, data), batched(ref, lr, True, data))
            for lr in RATES + HIGH}
    return {"single": ref.BatchNorm(4).forward([[1.0, -2.0, 3.0, 0.5]])[0],
            "backward": {n: hasattr(getattr(ref, n), "backward")
                         for n in ("BatchNorm", "LayerNorm", "RMSNorm", "Dropout")},
            "vanilla": {lr: online(ref, lr, data) for lr in RATES},
            "named": {lr: pair[lr] for lr in RATES}, "high": {lr: pair[lr] for lr in HIGH}}


def digest(result) -> dict:
    """Every listing `verify` prints, so that stays a list of comparisons."""
    named, high = result["named"], result["high"]
    fmt = lambda t: "; ".join(f"lr {lr}: plain {a[1]:.0f}% / bn {b[1]:.0f}%"    # noqa: E731
                              for lr, (a, b) in t.items())
    return {"named": fmt(named), "high": fmt(high),
            "plain_ok": [lr for lr, (a, _b) in high.items() if a[1] >= 99],
            "bn_ok": [lr for lr, (_a, b) in high.items() if b[1] >= 99],
            "flags": ", ".join(f"{n} {v}" for n, v in result["backward"].items()),
            "vanilla": ", ".join(f"lr {lr}: test {acc:.0f}% (loss {loss:.4f})"
                                 for lr, (loss, acc) in result["vanilla"].items())}


def verify(result):
    d, named, back = digest(result), result["named"], result["backward"]
    held = max(d["plain_ok"])
    return [
        practice.Check("MECHANISM: as shipped, the lesson's BatchNorm cannot go where the "
                       "exercise puts it",
                       result["single"] == [0.0] * 4 and back["Dropout"]
                       and not any(back[n] for n in ("BatchNorm", "LayerNorm", "RMSNorm")),
                       f"`RegularizedNetwork` trains one sample at a time, and `BatchNorm.forward` "
                       f"on a batch of one returns {result['single']} for [1.0, -2.0, 3.0, 0.5] — "
                       f"the batch variance is 0, so x_hat is 0 and the output is beta, and the "
                       f"hidden layer stops depending on its input. Nor is there a gradient: "
                       f"`hasattr(cls, 'backward')` gives {d['flags']}, so gamma and beta never "
                       f"move and any use of it back-propagates as if it were the identity"),
        practice.Check("ANSWER: at the three rates named, the vanilla network never diverges",
                       all(acc >= 99 for _loss, acc in result["vanilla"].values()),
                       f"the lesson's own network and loop, unmodified, over {EPOCHS} epochs — "
                       f"{d['vanilla']}. There is nothing at 0.01, 0.05 or 0.1 to rescue"),
        practice.Check("ANSWER: batched, so the layer can run at all, it is worse at all three "
                       "rates",
                       all(a[1] >= b[1] for a, b in named.values()),
                       f"mini-batches of {BATCH}, the lesson's own BatchNorm between z1 and the "
                       f"ReLU — {d['named']}. Normalising {H} pre-activations over {BATCH} samples "
                       f"throws away the scale the one output neuron reads, and the gamma that "
                       f"would learn it back never moves"),
        practice.Check("FINDING: BatchNorm lowers the rate at which training collapses, rather "
                       "than raising it",
                       held > max(d["bn_ok"], default=0.0),
                       f"past the named rates — {d['high']}. Plain holds 100% up to lr = {held} "
                       f"and only fails at {min(lr for lr in HIGH if lr > held)}, while BatchNorm "
                       f"reaches 100% at none of them — and where plain does fail, neither arm is "
                       f"usable. There is no rate at which the layer buys stability here"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
