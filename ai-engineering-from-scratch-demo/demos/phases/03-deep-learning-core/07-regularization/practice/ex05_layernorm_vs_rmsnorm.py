"""Exercise 5 — LayerNorm against RMSNorm on a four-layer network.

    Compare LayerNorm vs RMSNorm on a 4-layer network (not just 2). Initialize
    both with the same weights. Train for 200 epochs and compare final accuracy,
    training speed (time per epoch), and gradient magnitudes at the first layer.
    Verify that RMSNorm is faster with the same accuracy.

Reading of the exercise: the lesson has no four-layer network, so this builds a 2-16-16-16-1
one and inserts the lesson's own norm objects before each ReLU. Neither class has a backward
(exercise 3), so the gradient passes straight through them and gamma never moves — check 5 is
that. "Initialize both with the same weights" is taken literally: one seeded init, shared by
all three arms. Timing is reported per call as well as per epoch, because those disagree.
"""

from __future__ import annotations

import math
import random
import statistics
import time

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "07-regularization"
SIZES, LAYERS, EPOCHS, LR, SPLIT = [2, 16, 16, 16, 1], 4, 200, 0.05, 150
biggest = lambda delta, x: max(abs(d * v) for d in delta for v in x)      # noqa: E731
upstream = lambda rows, delta, width: [sum(d * row[c] for d, row in zip(delta, rows))  # noqa: E731
                                       for c in range(width)]


def init(seed=0) -> tuple:
    """One seeded draw, shared by all three arms — 'initialize both with the same weights'."""
    random.seed(seed)
    return ([[[random.gauss(0, 0.5) for _ in range(SIZES[i])] for _ in range(SIZES[i + 1])]
             for i in range(LAYERS)], [[0.0] * SIZES[i + 1] for i in range(LAYERS)])


def norms(ref, kind) -> list | None:
    """One of the lesson's own norm objects per hidden layer, or none at all."""
    cls = {"ln": ref.LayerNorm, "rms": ref.RMSNorm}.get(kind)
    return [cls(SIZES[i + 1]) for i in range(LAYERS - 1)] if cls else None


def forward(ref, w, b, layer_norms, x) -> tuple:
    """a0 -> [W, +b, norm, ReLU] x 3 -> W, +b, sigmoid. Returns activations and pre-ReLU."""
    acts, pre = [x], []
    for i in range(LAYERS - 1):
        z = [sum(w[i][r][c] * acts[i][c] for c in range(SIZES[i])) + b[i][r]
             for r in range(SIZES[i + 1])]
        pre.append(layer_norms[i].forward(z) if layer_norms else z)
        acts.append([max(0.0, v) for v in pre[i]])
    last = LAYERS - 1
    out = ref.sigmoid(sum(w[last][0][c] * acts[last][c] for c in range(SIZES[last])) + b[last][0])
    return acts, pre, out


def descend(w, b, acts, pre, delta) -> float:
    """Straight-through backward: the norms have none, so they scale nothing on the way back."""
    first = 0.0
    for i in range(LAYERS - 1, -1, -1):
        if i < LAYERS - 1:
            delta = [delta[r] * (1.0 if pre[i][r] > 0 else 0.0) for r in range(SIZES[i + 1])]
        if i == 0:
            first = biggest(delta, acts[0])
        nxt = upstream(w[i], delta, SIZES[i])
        for r, row in enumerate(w[i]):
            for c in range(SIZES[i]):
                row[c] -= LR * delta[r] * acts[i][c]
            b[i][r] -= LR * delta[r]
        delta = nxt
    return first


def run(ref, kind, data) -> dict:
    """200 epochs of the lesson's own online loop over a four-layer net."""
    w, b = init()
    layer_norms, times, grads, skew = norms(ref, kind), [], [], []
    for epoch in range(EPOCHS):
        clock = time.perf_counter()
        for x, y in data[:SPLIT]:
            acts, pre, out = forward(ref, w, b, layer_norms, x)
            skew += [] if epoch else [offcentre(acts, w, b)]
            grads.append(descend(w, b, acts, pre, [out - y]))
        times.append(time.perf_counter() - clock)
    return {"acc": accuracy(ref, w, b, layer_norms, data[SPLIT:]), "time": min(times),
            "grad": statistics.mean(grads[:SPLIT]), "skew": statistics.mean(skew) if skew else 0.0,
            "gamma": layer_norms[0].gamma if layer_norms else None}


def offcentre(acts, w, b) -> float:
    """|mean(z)| / rms(z) on the first hidden pre-activation — what LayerNorm subtracts."""
    z = [sum(w[0][r][c] * acts[0][c] for c in range(2)) + b[0][r] for r in range(SIZES[1])]
    return abs(statistics.mean(z)) / (math.sqrt(sum(v * v for v in z) / len(z)) + 1e-12)


def accuracy(ref, w, b, layer_norms, held) -> float:
    right = sum((forward(ref, w, b, layer_norms, x)[2] >= 0.5) == (y >= 0.5) for x, y in held)
    return 100.0 * right / len(held)


def percall(obj, reps=20000) -> float:
    vec = [random.Random(3 + i).gauss(0, 1) for i in range(SIZES[1])]
    clock = time.perf_counter()
    for _ in range(reps):
        obj.forward(vec)
    return 1e6 * (time.perf_counter() - clock) / reps


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    data = ref.make_circle_data(200, 42)
    return {"arms": {k: run(ref, k, data) for k in ("none", "ln", "rms")},
            "calls": {"ln": percall(ref.LayerNorm(SIZES[1])),
                      "rms": percall(ref.RMSNorm(SIZES[1]))}}


def verify(result):
    arms, calls = result["arms"], result["calls"]
    ln, rms, bare = arms["ln"], arms["rms"], arms["none"]
    per_epoch = 100 * (ln["time"] - rms["time"]) / ln["time"]
    return [
        practice.Check("ANSWER: RMSNorm is faster, and the accuracies are not the same",
                       calls["rms"] < 0.6 * calls["ln"] and rms["acc"] > ln["acc"],
                       f"{EPOCHS} epochs from one shared init: RMSNorm {rms['acc']:.1f}% at "
                       f"{1e3 * rms['time']:.2f} ms/epoch against LayerNorm {ln['acc']:.1f}% at "
                       f"{1e3 * ln['time']:.2f} ms — {per_epoch:.1f}% faster and "
                       f"{rms['acc'] - ln['acc']:.0f} points more accurate. Half of 'faster with "
                       f"the same accuracy' holds; the half naming accuracy does not"),
        practice.Check("MECHANISM: the gap is large per call and small per epoch",
                       calls["ln"] / calls["rms"] > 2.0 and per_epoch < 15,
                       f"one call on a {SIZES[1]}-vector costs {calls['ln']:.3f} us for LayerNorm "
                       f"against {calls['rms']:.3f} us for RMSNorm — "
                       f"{calls['ln'] / calls['rms']:.1f}x, since LayerNorm computes a mean, a "
                       f"variance and a beta where RMSNorm computes one sum of squares. An epoch "
                       f"is {SPLIT} samples x {LAYERS} layers of pure-Python matrix work, so the "
                       f"norm calls carry only {per_epoch:.1f}% of it"),
        practice.Check("FINDING: the fastest and most accurate arm is no normalisation at all",
                       bare["time"] < rms["time"] and bare["acc"] >= rms["acc"],
                       f"the same net with the norms removed: {bare['acc']:.1f}% at "
                       f"{1e3 * bare['time']:.2f} ms/epoch against RMSNorm's {rms['acc']:.1f}% at "
                       f"{1e3 * rms['time']:.2f} — no depth problem here for either to fix"),
        practice.Check("MECHANISM: the two are not the same transform here",
                       0.05 < rms["skew"] < 0.9,
                       f"RMSNorm is LayerNorm without the mean subtraction, so they coincide only "
                       f"where the input already has zero mean. On this net's first hidden "
                       f"pre-activation |mean(z)| / rms(z) averages {rms['skew']:.4f}, so the "
                       f"term LayerNorm subtracts is {rms['skew']:.0%} of the vector's own scale"),
        practice.Check("FINDING: gamma never moves, so 'training' the norms is not what is "
                       "being compared",
                       ln["gamma"] == [1.0] * SIZES[1] == rms["gamma"],
                       f"after {EPOCHS} epochs every gamma in both arms is still exactly 1.0, "
                       f"because neither class has a backward — the gradient passes through them "
                       f"untouched. First-layer |grad| is {bare['grad']:.4f} with no norm, "
                       f"{ln['grad']:.4f} with LayerNorm and {rms['grad']:.4f} with RMSNorm — the "
                       f"norms raise it {100 * (ln['grad'] / bare['grad'] - 1):.0f}% and differ "
                       f"from each other by {100 * abs(ln['grad'] / rms['grad'] - 1):.1f}%"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
