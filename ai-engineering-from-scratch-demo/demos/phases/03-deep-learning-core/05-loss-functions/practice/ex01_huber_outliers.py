"""Exercise 1 — Huber against MSE when 5% of the sin(x) targets are corrupted.

    Implement Huber loss (smooth L1 loss), which is MSE for small errors and MAE
    for large errors. Train a regression network predicting y = sin(x) with MSE vs
    Huber when 5% of training targets have random noise added (outliers). Compare
    final test error.

Reading of the exercise: "compare final test error" is an argument only if one
thing differs between the arms, so the network is a frozen 12-bump RBF hidden
layer with a trained linear output and dL/dy_pred is the only difference. Test
error is squared error against the clean sin(x) — MSE's own metric, so the
comparison runs on MSE's terms. Check 3 is the control that needs: Huber's inner
branch is e^2/2, half of MSE, so it also takes half-sized steps.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "05-loss-functions"
CENTERS = [-3.0 + 6.0 * i / 11 for i in range(12)]
GRID = [-3.0 + 6.0 * i / 199 for i in range(200)]
SIG, DELTA, LR, EPOCHS, SEEDS = 0.6, 0.5, 0.5, 1000, (3, 5, 11)


def feats(x) -> list:
    return [math.exp(-((x - c) ** 2) / (2 * SIG * SIG)) for c in CENTERS]


def huber(preds, targs, delta=DELTA) -> float:
    """MSE's e^2/2 inside |e| <= delta, MAE's delta*(|e| - delta/2) outside."""
    errs = [abs(p - t) for p, t in zip(preds, targs)]
    return sum(0.5 * e * e if e <= delta else delta * (e - 0.5 * delta)
               for e in errs) / len(errs)


def huber_gradient(preds, targs, delta=DELTA) -> list:
    return [(e if abs(e) <= delta else math.copysign(delta, e)) / len(preds)
            for e in (p - t for p, t in zip(preds, targs))]


def make_data(seed, frac=0.05):
    rng, xs = random.Random(seed), [-3.0 + 6.0 * i / 59 for i in range(60)]
    ys = [math.sin(x) for x in xs]
    for i in rng.sample(range(60), round(frac * 60)):
        ys[i] += rng.choice([-1, 1]) * rng.uniform(3.0, 6.0)
    return xs, ys


def train(xs, ys, grad_fn, lr=LR) -> tuple:
    """Full-batch descent on the output layer; the RBF layer is frozen."""
    w, b, rows = [0.0] * len(CENTERS), 0.0, [feats(x) for x in xs]
    for _ in range(EPOCHS):
        preds = [sum(wi * fi for wi, fi in zip(w, f)) + b for f in rows]
        for g, f in zip(grad_fn(preds, ys), rows):
            w = [wi - lr * g * fi for wi, fi in zip(w, f)]
            b -= lr * g
    return w, b


def error(pair) -> float:
    """Squared error against the uncorrupted curve, on a 200-point grid."""
    return sum((sum(wi * fi for wi, fi in zip(pair[0], feats(x))) + pair[1]
                - math.sin(x)) ** 2 for x in GRID) / len(GRID)


def fd_gap(loss, grad, preds, targs, i) -> float:
    """Coordinate i of the analytic gradient against a central difference of `loss`."""
    at = lambda s: [p + s * (j == i) for j, p in enumerate(preds)]           # noqa: E731
    return abs(grad(preds, targs)[i] - (loss(at(1e-6), targs) - loss(at(-1e-6), targs)) / 2e-6)


def solve() -> dict:
    ref = parity.load_reference(PHASE, LESSON, "main")
    sets = [make_data(s) for s in SEEDS]
    runs = {name: [error(train(xs, ys, fn, lr)) for xs, ys in sets] for name, fn, lr
            in [("mse", ref.mse_gradient, LR), ("huber", huber_gradient, LR),
                ("half", ref.mse_gradient, LR / 2)]}
    (xs, ys), pr, tg = sets[0], [0.4, -1.2, 0.9, 2.5], [0.5, 0.3, 0.95, -1.0]
    mse_share, huber_share = shares(ref, ys)
    return {k: sum(v) / len(SEEDS) for k, v in runs.items()} | {
        "worst": min(a / b for a, b in zip(runs["mse"], runs["huber"])),
        "mse_share": mse_share, "huber_share": huber_share,
        "wide": widest(ref, xs, ys), "clean": clean_ratio(ref, xs),
        "fd": gradcheck(ref, pr, tg)}


def widest(ref, xs, ys) -> float:
    """Huber at delta = 1e9 is MSE at half the rate — how far the fitted lines drift apart."""
    return max(abs(a - b) for a, b in zip(
        train(xs, ys, ref.mse_gradient)[0],
        train(xs, ys, lambda p, t: huber_gradient(p, t, 1e9), 2 * LR)[0]))


def clean_ratio(ref, xs) -> float:
    """Huber's error against MSE's on the same curve with no outliers at all."""
    clean = [math.sin(x) for x in xs]
    return error(train(xs, clean, huber_gradient)) / error(train(xs, clean, ref.mse_gradient))


def shares(ref, ys) -> tuple:
    """How much of the gradient at zero the |y| > 1.5 outliers own, under each loss."""
    hit = lambda g: sum(abs(v) for v, y in zip(g, ys) if abs(y) > 1.5) / sum(map(abs, g))
    return hit(ref.mse_gradient([0.0] * 60, ys)), hit(huber_gradient([0.0] * 60, ys))


def gradcheck(ref, preds, targets) -> float:
    """Worst finite-difference gap for both losses over the sample points."""
    return max(fd_gap(fn, gr, preds, targets, i) for fn, gr in
               [(ref.mse, ref.mse_gradient), (huber, huber_gradient)] for i in range(4))


def verify(result) -> list:
    ratio, half = result["mse"] / result["huber"], result["half"] / result["huber"]
    return [
        practice.Check("ANSWER: Huber's test error is 71x lower, in MSE's own units",
                       ratio > 60 and result["worst"] > 45,
                       f"mean over {len(SEEDS)} seeds of squared error against clean sin(x): "
                       f"MSE-trained {result['mse']:.5f}, Huber-trained {result['huber']:.5f} "
                       f"— {ratio:.1f}x, worst seed {result['worst']:.1f}x"),
        practice.Check("MECHANISM: 3 rows in 60 own a quarter of the MSE gradient",
                       result["mse_share"] > 0.20 and result["huber_share"] < 0.08,
                       f"the corrupted 5% carry {result['mse_share']:.1%} of sum|dL/dy_pred| "
                       f"at the zero start under MSE, {result['huber_share']:.1%} under "
                       f"Huber, whose row gradient is clipped at delta/n = {DELTA / 60:.4f}"),
        practice.Check("MECHANISM: inside delta, Huber *is* MSE at half the step size",
                       result["wide"] == 0.0 and result["clean"] > 1.0,
                       f"at delta = 1e9, lr = {2 * LR} it reproduces the MSE run at lr = {LR} "
                       f"to a weight deviation of {result['wide']:.1e}, bit for bit; on "
                       f"outlier-free targets it costs {result['clean']:.2f}x MSE's error, "
                       f"the gap of a half-rate run at {EPOCHS} epochs"),
        practice.Check("CONTROL: halving MSE's learning rate does not recover it", half > 45,
                       f"MSE at lr = {LR / 2} scores {result['half']:.5f} against "
                       f"{result['mse']:.5f} at lr = {LR}, still {half:.1f}x worse than "
                       f"Huber — the win is the clip, not the smaller step it implies"),
        practice.Check("both arms descend the loss they claim to", result["fd"] < 1e-8,
                       f"worst gap to a central difference over 4 residuals from 0.05 to 3.5, "
                       f"one either side of the delta = {DELTA} kink, for the lesson's "
                       f"`mse_gradient` and for `huber_gradient`: {result['fd']:.1e}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
