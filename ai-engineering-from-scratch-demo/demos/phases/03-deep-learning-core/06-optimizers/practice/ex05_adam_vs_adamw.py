"""Exercise 5 — Adam vs AdamW, once Adam is given the decay too.

    Compare Adam vs AdamW on a network with large weights. Initialize all weights
    to random values in [-5, 5] (much larger than normal). Train for 200 epochs
    with weight_decay=0.1. Plot the L2 norm of weights over training for both
    optimizers. AdamW should show faster weight shrinkage.

Reading of the exercise: the lesson's `Adam` takes no weight_decay, so "Adam with
weight_decay=0.1" cannot be run as shipped. Three arms are compared instead — Adam as written,
Adam with the penalty folded into the gradient (`l2`, the pre-AdamW reading), and AdamW — over
three seeds, the norm at epochs 0, 50 and 200 standing in for the plot.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "06-optimizers"
LR, WD, INIT, EPOCHS = 0.001, 0.1, 5.0, 200
MODES, SEEDS, N = ("adam", "l2", "adamw"), (0, 1, 42), 200
norm = lambda values: math.sqrt(sum(v * v for v in values))             # noqa: E731
build = lambda r, m: r.AdamW(lr=LR, weight_decay=WD) if m == "adamw" else r.Adam(lr=LR)  # noqa: E731


def train(ref, mode, seed, epochs=EPOCHS):
    """The lesson's loop from a uniform[-5, 5] start; `mode` picks where the decay goes."""
    data, opt, rng = ref.make_circle_data(), build(ref, mode), random.Random(seed)
    net = ref.OptimizerTestNetwork(opt, hidden_size=8)
    net.set_params([rng.uniform(-INIT, INIT) for _ in range(33)])
    trail, right, ones = [norm(net.get_params())], 0, 0
    for _epoch in range(epochs):
        right, ones = 0, 0
        for point, label in data:
            pred, params = net.forward(point), net.get_params()
            grads = net.compute_grads(label)
            grads = [g + WD * p for g, p in zip(grads, params)] if mode == "l2" else grads
            opt.step(params, grads)
            net.set_params(params)
            right, ones = right + ((pred >= 0.5) == (label >= 0.5)), ones + (pred >= 0.5)
        trail.append(norm(net.get_params()))
    return {"trail": trail, "ones": ones, "acc": 100.0 * right / len(data)}


def lesson_demo(ref, mode):
    """The lesson's own STEP 4 block verbatim, with an `l2` arm bolted on."""
    random.seed(42)
    weights = [random.uniform(-INIT, INIT) for _ in range(10)]
    start, opt = norm(weights), build(ref, mode)
    for _step in range(100):
        g = [random.gauss(0, 0.1) for _ in range(10)]
        opt.step(weights, [a + WD * w for a, w in zip(g, weights)] if mode == "l2" else g)
    return start - norm(weights), norm(weights)


def across(runs, mode, key, index=None):
    """One number per seed for `mode`: the three of them as text, plus min and max."""
    values = [runs[(mode, s)][key] if index is None else runs[(mode, s)][key][index] for s in SEEDS]
    return {"text": ", ".join(f"{v:.2f}" for v in values), "lo": min(values), "hi": max(values)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = {(mode, seed): train(ref, mode, seed) for mode in MODES for seed in SEEDS}
    zeros = sum(1 for _point, label in ref.make_circle_data() if label == 0.0)
    quiet, decay = [3.0, -4.0], ref.AdamW(lr=LR, weight_decay=WD)
    for _step in range(100): decay.step(quiet, [0.0, 0.0])                    # noqa: E701
    return {"runs": runs, "demo": {mode: lesson_demo(ref, mode) for mode in MODES},
            "majority": 100.0 * max(zeros, N - zeros) / N, "pull": WD * LR * INIT,
            "geometric": abs(norm(quiet) - 5.0 * (1 - WD * LR) ** 100) / norm(quiet)}


def verify(result):
    runs, demo, majority = result["runs"], result["demo"], result["majority"]
    end, acc = ({m: across(runs, m, "trail", EPOCHS) for m in MODES},
                {m: across(runs, m, "acc") for m in MODES})
    start, mid = across(runs, "adam", "trail", 0), across(runs, "adamw", "trail", 50)
    return [
        practice.Check("ANSWER: against Adam as the lesson ships it, AdamW does shrink faster",
                       end["adamw"]["hi"] < 0.5 * end["adam"]["lo"],
                       f"weight L2 norm from a uniform[-5, 5] start ({start['text']} at epoch 0) to "
                       f"epoch 200 over seeds {SEEDS}: Adam {end['adam']['text']}, AdamW "
                       f"{end['adamw']['text']} (epoch 50: {mid['text']}). The lesson's Adam takes "
                       f"no weight_decay at all, so its arm is decay-free"),
        practice.Check("FINDING: give Adam the same weight_decay and the ordering reverses",
                       end["l2"]["hi"] < end["adamw"]["lo"],
                       f"folding wd * w into the gradient before Adam.step — the pre-AdamW reading "
                       f"of 'Adam with weight decay' — ends at {end['l2']['text']} against AdamW's "
                       f"{end['adamw']['text']} on every seed, so 'AdamW should show faster weight "
                       f"shrinkage' is false at a matched weight_decay"),
        practice.Check("MECHANISM: decoupled decay is proportional, coupled decay is normalised",
                       result["geometric"] < 1e-15,
                       f"with zero gradients the lesson's AdamW multiplies every weight by "
                       f"(1 - lr * wd) per step: 100 steps land on n0 * (1 - lr*wd)^100 to "
                       f"{result['geometric']:.1e} relative — a pull of {result['pull']:.0e} at "
                       f"|w| = {INIT} that vanishes with w. The coupled term instead enters the "
                       f"numerator, divided by sqrt(v_hat) — a pull of order lr = {LR} whatever w"),
        practice.Check("CONTROL: the extra shrinkage is bought by destroying the classifier",
                       acc["l2"]["lo"] == acc["l2"]["hi"] == majority
                       and acc["adamw"]["lo"] > majority,
                       f"final training accuracy: l2 {acc['l2']['text']}, AdamW "
                       f"{acc['adamw']['text']}, Adam {acc['adam']['text']}. The l2 arm sits exactly "
                       f"on the majority-class rate ({majority:.1f}% of the circle data is labelled "
                       f"0), predicting class 1 for {runs[('l2', 0)]['ones']} of {N} points"),
        practice.Check("FINDING: the lesson's own STEP 4 reports the wrong ratio",
                       demo["adamw"][0] > 9 * demo["adam"][0],
                       f"its 100-step demo shrinks the norm by {demo['adam'][0]:.4f} under Adam and "
                       f"{demo['adamw'][0]:.4f} under AdamW, a factor of "
                       f"{demo['adamw'][0] / demo['adam'][0]:.1f}; what it prints is norm_adam / "
                       f"norm_adamw = {demo['adam'][1] / demo['adamw'][1]:.1f}x, a ratio of endpoints "
                       f"— and the l2 arm shrinks it by {demo['l2'][0]:.4f} again"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
