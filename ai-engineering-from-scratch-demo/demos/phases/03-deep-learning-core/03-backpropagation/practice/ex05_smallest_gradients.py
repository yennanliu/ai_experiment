"""Exercise 5 — every parameter's gradient after training, and which layer is smallest.

    Build a visualization: after training on XOR, print the gradient of every
    parameter in the network. Identify which layer has the smallest gradients.
    This demonstrates the vanishing gradient problem you read about in the
    Concept section.

Reading of the exercise: "print the gradient of every parameter" is a table, so check 1 is
that table, reduced to per-layer mean/min/max — 17 numbers is a table, 17 lines is a log wall.
"Which layer has the smallest gradients" needs a rule for *smallest*, and checks 2-3 show that
the two obvious rules disagree and that the word "after" costs an order of magnitude. Checks
4-5 measure the attenuation the Concept section describes, at the moment it is visible.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "03-backpropagation"
XOR = [([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 0.0)]
SHALLOW, DEEP = [2, 4, 1], [2, 4, 4, 4, 1]      # the lesson's own net, and a deeper one
EPOCHS, LR, SEEDS = 1000, 1.0, (42, 7, 99)      # train_xor's own epoch count and rate


def backward(ref, net) -> float:
    """One backward pass over the whole XOR batch, as `train_xor` does it."""
    total = ref.Value(0.0)
    for inputs, target in XOR:
        total = total + ref.mse_loss(net([ref.Value(i) for i in inputs]), target)
    net.zero_grad()
    total.backward()
    return total.data


def snapshot(net) -> list:
    """(mean, min, max) of |grad| per layer — the exercise's printout, reduced."""
    return [(statistics.mean(abs(p.grad) for p in layer.parameters()),
             min(abs(p.grad) for p in layer.parameters()),
             max(abs(p.grad) for p in layer.parameters())) for layer in net.layers]


def run(ref, seed, sizes, epochs=EPOCHS) -> dict:
    random.seed(seed)
    net = ref.Network(sizes)
    start_loss, start = backward(ref, net), snapshot(net)
    for _epoch in range(epochs):
        backward(ref, net)
        for param in net.parameters():
            param.data -= LR * param.grad
    end_loss = backward(ref, net)
    return {"start": start, "end": snapshot(net), "loss": (start_loss, end_loss),
            "n": len(net.parameters())}


def argmin_layer(rows, index) -> int:
    """Which layer is smallest, 1-based, judged by field `index` of each row."""
    return min(range(len(rows)), key=lambda i: rows[i][index]) + 1


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {"shallow": {s: run(ref, s, SHALLOW) for s in SEEDS},
            "deep": {s: run(ref, s, DEEP) for s in SEEDS},
            "sweep": {n: run(ref, SEEDS[0], [2] + [4] * n + [1], epochs=0)["start"]
                      for n in (1, 2, 3, 4)}}


def ratios(runs, when, index=0) -> list:
    return [r[when][0][index] / r[when][-1][index] for r in runs.values()]


def rankings(shallow, deep) -> dict:
    """Which layer is smallest, under each of the two readings of 'smallest'."""
    return {"smallest": [argmin_layer(r["end"], 0) for r in shallow.values()],
            "deep_mean": [argmin_layer(r["end"], 0) for r in deep.values()],
            "deep_min": [argmin_layer(r["end"], 1) for r in deep.values()],
            "deep_start": [argmin_layer(r["start"], 0) for r in deep.values()]}


def strings(shallow, deep, sweep) -> dict:
    """The tables and lists `verify` prints, kept out of it."""
    one, steps = shallow[SEEDS[0]], [sweep[n][0][0] / sweep[n + 1][0][0] for n in (1, 2, 3)]
    return {"steps": steps, "attn": ", ".join(f"{s:.1f}" for s in steps),
            "table": " | ".join(f"L{i + 1} mean {m:.2e} min {lo:.2e} max {hi:.2e}"
                                for i, (m, lo, hi) in enumerate(one["end"])),
            "deep_row": ", ".join(f"{m:.2e}" for m, _lo, _hi in deep[SEEDS[0]]["start"]),
            "firsts": ", ".join(f"{n}: {rows[0][0]:.2e}" for n, rows in sweep.items()),
            "ends": ", ".join(f"{deep[s]['loss'][1]:.6f}" for s in SEEDS)}


def digest(result) -> dict:
    """Every summary `verify` quotes, so that stays a list of comparisons."""
    shallow, deep = result["shallow"], result["deep"]
    start_r, end_r = ratios(shallow, "start"), ratios(shallow, "end")
    return {"one": shallow[SEEDS[0]], "start_r": start_r, "end_r": end_r,
            "starts": ", ".join(f"{r:.3f}" for r in start_r),
            "finals": ", ".join(f"{r:.2f}" for r in end_r),
            **rankings(shallow, deep), **strings(shallow, deep, result["sweep"])}


def verify(result):
    d = digest(result)
    one = d["one"]
    start_r, end_r = d["start_r"], d["end_r"]
    return [
        practice.Check("ANSWER: on the lesson's own 2-4-1 net the hidden layer is smallest, "
                       "at all three seeds",
                       d["smallest"] == [1, 1, 1],
                       f"all {one['n']} gradients after {EPOCHS} epochs, by layer: {d['table']}. "
                       f"Loss {one['loss'][0]:.4f} -> {one['loss'][1]:.6f}; layer 1 is smaller "
                       f"than layer 2 by mean |grad| at every seed, ratios {d['finals']}"),
        practice.Check("FINDING: measuring *after* training understates the effect ~10x",
                       max(start_r) < 0.2 and min(end_r) > 0.4,
                       f"the same ratio before the first update is {d['starts']} — a "
                       f"{1 / max(start_r):.0f}-{1 / min(start_r):.0f}x attenuation — "
                       f"against {min(end_r):.2f}-{max(end_r):.2f} after. Training ends at a "
                       f"minimum, where *every* gradient is small, so the last epoch is the worst "
                       f"moment to look for a gradient that vanishes with depth"),
        practice.Check("FINDING: on a deeper net 'which layer is smallest' has two answers, and "
                       "neither is layer 1",
                       d["deep_mean"] != d["deep_min"] and 1 not in d["deep_mean"],
                       f"a {'-'.join(map(str, DEEP))} net after {EPOCHS} epochs: by mean |grad| "
                       f"the smallest layer is {d['deep_mean']} at the three seeds, by the single "
                       f"smallest parameter it is {d['deep_min']}. Before training both rules "
                       f"agree on layer {d['deep_start'][0]} at every seed — the ordering the "
                       f"Concept section describes is there, and training erases it"),
        practice.Check("MECHANISM: at init the attenuation is monotone and about 1/5 per layer",
                       all(s > 3 for s in d["steps"]),
                       f"mean |grad| per layer at init on the deep net, input to output: "
                       f"{d['deep_row']}. Adding a hidden layer divides the input layer's "
                       f"gradient by {d['attn']} ({d['firsts']} for 1, 2, 3, 4 hidden layers) "
                       f"— sigmoid' <= 1/4 and the "
                       f"He-scaled weights are under 1, so each layer multiplies by less than 1"),
        practice.Check("CONTROL: the deep net's gradients are not small because it converged",
                       all(result["deep"][s]["loss"][1] > 0.9 for s in SEEDS),
                       f"final loss {d['ends']} against 1.0, the loss of predicting 0.5 on all "
                       f"four rows. The deep "
                       f"net never left its initialisation, so its post-training gradients are "
                       f"small for the reason the exercise says — unlike the 2-4-1 net's, which "
                       f"are small because it succeeded"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
