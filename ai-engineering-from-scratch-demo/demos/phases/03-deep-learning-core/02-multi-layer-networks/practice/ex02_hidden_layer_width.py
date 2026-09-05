"""Exercise 2 — what widening the circle classifier's hidden layer actually changes.

    Change the hidden layer size in the circle classifier from 8 to 2, then to 32.
    Run the forward pass with random weights each time. Does the number of hidden
    neurons change the output range or distribution? Why?

Reading of the exercise: "the output range" is ambiguous and the ambiguity is the
finding, so check 1 reports the range one net covers over the data and check 2 the
range the ensemble covers — they answer differently. "Why?" is check 3, a sqrt(h)
law measured at four widths rather than asserted. Checks 4 and 5 ask what all this
buys, since the output distribution moves a long way and the classifier does not.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "02-multi-layer-networks"
WIDTHS = [2, 8, 32, 128]
NETS, SEED = 400, 7


def circle(seed=42, n=200) -> list:
    """The lesson's own dataset, drawn from a private RNG with its seed."""
    rng = random.Random(seed)
    pts = [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)]
    return [([x, y], 1 if x * x + y * y < 0.25 else 0) for x, y in pts]


def build(ref, rng, h):
    return ref.Network([
        ref.Layer(2, h, weights=[[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(h)]),
        ref.Layer(h, 1, weights=[[rng.uniform(-1, 1) for _ in range(h)]])])


def run(net, data) -> tuple:
    """Outputs, and the output neuron's pre-activation z, over the whole dataset."""
    out, zed = [], []
    for x, _t in data:
        out.append(net.forward(x)[0])
        zed.append(sum(w * a for w, a in
                       zip(net.layers[1].weights[0], net.layers[0].last_output)))
    return out, zed


def survey(ref, data, h, nets=NETS) -> dict:
    """Outputs, per-net ranges, accuracies and output-layer pre-activations at width h."""
    rng = random.Random(SEED)
    runs = [run(build(ref, rng, h), data) for _ in range(nets)]
    accs = [100 * sum((v >= 0.5) == bool(t) for v, (_x, t) in zip(r, data)) / len(data)
            for r, _z in runs]
    pooled = [v for r, _z in runs for v in r]
    return {"lo": min(pooled), "hi": max(pooled), "sd": statistics.pstdev(pooled),
            "span": statistics.mean(max(r) - min(r) for r, _z in runs),
            "acc": statistics.mean(accs), "best": max(accs), "worst": min(accs),
            "const": sum(len({v >= 0.5 for v in r}) == 1 for r, _z in runs),
            "zsd": statistics.pstdev([v for _r, z in runs for v in z])}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    data = circle()
    lesson = [build(ref, random.Random(SEED), 8).forward(x)[0] for x, _ in data]
    return {"by_width": {h: survey(ref, data, h) for h in WIDTHS},
            "inside": sum(t for _x, t in data), "n": len(data),
            "lesson_span": max(lesson) - min(lesson),
            "lesson_acc": 100 * sum((v >= 0.5) == bool(t)
                                    for v, (_x, t) in zip(lesson, data)) / len(data)}


def verify(result):
    w = result["by_width"]
    majority = 100 * (result["n"] - result["inside"]) / result["n"]
    ratio = [w[h]["zsd"] / math.sqrt(h) for h in WIDTHS]
    table = lambda key, fmt: ", ".join(f"h={h}: {w[h][key]:{fmt}}" for h in WIDTHS)  # noqa: E731
    return [
        practice.Check("ANSWER: yes for the range one net covers — it more than triples "
                       "from h=2 to h=32",
                       w[32]["span"] > 2.9 * w[2]["span"],
                       "mean output range over the 200 points, one net at a time — "
                       + table("span", ".4f") + f". The lesson's own 2-8-1 net spans "
                       f"{result['lesson_span']:.4f}"),
        practice.Check("…and yes for the distribution, which goes from unimodal at 0.5 "
                       "to saturated at both ends",
                       w[2]["lo"] > 0.2 and w[128]["lo"] < 1e-5 and w[128]["hi"] > 0.9999,
                       f"pooled over {NETS} nets x {result['n']} points, "
                       + ", ".join(f"h={h}: [{w[h]['lo']:.2e}, {w[h]['hi']:.6f}] sd "
                                   f"{w[h]['sd']:.4f}" for h in WIDTHS)
                       + " — at h=2 no forward pass anywhere in the sweep reaches 0.2 or 0.8"),
        practice.Check("WHY: the output neuron sums h roughly equal terms of random sign, "
                       "so its pre-activation spreads as sqrt(h)",
                       max(ratio) - min(ratio) < 0.03 and abs(ratio[3] - 0.30) < 0.02,
                       "sd of the output layer's pre-activation, " + table("zsd", ".4f")
                       + " — divided by sqrt(h) that is "
                       + ", ".join(f"{r:.3f}" for r in ratio)
                       + f" — constant to within {max(ratio) - min(ratio):.3f} across a 64x "
                       "change in h. Var(z) = h * Var(w) * E[a^2] with w ~ U(-1, 1) and "
                       "hidden activations near 0.5, i.e. sqrt(h/3 * 0.25) = 0.289 * sqrt(h). "
                       "Sigmoid then squashes the wider z, which is why the ends fill in"),
        practice.Check("FINDING: none of that reaches the accuracy, which is flat in h",
                       max(w[h]["acc"] for h in WIDTHS) - min(w[h]["acc"] for h in WIDTHS) < 3,
                       "mean accuracy " + table("acc", ".1f")
                       + " %. MECHANISM: the decision is sigmoid(z) >= 0.5, i.e. z >= 0, and "
                       "widening rescales z symmetrically about zero — it changes how far "
                       "from the threshold the net lands, never which side"),
        practice.Check("FINDING: at every width, a random net is a constant classifier and "
                       "never beats the majority class",
                       max(w[h]["best"] for h in WIDTHS) <= majority
                       and min(w[h]["const"] for h in WIDTHS) > 0.6 * NETS,
                       f"best of {NETS * len(WIDTHS)} nets is {max(w[h]['best'] for h in WIDTHS):.1f}%, "
                       f"exactly the {majority:.1f}% of always answering 'outside'; worst is "
                       f"{min(w[h]['worst'] for h in WIDTHS):.1f}%, exactly always 'inside'. "
                       + table("const", "d") + " of 400 nets give one answer for all 200 "
                       f"points. The lesson's printed {result['lesson_acc']:.1f}% is that "
                       f"floor: its 2-8-1 net answers 'inside' everywhere, so it scores the "
                       f"{result['inside']} positives and nothing else"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
