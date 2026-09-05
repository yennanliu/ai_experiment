"""Exercise 2 — what widening the circle classifier's hidden layer actually changes.

    Change the hidden layer size in the circle classifier from 8 to 2, then to 32.
    Run the forward pass with random weights each time. Does the number of hidden
    neurons change the output range or distribution? Why?

Reading of the exercise: "the output range" is ambiguous and the ambiguity is the
finding, so check 1 reports the range one net covers and check 2 the range the ensemble
covers — they answer differently. "Why?" is check 3, a sqrt(h) law measured at four
widths; checks 4 and 5 ask what that buys, since the classifier does not move at all.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "02-multi-layer-networks"
WIDTHS = [2, 8, 32, 128]
NETS, SEED = 400, 7
table = lambda w, k, f: ", ".join(f"h={h}: {w[h][k]:{f}}" for h in WIDTHS)   # noqa: E731


def circle(seed=42, n=200) -> list:      # the lesson's own dataset, its own seed
    rng = random.Random(seed)
    pts = [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)]
    return [([x, y], 1 if x * x + y * y < 0.25 else 0) for x, y in pts]


def build(ref, rng, h):
    return ref.Network([
        ref.Layer(2, h, weights=[[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(h)]),
        ref.Layer(h, 1, weights=[[rng.uniform(-1, 1) for _ in range(h)]])])


def run(net, data) -> list:     # this net's output over every point in the dataset
    return [net.forward(x)[0] for x, _t in data]


def scoreboard(outs, data) -> dict:   # accuracy, output range, constant-answer count
    accs = [100 * sum((v >= 0.5) == bool(t) for v, (_x, t) in zip(r, data)) / len(data)
            for r in outs]
    return {"span": statistics.mean(max(r) - min(r) for r in outs),
            "acc": statistics.mean(accs), "best": max(accs), "worst": min(accs),
            "const": sum(len({v >= 0.5 for v in r}) == 1 for r in outs)}


def survey(ref, data, h) -> dict:
    rng = random.Random(SEED)
    outs = [run(build(ref, rng, h), data) for _ in range(NETS)]
    pooled = [v for r in outs for v in r]
    return {"lo": min(pooled), "hi": max(pooled), "sd": statistics.pstdev(pooled),
            # biases default to 0.0, so logit(output) recovers the output neuron's z exactly
            "zsd": statistics.pstdev([math.log(v / (1 - v)) for v in pooled]),
            **scoreboard(outs, data)}


def summary(w) -> dict:     # cross-width scalars, so `verify` only compares numbers
    ratio, accs = [w[h]["zsd"] / math.sqrt(h) for h in WIDTHS], [w[h]["acc"] for h in WIDTHS]
    return {"ratio": ratio, "ratio_spread": max(ratio) - min(ratio),
            "acc_spread": max(accs) - min(accs), "best": max(w[h]["best"] for h in WIDTHS),
            "worst": min(w[h]["worst"] for h in WIDTHS),
            "const": min(w[h]["const"] for h in WIDTHS)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    data = circle()
    lesson = scoreboard([[build(ref, random.Random(SEED), 8).forward(x)[0]
                          for x, _ in data]], data)
    w = {h: survey(ref, data, h) for h in WIDTHS}
    return {"by_width": w, "inside": sum(t for _x, t in data), "n": len(data),
            "lesson_span": lesson["span"], "lesson_acc": lesson["acc"], **summary(w)}


def verify(result):
    r, w = result, result["by_width"]
    majority = 100 * (r["n"] - r["inside"]) / r["n"]
    return [
        practice.Check("ANSWER: yes for the range one net covers — it more than triples by h=32",
                       w[32]["span"] > 2.9 * w[2]["span"],
                       "mean output range over the 200 points, one net at a time — "
                       + table(w, "span", ".4f")
                       + f". The lesson's own 2-8-1 net spans {r['lesson_span']:.4f}"),
        practice.Check("…and yes for the distribution: unimodal at 0.5, then both ends",
                       w[2]["lo"] > 0.2 and w[128]["lo"] < 1e-5 and w[128]["hi"] > 0.9999,
                       f"pooled over {NETS} nets x {r['n']} points, "
                       + ", ".join(f"h={h}: [{w[h]['lo']:.2e}, {w[h]['hi']:.6f}] sd "
                                   f"{w[h]['sd']:.4f}" for h in WIDTHS)
                       + " — at h=2 nothing in the sweep reaches 0.2 or 0.8"),
        practice.Check("WHY: the output neuron's pre-activation spreads as sqrt(h)",
                       r["ratio_spread"] < 0.03 and abs(r["ratio"][3] - 0.30) < 0.02,
                       "sd of the output layer's pre-activation, " + table(w, "zsd", ".4f")
                       + " — over sqrt(h) that is " + ", ".join(f"{v:.3f}" for v in r["ratio"])
                       + f", constant to within {r['ratio_spread']:.3f} across a 64x change in "
                       "h. MECHANISM: Var(z) = h*Var(w)*E[a^2] = h/3 * 0.25, sd 0.289*sqrt(h)"),
        practice.Check("FINDING: none of that reaches the accuracy, which is flat in h",
                       r["acc_spread"] < 3,
                       "mean accuracy " + table(w, "acc", ".1f")
                       + f" %, a spread of {r['acc_spread']:.1f} points. MECHANISM: the "
                       "decision is z >= 0 and widening rescales z symmetrically about it, so "
                       "it moves how far from the threshold a net lands, never which side"),
        practice.Check("FINDING: every width gives a constant classifier at the majority rate",
                       r["best"] <= majority and r["const"] > 0.6 * NETS,
                       f"best of {NETS * len(WIDTHS)} nets is {r['best']:.1f}%, exactly the "
                       f"{majority:.1f}% of always saying 'outside'; worst {r['worst']:.1f}% is "
                       f"always 'inside'. " + table(w, "const", "d") + f" of {NETS} nets give "
                       f"one answer for all {r['n']} points, and the lesson's printed "
                       f"{r['lesson_acc']:.1f}% is that floor — 'inside' everywhere, scoring "
                       f"its {r['inside']} positives"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
