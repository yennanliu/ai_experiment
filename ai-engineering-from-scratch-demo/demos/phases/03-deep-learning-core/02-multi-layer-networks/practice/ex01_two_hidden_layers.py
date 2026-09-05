"""Exercise 1 — a 2-4-2-1 forward pass, and what a second hidden layer does to XOR.

    Build a 2-4-2-1 network (two hidden layers) and run the forward pass on XOR
    data with random weights. Print the intermediate hidden layer outputs to see
    how the representation transforms at each layer.

Reading of the exercise: "see how the representation transforms" is the point, so
the checks measure the transform rather than narrate it — check 2 tracks the
distance between the two inputs XOR has to separate, layer by layer. It turns out
to be a contraction, so check 3 asserts the mechanism, check 4 the consequence and
check 5 the depth control. Check 6 is the shape mismatch the lesson's objectives
ask the reader to be able to spot.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "02-multi-layer-networks"
XOR = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
DEEP, SHALLOW = [2, 4, 2, 1], [2, 4, 1]
TRIALS, WIDE = 2000, 20000


def rand_net(ref, rng, sizes, scale=1.0):
    """The lesson's own Layer, with its default init drawn from a private RNG."""
    return ref.Network([
        ref.Layer(sizes[i - 1], sizes[i],
                  weights=[[rng.uniform(-scale, scale) for _ in range(sizes[i - 1])]
                           for _ in range(sizes[i])]) for i in range(1, len(sizes))])


def trace(net, x) -> list:
    net.forward(x)
    return [list(x)] + [list(layer.last_output) for layer in net.layers]


def gaps(net) -> list:
    """||h([0,1]) - h([1,0])|| at the input and after each layer — XOR's hard pair."""
    return [math.dist(u, v) for u, v in zip(trace(net, [0, 1]), trace(net, [1, 0]))]


def contraction(ref, seed, sizes, trials=TRIALS) -> list:
    rng = random.Random(seed)
    return [statistics.mean(col)
            for col in zip(*[gaps(rand_net(ref, rng, sizes)) for _ in range(trials)])]


def solve_rate(ref, seed, scale, trials=WIDE) -> int:
    rng = random.Random(seed)
    nets = [rand_net(ref, rng, DEEP, scale) for _ in range(trials)]
    return sum(all((1 if n.forward(x)[0] >= 0.5 else 0) == y for x, y in XOR) for n in nets)


def truncated(ref) -> tuple:
    """One layer declared 8 wide but fed 4 values, against the correctly wired twin."""
    stack = lambda n: ref.Network([                                      # noqa: E731
        ref.Layer(2, 4, weights=[[0.5, 0.5]] * 4),
        ref.Layer(n, 2, weights=[[0.5] * n, [0.5] * n]),
        ref.Layer(2, 1, weights=[[0.5, 0.5]])])
    return stack(8).forward([1, 1])[0], stack(4).forward([1, 1])[0]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    deep = contraction(ref, 1, DEEP)
    return {
        "trace": [trace(rand_net(ref, random.Random(0), DEEP), x) for x, _ in XOR],
        "deep": deep, "shallow": contraction(ref, 1, SHALLOW),
        "ratios": [deep[i + 1] / deep[i] for i in range(3)],
        "origins": [trace(rand_net(ref, random.Random(s), DEEP), [0, 0])[1] for s in range(20)],
        "flat": solve_rate(ref, 11, 1.0), "sharp": solve_rate(ref, 11, 20.0),
        "truncated": truncated(ref),
    }


def verify(result):
    deep, ratios = result["deep"], result["ratios"]
    layers = " -> ".join(f"{g:.4f}" for g in deep)
    wide, right = result["truncated"]
    return [
        practice.Check("the fixture: one 2-4-2-1 forward pass, every layer's output",
                       [len(h) for h in result["trace"][1]] == [2, 4, 2, 1],
                       " | ".join("[" + ", ".join(f"{v:.4f}" for v in h) + "]"
                                  for h in result["trace"][1]) + " for input [0, 1]"),
        practice.Check("FINDING: the representation does not transform, it contracts",
                       deep[3] / deep[0] < 0.01,
                       f"mean ||h([0,1]) - h([1,0])|| over {TRIALS} random nets, input to "
                       f"output: {layers} — the pair XOR must separate reaches the output "
                       f"{deep[0] / deep[3]:.0f}x closer together than it started"),
        practice.Check("CONTROL: one hidden layer less and the same pair stays "
                       f"{result['shallow'][2] / deep[3]:.1f}x further apart",
                       result["shallow"][2] > 5 * deep[3],
                       f"2-4-1 under identical init ends {result['shallow'][2]:.4f} apart "
                       f"against {deep[3]:.4f} for 2-4-2-1 — the loss is the extra layer, "
                       f"not the fixture"),
        practice.Check("MECHANISM: every layer shrinks distances, and the origin carries "
                       "no information at all",
                       max(ratios) < 0.27 and all(h == [0.5] * 4 for h in result["origins"]),
                       f"per-layer ratios {ratios[0]:.3f}, {ratios[1]:.3f}, {ratios[2]:.3f} "
                       f"— sigmoid is 1/4-Lipschitz and U(-1, 1) weights have mean |w| = "
                       f"0.5, so the composed map is a contraction. Separately the default "
                       f"biases are all 0.0, so [0, 0] leaves the first hidden layer as "
                       f"exactly [0.5, 0.5, 0.5, 0.5] in all 20 seeds tried"),
        practice.Check("ANSWER: no random 2-4-2-1 net gets XOR right",
                       result["flat"] == 0 and result["sharp"] > 0,
                       f"{result['flat']} of {WIDE} nets at the lesson's U(-1, 1) init solve "
                       f"all four rows; at U(-20, 20), the magnitude the lesson hand-tunes "
                       f"to, {result['sharp']} of {WIDE} do. Depth alone buys nothing — the "
                       f"weights have to survive the contraction"),
        practice.Check("FINDING: the forward pass never checks a dimension",
                       wide == right,
                       f"a middle Layer(8, 2) fed only 4 values returns {wide:.16f}, "
                       f"bit-identical to the correctly wired Layer(4, 2), and raises "
                       f"nothing. MECHANISM: `forward` pairs weights to inputs with zip(), "
                       f"which stops at the shorter — the four unmatched weights per neuron "
                       f"are silently dropped"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
