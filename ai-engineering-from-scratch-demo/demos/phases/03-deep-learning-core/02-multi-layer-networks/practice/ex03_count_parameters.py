"""Exercise 3 — count_parameters on the classic MNIST stack, and what it counts.

    Implement a `count_parameters` method on the Network class that returns the
    total number of trainable weights and biases. Test it on a 784-256-128-10
    network (the classic MNIST architecture). How many parameters does it have?

Reading of the exercise: `count_parameters` already exists in the lesson's own `code/main.py`,
so "implement" is best read as "check that it counts what it claims to". Check 1 answers the
arithmetic against a closed form, check 2 against an independent count of the forward pass's own
work; checks 3-4 feed it networks whose declared and actual shapes disagree — what a parameter
count is for — and check 5 prices the answer.
"""

from __future__ import annotations

import random
import sys

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "02-multi-layer-networks"
MNIST = [784, 256, 128, 10]
SHALLOW = [784, 295, 10]


def stack(ref, sizes, rng):
    """The lesson's Layers, sized head to tail, with weights from a private RNG."""
    return ref.Network([ref.Layer(sizes[i - 1], sizes[i], weights=[[rng.uniform(-1, 1)
        for _ in range(sizes[i - 1])] for _ in range(sizes[i])]) for i in range(1, len(sizes))])


def closed_form(sizes) -> int:
    return sum(sizes[i - 1] * sizes[i] + sizes[i] for i in range(1, len(sizes)))


def sigmoid_calls(ref, net, x) -> int:
    """One forward pass, counting activations — an independent count of the biases."""
    calls, real = [], ref.sigmoid
    ref.sigmoid = lambda z: (calls.append(z), real(z))[1]              # noqa: E731
    net.forward(x)
    ref.sigmoid = real
    return len(calls)


def deep_bytes(obj) -> int:
    if isinstance(obj, list):
        return sys.getsizeof(obj) + sum(deep_bytes(v) for v in obj)
    return sys.getsizeof(obj)


def flat(ref, n_inputs, n_neurons):
    return ref.Layer(n_inputs, n_neurons, weights=[[0.01] * n_inputs] * n_neurons)


def mismatches(ref) -> dict:
    """Two networks whose declared shape and computed shape disagree."""
    ghost = ref.Network([flat(ref, 784, 256), flat(ref, 512, 128), flat(ref, 128, 10)])
    thin = ref.Network([ref.Layer(2, 5, weights=[[1.0, 1.0]])])
    return {"ghost": ghost.count_parameters(), "ghost_out": len(ghost.forward([0.5] * 784)),
            "thin": thin.count_parameters(), "thin_out": len(thin.forward([1.0, 1.0]))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    net = stack(ref, MNIST, random.Random(0))
    total = net.count_parameters()
    weights = sum(MNIST[i - 1] * MNIST[i] for i in range(1, len(MNIST)))
    return {"total": total, "closed": closed_form(MNIST), "weights": weights,
            "biases": total - weights, "first": MNIST[0] * MNIST[1] + MNIST[1],
            "calls": sigmoid_calls(ref, net, [0.5] * 784), "shallow": closed_form(SHALLOW),
            "bytes": sum(deep_bytes(l.weights) + deep_bytes(l.biases) for l in net.layers),
            **mismatches(ref)}


def verify(result):
    r, per = result, [MNIST[i - 1] * MNIST[i] + MNIST[i] for i in range(1, 4)]
    total, ghost, byt = r["total"], r["ghost"], r["bytes"]
    return [
        practice.Check("ANSWER: the 784-256-128-10 network has 235,146 parameters",
                       total == 235146 == r["closed"],
                       f"{per[0]:,} + {per[1]:,} + {per[2]:,} = {total:,}, matching the closed form "
                       f"sum(n_prev * n + n) exactly. {r['weights']:,} are weights and {r['biases']} "
                       f"are biases, so the biases are {100 * r['biases'] / total:.2f}% of the model "
                       f"— and the very first weight matrix alone is {100 * r['first'] / total:.1f}% "
                       f"of it. A shallow 784-295-10 costs {r['shallow']:,}, within "
                       f"{100 * abs(r['shallow'] - total) / total:.1f}% of the same budget for one "
                       f"hidden layer instead of two"),
        practice.Check("…and the bias count is confirmed by counting the forward pass's own work",
                       r["calls"] == r["biases"] == 394,
                       f"one forward pass calls `sigmoid` exactly {r['calls']} times, once per "
                       f"neuron, which is the same {r['biases']} the method sums from "
                       f"len(layer.biases) — 256 + 128 + 10"),
        practice.Check("FINDING: it counts weights the forward pass never reads",
                       ghost == 267914 and r["ghost_out"] == 10,
                       f"declare the middle layer Layer(512, 128) between a 256-wide layer and the "
                       f"output and the count reports {ghost:,} against {total:,} — "
                       f"{ghost - total:,} too many, {100 * (ghost - total) / total:.1f}% — while "
                       f"the network still returns {r['ghost_out']} outputs and raises nothing. "
                       f"MECHANISM: `forward` zips weights against inputs, so the 256 unmatched "
                       f"columns per neuron are dropped at run time and counted at rest"),
        practice.Check("…and it can double-count a layer's two contradictory shapes at once",
                       r["thin"] == 7 and r["thin_out"] == 1,
                       f"Layer(2, 5, weights=[[1.0, 1.0]]) declares 5 neurons but supplies one row. "
                       f"`forward` loops over len(self.weights) and returns {r['thin_out']} value; "
                       f"`count_parameters` takes the weights from the rows and the biases from "
                       f"n_neurons and reports {r['thin']} = 2 + 5. Neither half is checked"),
        practice.Check("what 235,146 pure-Python parameters cost to hold",
                       31 < byt / total < 35,
                       f"{byt:,} bytes = {byt / 1e6:.2f} MB, {byt / total:.1f} bytes per parameter "
                       f"— an 8-byte list slot plus a 24-byte float object each. The same weights "
                       f"as float32 would be {4 * total / 1e6:.2f} MB, so the from-scratch "
                       f"representation costs {byt / (4 * total):.1f}x the array a framework "
                       f"would allocate"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
