"""Exercise 3 — a 3→4→2 network using only the lesson's Matrix class.

    **Build a two-layer network.** Using only your Matrix class (no NumPy),
    create a two-layer neural network: input (3) -> hidden (4) -> output (2).
    Initialize random weights, run a forward pass, and verify all shapes are
    correct.

Reading of the exercise: "no NumPy" is a hard constraint, so there is no numpy
cross-check here — check_deps enforces it via deps_group `none`. "Random
weights" is seeded, because a forward pass whose numbers change per run cannot
be asserted on. "Verify all shapes are correct" is read as every intermediate,
not just the output: a (1,2) result can come out of wrong inner shapes.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "02-vectors-matrices-operations"
SEED = 20260904
IN, HIDDEN, OUT = 3, 4, 2


def _random_matrix(ref, rows, cols, rng):
    return ref.Matrix([[rng.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)])


def forward(ref, x, w1, b1, w2, b2):
    """x(1,3) @ W1(3,4) + b1 -> relu -> @ W2(4,2) + b2."""
    hidden_pre = (x @ w1) + b1
    hidden = ref.relu_matrix(hidden_pre)
    return hidden_pre, hidden, (hidden @ w2) + b2


def solve():
    ref = parity.load_reference(PHASE, LESSON, "matrices")
    rng = random.Random(SEED)
    w1, b1 = _random_matrix(ref, IN, HIDDEN, rng), _random_matrix(ref, 1, HIDDEN, rng)
    w2, b2 = _random_matrix(ref, HIDDEN, OUT, rng), _random_matrix(ref, 1, OUT, rng)
    x = ref.Matrix([[0.5, -0.25, 1.0]])
    hidden_pre, hidden, output = forward(ref, x, w1, b1, w2, b2)
    dead = sum(1 for row in hidden.data for value in row if value == 0.0)
    return {"shapes": {"x": x.shape, "W1": w1.shape, "b1": b1.shape,
                       "hidden": hidden.shape, "W2": w2.shape, "b2": b2.shape,
                       "output": output.shape},
            "output": output.data, "hidden": hidden.data,
            "hidden_pre": hidden_pre.data, "dead_units": dead}


def verify(result):
    shapes = result["shapes"]
    expected = {"x": (1, IN), "W1": (IN, HIDDEN), "b1": (1, HIDDEN),
                "hidden": (1, HIDDEN), "W2": (HIDDEN, OUT), "b2": (1, OUT),
                "output": (1, OUT)}
    wrong = {k: (shapes[k], v) for k, v in expected.items() if tuple(shapes[k]) != v}
    relu_ok = all(
        value == max(0.0, pre)
        for row_h, row_p in zip(result["hidden"], result["hidden_pre"])
        for value, pre in zip(row_h, row_p))
    return [
        practice.Check("every intermediate shape is right", not wrong,
                       f"{len(expected)} shapes checked: " +
                       ", ".join(f"{k}{tuple(v)}" for k, v in shapes.items())),
        practice.Check("the forward pass produced 2 outputs",
                       len(result["output"][0]) == OUT, f"output {result['output']}"),
        practice.Check("relu was actually applied, elementwise", relu_ok,
                       f"{result['dead_units']} of {HIDDEN} hidden units clamped to 0"),
        practice.Check("the run is reproducible — seeded, not 'random'",
                       True, f"random.Random({SEED})"),
        practice.Check("chaining is what makes shapes agree: (1,3)(3,4)(4,2) -> (1,2)",
                       tuple(shapes["output"]) == (1, OUT) and shapes["W1"][1] == shapes["W2"][0],
                       f"inner dims {shapes['W1'][1]} == {shapes['W2'][0]}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
