"""Exercise 4 — a linear layer's analytical backward pass, gradient-checked.

    **Gradient checking a neural network layer.** Implement a single linear layer
    `y = Wx + b` and its analytical backward pass. Use `numerical_gradient` to
    verify correctness for a 3x2 weight matrix.

Reading of the exercise: `numerical_gradient(f, x, h)` differentiates a scalar
function, so the layer needs a scalar to differentiate — a sum-of-outputs
objective is used, which makes every entry of dW and db non-trivial. The check
that matters is not "gradients agree" but check 4: a deliberately *wrong*
backward pass must be caught. A gradient check that has never rejected anything
is not a check.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "13-numerical-stability"
OUT, IN = 3, 2
W = [[0.5, -0.25], [1.0, 0.75], [-0.5, 0.125]]
B = [0.1, -0.2, 0.3]
X = [1.5, -2.0]


def forward(weights, bias, x):
    return [sum(row[j] * x[j] for j in range(IN)) + bias[i]
            for i, row in enumerate(weights)]


def objective(flat):
    """Σ y, as a function of the flattened [W | b] parameters."""
    weights = [flat[i * IN:(i + 1) * IN] for i in range(OUT)]
    bias = flat[OUT * IN:]
    return sum(forward(weights, bias, X))


def backward(x, upstream=None):
    """d(Σy)/dW = 1·xᵀ and d(Σy)/db = 1, flattened to match `objective`."""
    upstream = upstream or [1.0] * OUT
    grad_w = [g * xj for g in upstream for xj in x]
    grad_b = list(upstream)
    return grad_w + grad_b


def solve():
    ref = parity.load_reference(PHASE, LESSON, "numerical")
    flat = [v for row in W for v in row] + list(B)
    analytical = backward(X)
    numerical = ref.numerical_gradient(objective, flat)
    wrong = [g * 1.05 for g in analytical]            # 5% off, uniformly
    subtle = list(analytical)
    subtle[1] = -subtle[1]                            # one sign flipped
    return {
        "analytical": analytical, "numerical": list(numerical),
        "n_params": len(flat), "output": forward(W, B, X),
        "ok": ref.check_gradient(analytical, numerical),
        "rejects_wrong": not ref.check_gradient(wrong, numerical),
        "rejects_subtle": not ref.check_gradient(subtle, numerical),
        "worst": max(abs(a - b) for a, b in zip(analytical, numerical)),
    }


def verify(result):
    return [
        practice.Check(f"a {OUT}x{IN} layer has {result['n_params']} parameters, "
                       f"all differentiated",
                       result["n_params"] == OUT * IN + OUT == 9,
                       f"6 weights + 3 biases; forward pass y = "
                       f"{[round(v, 4) for v in result['output']]}"),
        practice.Check("analytical and numerical gradients agree",
                       result["ok"] and result["worst"] < 1e-6,
                       f"worst absolute difference {result['worst']:.3g}; "
                       f"dW = 1·xᵀ so each row repeats x = {X}"),
        practice.Check("check_gradient accepts the correct pass",
                       result["ok"], "the lesson's own tolerance, unmodified"),
        practice.Check("…and rejects a backward pass that is 5% wrong",
                       result["rejects_wrong"],
                       "a uniformly scaled gradient is the classic bug — a missing "
                       "1/batch_size or a doubled learning-rate term — and the check "
                       "catches it"),
        practice.Check("…and rejects a single flipped sign",
                       result["rejects_subtle"],
                       "one of nine entries negated. This is the check that makes the "
                       "other four mean anything: a gradient check that has never "
                       "rejected a wrong gradient has not been tested"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
