"""Exercise 3 — all five gradients of relu(w1·x1 + w2·x2 + b), against PyTorch.

    Build a computation graph for a single neuron: `y = relu(w1*x1 + w2*x2 +
    b)`. Compute all five gradients and verify against PyTorch.

Reading of the exercise: "all five" means the inputs get gradients too, not just
the parameters — x1 and x2 are leaves of this graph. The interesting part is the
choice of test point. ReLU is not differentiable at 0, and autodiff engines pick
a convention there rather than refusing; check 4 runs the pre-activation exactly
at 0 to find out which convention each engine picked, and whether they agree.

Tier T1: needs torch. Without it the exercise skips with a remedy rather than
quietly asserting less (`DESIGN D2`).
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "05-chain-rule-and-autodiff"
TOL = 1e-9
W1, W2, X1, X2, B = 2.0, -3.0, 1.5, 0.5, 1.0      # pre-activation = 2.5 > 0


def _mine(ref, w1, w2, x1, x2, b):
    values = [ref.Value(v) for v in (w1, w2, x1, x2, b)]
    vw1, vw2, vx1, vx2, vb = values
    y = (vw1 * vx1 + vw2 * vx2 + vb).relu()
    y.backward()
    return y.data, [v.grad for v in values]


def _torch(torch, w1, w2, x1, x2, b):
    tensors = [torch.tensor(v, requires_grad=True) for v in (w1, w2, x1, x2, b)]
    tw1, tw2, tx1, tx2, tb = tensors
    y = torch.relu(tw1 * tx1 + tw2 * tx2 + tb)
    y.backward()
    return y.item(), [t.grad.item() for t in tensors]


def _worst(a, b) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def solve():
    try:
        import torch
    except ImportError:
        raise practice.Skip("needs PyTorch — uv sync --extra llm") from None
    ref = parity.load_reference(PHASE, LESSON, "autodiff")
    mine_y, mine_g = _mine(ref, W1, W2, X1, X2, B)
    torch_y, torch_g = _torch(torch, W1, W2, X1, X2, B)
    # pre-activation exactly 0: relu's kink
    kink_mine = _mine(ref, 1.0, -1.0, 1.0, 1.0, 0.0)[1]
    kink_torch = _torch(torch, 1.0, -1.0, 1.0, 1.0, 0.0)[1]
    # and just below it, where relu is flat
    dead_mine = _mine(ref, 1.0, 1.0, -1.0, -1.0, 0.0)[1]
    return {"mine": (mine_y, mine_g), "torch": (torch_y, torch_g),
            "kink": (kink_mine, kink_torch), "dead": dead_mine,
            "version": torch.__version__}


def verify(result):
    (mine_y, mine_g), (torch_y, torch_g) = result["mine"], result["torch"]
    worst = _worst(mine_g, torch_g)
    names = ["w1", "w2", "x1", "x2", "b"]
    analytic = [X1, X2, W1, W2, 1.0]           # d/d· of (w1x1+w2x2+b) where relu is active
    kink_mine, kink_torch = result["kink"]
    grads_text = ", ".join(f"∂y/∂{n}={g:g}" for n, g in zip(names, mine_g))
    return [
        practice.Check("forward pass agrees with torch",
                       abs(mine_y - torch_y) <= TOL,
                       f"relu(2·1.5 + (−3)·0.5 + 1) = {mine_y} vs torch {torch_y}"),
        practice.Check("all five gradients exist and match torch",
                       len(mine_g) == 5 and worst <= TOL,
                       grads_text + f"; worst gap vs torch {worst:.3g} "
                       f"(torch {result['version']})"),
        practice.Check("…and match the analytic values, since relu is active here",
                       _worst(mine_g, analytic) <= TOL,
                       f"pre-activation 2.5 > 0, so ∂y/∂wᵢ = xᵢ and ∂y/∂xᵢ = wᵢ"),
        practice.Check("at the relu kink both engines choose the same subgradient",
                       _worst(kink_mine, kink_torch) <= TOL,
                       f"pre-activation exactly 0: mine {[round(g, 3) for g in kink_mine]}, "
                       f"torch {[round(g, 3) for g in kink_torch]} — relu is not "
                       f"differentiable there, so this is a convention, not a derivation"),
        practice.Check("a dead unit passes zero gradient to every leaf",
                       all(g == 0.0 for g in result["dead"]),
                       f"pre-activation −2: {[round(g, 3) for g in result['dead']]} — "
                       f"nothing upstream of a dead relu learns"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
