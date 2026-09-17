"""Exercise 1 — the gradients are bit-identical at every k, and the one thing that breaks this is missing.

    Verify correctness. Run `model_forward` + `model_backward` (full
    activations) vs `model_forward_checkpointed` + `model_backward_checkpointed`
    (segments). Parameter gradients must be identical to machine precision.

Reading of the exercise: the comparison is run exactly as written, at every
segment size from 1 to L, through the lesson's own `verify_equivalence`. Two
further arms are added: a deliberately corrupted saved input, to establish that
the test discriminates at all, and the same model with dropout, because that is
the one place where a checkpointed backward can legitimately disagree.

**ANSWER: the maximum parameter-gradient difference is exactly 0.0 at every k
from 1 to 8** -- not "within machine precision" but bitwise equal. Nothing here
needs a tolerance.

**MECHANISM: `model_backward` never reads the stored intermediates.**
`layer_backward` recomputes `h_pre` and `h` from `x_in`, so the only thing the
"full activations" path uses from its `activations` list is the per-layer input
-- which is exactly what the checkpointed path saves. At k=1 the two lists are
element-for-element identical, so the exercise's "full activations vs segments"
compares two paths that store the same tensors and recompute the same
intermediates from them, in the same order, in float32.

**FINDING: the test does discriminate, so the 0.0 is a real result.** Scaling one
saved segment input by 1.0001 moves the gradients by 2.7e-04, four orders of
magnitude above the 0.0 -- the check is sound and its subject is trivial.

**FINDING: the hazard the check exists for has nothing to attach to.** In real
checkpointing the recomputed forward must reproduce the original's random state;
the module has no stochastic op at all. Adding dropout and recomputing with a
fresh mask moves the gradients by **120% of their own magnitude** -- the same
seed returns to 0.0. That, not float error, is what "identical to machine
precision" is protecting against, and this model cannot exhibit it.

Structure: `arms` runs the lesson's own equivalence check across k; `dropout_run`
is the same network with a mask, so the fresh-vs-reused comparison is possible.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "34-gradient-checkpointing"
LAYERS, HIDDEN, INNER, BATCH = 6, 16, 32, 4
SEGMENTS = (1, 2, 3, 4, 5, 6, 7, 8)
RATE, SEED, PERTURB = 0.5, 5, 1.0001


def worst(left, right):
    """Largest absolute disagreement between two lists of gradient tuples."""
    return max(float(np.max(np.abs(a - b)))
               for first, second in zip(left, right) for a, b in zip(first, second))


def corrupted(ref, params, x, grad_out):
    """The same comparison with one saved segment input scaled, as a control."""
    _, saved = ref.model_forward_checkpointed(x, params, k=2)
    _, clean = ref.model_backward_checkpointed(grad_out, saved, params, k=2)
    saved[1] = saved[1] * PERTURB
    _, dirty = ref.model_backward_checkpointed(grad_out, saved, params, k=2)
    return worst(clean, dirty)


def dropout_run(ref, params, x, rng):
    """Forward keeping the inputs and the masks; the masks are the recomputed state."""
    activations, masks, h = [x], [], x
    for w1, b1, w2, b2 in params:
        pre = ref.linear_forward(h, w1, b1)
        mask = ((rng.random(pre.shape) > RATE) / (1 - RATE)).astype(np.float32)
        masks.append(mask)
        h = ref.linear_forward(ref.relu(pre) * mask, w2, b2)
        activations.append(h)
    return activations, masks


def dropout_backward(ref, grad_out, activations, params, masks):
    grads, g = [], grad_out
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        pre = ref.linear_forward(activations[i], w1, b1)
        hidden = ref.relu(pre) * masks[i]
        g_pre = ((g @ w2.T) * masks[i]) * (pre > 0)
        grads.append((activations[i].T @ g_pre, hidden.T @ g))
        g = g_pre @ w1.T
    return grads[::-1]


def dropout_gap(ref, params, x, grad_out):
    """Recompute the masks from a fresh stream, then from the original seed."""
    activations, masks = dropout_run(ref, params, x, np.random.default_rng(SEED))
    truth = dropout_backward(ref, grad_out, activations, params, masks)
    _, fresh = dropout_run(ref, params, x, np.random.default_rng(SEED + 100))
    _, again = dropout_run(ref, params, x, np.random.default_rng(SEED))
    scale = max(float(np.max(np.abs(a))) for row in truth for a in row)
    return {
        "fresh": worst(truth, dropout_backward(ref, grad_out, activations, params, fresh)) / scale,
        "reused": worst(truth, dropout_backward(ref, grad_out, activations, params, again)),
        "stochastic_names": [name for name in dir(ref)
                             if any(word in name.lower()
                                    for word in ("dropout", "random", "rng", "seed"))],
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((BATCH, HIDDEN)).astype(np.float32)
    params = ref.make_params(LAYERS, HIDDEN, INNER)
    out, activations = ref.model_forward(x, params)
    grad_out = rng.standard_normal(out.shape).astype(np.float32)
    _, saved = ref.model_forward_checkpointed(x, params, k=1)
    return {
        "arms": {k: ref.verify_equivalence(n_layers=LAYERS, hidden=HIDDEN, inner=INNER,
                                           batch=BATCH, k=k) for k in SEGMENTS},
        "same_store": ([a.shape for a in activations] == [a.shape for a in saved]
                       and all(np.array_equal(a, b) for a, b in zip(activations, saved))),
        "stored": len(activations),
        "control": corrupted(ref, params, x, grad_out),
        "dropout": dropout_gap(ref, params, x, grad_out),
    }


def verify(result):
    arms, drop = result["arms"], result["dropout"]
    return [
        practice.Check(
            f"ANSWER: the gradient difference is exactly 0.0 at every k from 1 to {max(SEGMENTS)}",
            all(row["max_grad_diff"] == 0.0 and row["output_match"] for row in arms.values()),
            "verify_equivalence reports " + ", ".join(
                f"k={k} {row['max_grad_diff']}" for k, row in arms.items())
            + ". Not 'within machine precision' -- bitwise equal, at every segment size "
            "including the ones that do not divide the layer count, so no tolerance is in play",
        ),
        practice.Check(
            "MECHANISM: model_backward never reads the stored intermediates",
            result["same_store"],
            f"layer_backward recomputes h_pre and h from x_in, so the only thing the full path "
            f"uses from its activations list is the per-layer input -- which is what the "
            f"checkpointed path saves. At k=1 the two lists are both {result['stored']} tensors "
            "and element-for-element identical, so 'full activations vs segments' compares two "
            "paths that store the same tensors and recompute the same intermediates from them",
        ),
        practice.Check(
            "FINDING: the test does discriminate, so the 0.0 is a result and not a no-op",
            result["control"] > 1e-5,
            f"scaling one saved segment input by {PERTURB} moves the gradients by "
            f"{result['control']:.2e}, four orders of magnitude above the 0.0 the honest run "
            "reports. The check is sound; its subject is trivial",
        ),
        practice.Check(
            "FINDING: the hazard the check exists for is absent from the model",
            (not drop["stochastic_names"] and drop["fresh"] > 0.5
             and drop["reused"] == 0.0),
            f"real checkpointing must reproduce the original forward's random state, and the "
            f"module has no stochastic op -- {drop['stochastic_names']}. Adding dropout at rate "
            f"{RATE} and recomputing with a fresh mask moves the gradients by "
            f"{drop['fresh']:.2%} of their own magnitude; recomputing from the same seed returns "
            f"to {drop['reused']}. That, not float error, is what the exercise's threshold guards",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
