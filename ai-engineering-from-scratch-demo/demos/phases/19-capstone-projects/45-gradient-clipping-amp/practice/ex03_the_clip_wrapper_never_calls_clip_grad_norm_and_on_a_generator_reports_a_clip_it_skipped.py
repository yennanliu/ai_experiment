"""Exercise 3 — the no-clip test passes, but the "wrapper" never calls clip_grad_norm_ and silently skips clipping when handed a generator.

    Add a unit test that the gradient-clip wrapper returns the pre-clip and post-clip norm correctly when no clipping occurs.

Reading of the exercise: the test is `no_clip_failures(clip)`: it takes a
clip function, so the same test can grade the lesson's
`clip_global_l2_norm` and two broken versions of it. It uses real gradients
from one backward pass of the lesson's seed-7 toy model (global norm
1.00801) and two thresholds that should not clip: 10.0, and the norm
itself. For each it asserts pre == post == `compute_global_l2_norm`, that
the norm agrees with `torch.nn.utils.clip_grad_norm_` to 1e-6 relative, and
that every gradient tensor is bit-identical afterwards. The broken versions
show the test can fail: one always reports `post = max_norm`, the other
rescales the gradients to the threshold even when it should not.

**ANSWER: the lesson's wrapper passes the test with 0 failures, and both
broken versions fail it**, each on one assertion at threshold 10.0 (at the
norm itself they coincide with the correct answer: post = max_norm = pre,
and a rescale by norm/norm = 1). When no clipping
occurs the lesson returns pre == post == 1.0080091 (float64). That differs
from torch's float32 total norm by 4.9e-8. The gradients are untouched,
including at a threshold exactly equal to the norm.

**FINDING: the "wrapper around clip_grad_norm_" never calls it.** Both the
doc and the docstring say it wraps `torch.nn.utils.clip_grad_norm_`, but the
function body re-implements the clip by hand. The two also differ at the
boundary. At `max_norm` equal to the norm, torch still multiplies every
gradient by max/(norm + 1e-6) and changes them, while the lesson leaves
them bit-identical.

**FINDING: handed `model.parameters()`, the wrapper reports a clip it did not
do.** Asked to clip to 0.1, `clip_global_l2_norm(model.parameters(), 0.1)`
returns (1.008, 0.1), but the gradient norm afterwards is still 1.008. The
norm pass uses up the generator, so the scaling loop runs over nothing.
`clip_grad_norm_` accepts a generator. The lesson's `AmpTrainState` passes
a `list`, so the shipped loop is not affected.

Expected output: three PASS checks.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "45-gradient-clipping-amp"


def fresh_grads(ref):
    model, inputs, targets = ref.build_toy_model()
    torch.nn.functional.mse_loss(model(inputs), targets).backward()
    return model


def no_clip_failures(ref, clip):
    """The unit test: returns the failed assertions (empty means it passes)."""
    failures = []
    norm = ref.compute_global_l2_norm(fresh_grads(ref).parameters())
    for max_norm in (10.0, norm):
        model = fresh_grads(ref)
        params = list(model.parameters())
        before = [p.grad.clone() for p in params]
        pre, post = clip(params, max_norm)
        torch_norm = float(torch.nn.utils.clip_grad_norm_(fresh_grads(ref).parameters(), max_norm))
        failures += [f"max_norm={max_norm}: {msg}" for msg, ok in (
            ("pre != post", pre == post),
            ("pre != global norm", pre == norm),
            ("pre disagrees with clip_grad_norm_", abs(pre - torch_norm) <= 1e-6 * torch_norm),
            ("grads changed", all(torch.equal(a, p.grad) for a, p in zip(before, params))),
        ) if not ok]
    return failures


def mutants(ref):
    def reports_max_norm(params, max_norm):
        return ref.clip_global_l2_norm(params, max_norm)[0], max_norm

    def always_rescales(params, max_norm):
        pre = ref.compute_global_l2_norm(params)
        for p in params:
            p.grad.mul_(max_norm / pre)
        return pre, pre

    return {"reports_max_norm": reports_max_norm, "always_rescales": always_rescales}


def boundary_changes_grads(ref, clip):
    model = fresh_grads(ref)
    params = list(model.parameters())
    before = [p.grad.clone() for p in params]
    clip(params, ref.compute_global_l2_norm(params))
    return not all(torch.equal(a, p.grad) for a, p in zip(before, params))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    model = fresh_grads(ref)
    body = inspect.getsource(ref.clip_global_l2_norm).split('"""')[2]
    return {
        "lesson": no_clip_failures(ref, ref.clip_global_l2_norm),
        "mutants": {name: len(no_clip_failures(ref, fn)) for name, fn in mutants(ref).items()},
        "norm": ref.compute_global_l2_norm(model.parameters()),
        "torch_gap": abs(ref.compute_global_l2_norm(model.parameters())
                         - float(torch.nn.utils.clip_grad_norm_(fresh_grads(ref).parameters(), 1e9))),
        "calls_torch": "clip_grad_norm_" in body,
        "boundary": {"lesson": boundary_changes_grads(ref, ref.clip_global_l2_norm),
                     "torch": boundary_changes_grads(ref, torch.nn.utils.clip_grad_norm_)},
        "generator": (ref.clip_global_l2_norm(model.parameters(), 0.1), ref.compute_global_l2_norm(model.parameters())),
    }


def verify(result):
    (pre, post), after = result["generator"]
    return [
        practice.Check(
            "ANSWER: the lesson's wrapper passes the no-clip test and both broken versions fail it",
            result["lesson"] == [] and result["mutants"] == {"reports_max_norm": 1, "always_rescales": 1}
            and round(result["norm"], 6) == 1.008009 and 1e-9 < result["torch_gap"] < 1e-6,
            f"lesson failures {result['lesson']}; mutant failures {result['mutants']}; norm "
            f"{result['norm']:.7f}, float32 torch gap {result['torch_gap']:.2g}",
        ),
        practice.Check(
            "FINDING: the 'wrapper around clip_grad_norm_' never calls it, and differs at the boundary",
            not result["calls_torch"] and result["boundary"] == {"lesson": False, "torch": True},
            f"calls clip_grad_norm_: {result['calls_torch']}; grads changed at max_norm == norm: {result['boundary']}",
        ),
        practice.Check(
            "FINDING: handed model.parameters(), it reports a clip to 0.1 and leaves the norm at 1.008",
            post == 0.1 and round(pre, 6) == round(after, 6) == 1.008009,
            f"returned ({pre:.4f}, {post}); norm afterwards {after:.4f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
