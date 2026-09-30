"""Exercise 3 -- the lesson's gloo pipeline steps on the sum of microbatch gradients, 4x the full-batch gradient.

    Add gradient accumulation across pipeline microbatches and check the gradient equals the gradient of the equivalent full-batch forward.

Reading of the exercise: the setup is the lesson's own 2-stage pipeline:
`StageMLP(16, 32, 16)` then `StageMLP(16, 32, 4)`, seeded 23 and 24 as its
ranks seed them, M = 4 microbatches of 8 drawn from its generator (seed
122), MSE against zeros. Accumulation is written the pipeline way: stage 1
runs on a detached copy of stage 0's activation, and the activation's
gradient is handed back to stage 0's backward, one microbatch at a time,
with no zeroing in between. It is compared with one forward and backward on
the concatenated 32 rows. Then the lesson's real gloo pipeline (`_pipe_worker`,
2 ranks) is run and the gradient it hands `optim.step()` is recorded.

**ANSWER: with each microbatch loss scaled by 1/M, the accumulated gradient
equals the full-batch gradient** on both stages, to within float32 rounding
(under 1e-8, asserted below 1e-6). Without the scale it is exactly M = 4
times the full-batch gradient.

**FINDING: the lesson's gloo pipeline already accumulates, but steps on the
sum, not the mean.** `_pipe_worker` calls `loss.backward()` per microbatch
with an unscaled `MSELoss`, so the gradient `optim.step()` sees on each rank
is 4.0 times the full-batch gradient (relative error under 1e-5). The doc
says "the gradient at the end of a pipeline step is the gradient on the
combined M*B examples"; in the code it is M times that, so the effective
learning rate is 0.05 x M, not 0.05.

Structure: `launch()` starts this file once per rank with a 60 s timeout and
kills stragglers; `worker()` runs the lesson's `_pipe_worker` with a global
optimizer pre-step hook that records the gradients. Expected output: two
PASS checks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types

from harness import parity, practice

try:
    import torch
    from torch.optim import optimizer
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "79-pipeline-parallel"
BATCH, M = 8, 4


def stages_and_data(ref):
    stages = []
    for rank, dims in enumerate([(16, 32, 16), (16, 32, 4)]):
        torch.manual_seed(ref.SEED + rank)
        stages.append(ref.StageMLP(*dims))
    g = torch.Generator().manual_seed(ref.SEED + 99)
    return stages, [torch.randn(BATCH, 16, generator=g) for _ in range(M)]


def grads(stages):
    out = [p.grad.clone() for s in stages for p in s.parameters()]
    for s in stages:
        s.zero_grad(set_to_none=True)
    return out


def pipelined(stages, xs, scale):
    """Accumulate across microbatches the way a 2-stage pipeline does it."""
    for x in xs:
        act = stages[0](x)
        recv = act.detach().requires_grad_(True)
        pred = stages[1](recv)
        (torch.nn.functional.mse_loss(pred, torch.zeros_like(pred)) * scale).backward()
        act.backward(recv.grad)
    return grads(stages)


def full_batch(stages, xs):
    pred = stages[1](stages[0](torch.cat(xs)))
    torch.nn.functional.mse_loss(pred, torch.zeros_like(pred)).backward()
    return grads(stages)


def worker(rank, init_file):
    ref = parity.load_reference(PHASE, LESSON, "main")
    seen = []
    hook = optimizer.register_optimizer_step_pre_hook(
        lambda opt, a, k: seen.append([p.grad.tolist() for p in opt.param_groups[0]["params"]]))

    def put(item):
        hook.remove()
        print(json.dumps({"rank": item[0], "grads": seen[0]}), flush=True)

    queue = types.SimpleNamespace(put=put, close=lambda: None, join_thread=lambda: None)
    ref._pipe_worker(int(rank), 2, init_file, ref._loopback_iface(), 1, BATCH, M, queue)


def launch(timeout=60):
    with tempfile.TemporaryDirectory() as tmp:
        argv = [sys.executable, __file__, "--rank", os.path.join(tmp, "rendezvous")]
        pipe = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
        procs = [subprocess.Popen(argv, env=os.environ | {"RANK": str(r)}, **pipe) for r in range(2)]
        try:
            outs = [p.communicate(timeout=timeout) for p in procs]
        finally:
            for p in procs:
                p.kill()
    if any(map(subprocess.Popen.poll, procs)):
        raise RuntimeError(f"a rank failed: {[err[-300:] for _, err in outs]}")
    # procs[r] ran with RANK=r, so outs are already in rank order
    return [torch.tensor(g) for out, _ in outs for g in json.loads(out.splitlines()[-1])["grads"]]


def worst(a, b, factor=1.0):
    return max(((x - factor * y).abs().max() / (factor * y).abs().max()).item() for x, y in zip(a, b))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    stages, xs = stages_and_data(ref)
    full = full_batch(stages, xs)
    mean, total = pipelined(stages, xs, 1 / M), pipelined(stages, xs, 1.0)
    lesson = launch()
    return {
        "mean_err": max((a - b).abs().max().item() for a, b in zip(mean, full)),
        "sum_ratio_err": worst(total, full, M),
        "lesson_vs_full_times_m": worst(lesson, full, M), "lesson_vs_full": worst(lesson, full),
        "lesson_vs_sum": worst(lesson, total), "tensors": len(lesson),
        "doc_claims_mean": "the gradient on the combined M*B examples" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: scaling each microbatch loss by 1/M makes the accumulated gradient the full-batch one",
            r["mean_err"] < 1e-6 and r["sum_ratio_err"] < 1e-5,
            f"max |accumulated - full batch| {r['mean_err']:.2g}; unscaled sum vs {M} x full batch, "
            f"relative {r['sum_ratio_err']:.2g}",
        ),
        practice.Check(
            "FINDING: the lesson's gloo pipeline steps on the sum of microbatch gradients, M times the mean",
            r["tensors"] == 8 and r["lesson_vs_full_times_m"] < 1e-5 and r["lesson_vs_sum"] < 1e-5
            and r["lesson_vs_full"] > 2.9 and r["doc_claims_mean"],
            f"8 recorded tensors vs {M} x full batch: relative {r['lesson_vs_full_times_m']:.2g}; vs the full "
            f"batch itself: {r['lesson_vs_full']:.2f} (i.e. {M}x); doc claims the combined-batch gradient",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        raise SystemExit(worker(os.environ["RANK"], sys.argv[2]))
    raise SystemExit(practice.selfcheck(globals()))
