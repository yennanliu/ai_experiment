"""Exercise 4 -- checkpointing cuts the stash 6x, but spent on more microbatches it drops throughput 3.7%.

    Pair the pipeline with activation checkpointing and measure the memory drop versus compute cost.

Reading of the exercise: the pipeline is 4 stages of the lesson's
`StageMLP(64, 256, 64)`, M = 8 microbatches of 32 rows, run in the order of
the lesson's `gpipe_schedule` (every forward before any backward). Memory is
the activation bytes a stage keeps for its backward, counted exactly with
`torch.autograd.graph.saved_tensors_hooks` (parameters excluded, shared
storage counted once). Checkpointing wraps each stage in
`torch.utils.checkpoint` (non-reentrant). Compute is counted as stage
forward calls, then priced with the lesson's own `FORWARD_UNITS = 1`,
`BACKWARD_UNITS = 2`; wall-clock is reported as well.

**ANSWER: checkpointing cuts each stage's stashed activations from 8 x 48
KiB to 8 x 8 KiB, 6.0x (3.4x counting the one microbatch rebuilt during its
backward), and costs one extra forward per microbatch**: 16 forward calls
per stage instead of 8, priced at (1+1+2)/(1+2) = 1.33x the compute.
Gradients are bit-identical. Wall-clock grows more than the unit price
says, about 2.6x on this machine, because on stages this small the
checkpoint bookkeeping costs as much as the math (only > 1.05x is
asserted, since it depends on load). Without checkpointing
the stash is exactly linear in M, as the doc says: 1, 2, 4, 8 microbatches
keep 1, 2, 4, 8 times one microbatch's bytes.

**FINDING: spending the saved memory on more microbatches loses
throughput.** The memory GPipe needs at M = 8 without checkpointing holds
M = 42 with it, and the bubble falls from 27.3% to 6.7%. But the recompute
makes each backward 3 units instead of 2, so throughput is 42/(45 x 4) =
0.233 microbatches per unit, against 8/(11 x 3) = 0.242: 3.7% lower. On
these layers checkpointing buys memory, not speed.

Expected output: two PASS checks.
"""

from __future__ import annotations

import time

from harness import parity, practice

try:
    import torch
    from torch.utils.checkpoint import checkpoint
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "79-pipeline-parallel"
N, M, BATCH, DIM, HID = 4, 8, 32, 64, 256


def make_pack(params, store, mb):
    """Saved-tensor hook: record the bytes of every non-parameter tensor autograd stashes."""
    def pack(t):
        if t.data_ptr() not in params:
            store[(mb, t.data_ptr())] = t.numel() * t.element_size()
        return t
    return pack


def make_counted(calls):
    def counted(st, t):
        calls[0] += 1
        return st(t)
    return counted


def forward_mb(stages, x, mb, ctx):
    params, stash, counted, ckpt = ctx
    for k, st in enumerate(stages):
        with torch.autograd.graph.saved_tensors_hooks(make_pack(params, stash[k], mb), lambda t: t):
            x = checkpoint(counted, st, x, use_reentrant=False) if ckpt else counted(st, x)
    return x


def run(ref, stages, m, ckpt, calls=None):
    """GPipe order: all m forwards per stage, then all backwards. Returns stash bytes per stage."""
    params = {p.data_ptr() for s in stages for p in s.parameters()}
    stash = [dict() for _ in stages]
    g = torch.Generator().manual_seed(0)
    ctx = (params, stash, make_counted([0] if calls is None else calls), ckpt)
    outs = [forward_mb(stages, torch.randn(BATCH, DIM, generator=g), mb, ctx) for mb in range(m)]
    for x in reversed(outs):
        x.pow(2).mean().backward()
    return [sum(s.values()) for s in stash]


def build(ref):
    torch.manual_seed(ref.SEED)
    return [ref.StageMLP(DIM, HID, DIM) for _ in range(N)]


def measure(ref, ckpt):
    stages, calls = build(ref), [0]
    stash = run(ref, stages, M, ckpt, calls)
    grads = [p.grad.clone() for s in stages for p in s.parameters()]
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        run(ref, stages, M, ckpt)
        best = min(best, time.perf_counter() - t0)
    return {"stash": stash, "calls": calls[0], "grads": grads, "seconds": best}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    plain, ck = measure(ref, False), measure(ref, True)
    one = run(ref, build(ref), 1, False)[0]
    f, b = ref.FORWARD_UNITS, ref.BACKWARD_UNITS
    small = ck["stash"][0] // M
    m_ck = max(m for m in range(1, 1000) if m * small + one <= M * one)
    return {
        "per_mb": [plain["stash"][0] // M, ck["stash"][0] // M], "one": one,
        "linear": [run(ref, build(ref), m, False)[0] // one for m in (1, 2, 4, 8)],
        "calls": [plain["calls"] // N, ck["calls"] // N], "cost": (2 * f + b) / (f + b),
        "same_grads": all(torch.equal(a, c) for a, c in zip(plain["grads"], ck["grads"])),
        "wall": ck["seconds"] / plain["seconds"], "m_ck": m_ck,
        "bubble": [ref.bubble_fraction(N, M), ref.bubble_fraction(N, m_ck)],
        "throughput": [M / ((M + N - 1) * (f + b)), m_ck / ((m_ck + N - 1) * (2 * f + b))],
    }


def answer_check(r):
    drop, peak = r["per_mb"][0] / r["per_mb"][1], M * r["one"] / (M * r["per_mb"][1] + r["one"])
    return practice.Check(
        "ANSWER: checkpointing cuts the stash 6.0x for one extra forward per microbatch (1.33x compute)",
        r["per_mb"] == [49152, 8192] and r["linear"] == [1, 2, 4, 8] and r["calls"] == [8, 16]
        and round(r["cost"], 2) == 1.33 and r["same_grads"] and r["wall"] > 1.05,
        f"stash bytes per microbatch per stage {r['per_mb']} ({drop:.1f}x; {peak:.1f}x with one rebuilt), "
        f"linear in M {r['linear']}; forward calls per stage {r['calls']}, priced {r['cost']:.2f}x; "
        f"wall-clock {r['wall']:.2f}x; gradients identical {r['same_grads']}",
    )


def throughput_check(r):
    loss = 1 - r["throughput"][1] / r["throughput"][0]
    return practice.Check(
        "FINDING: spending the saved memory on more microbatches loses throughput",
        r["m_ck"] == 42 and [round(x, 3) for x in r["bubble"]] == [0.273, 0.067] and round(loss, 3) == 0.037,
        f"same memory holds M={r['m_ck']} with checkpointing; bubble {r['bubble'][0]:.1%} -> "
        f"{r['bubble'][1]:.1%}; throughput {r['throughput'][0]:.3f} -> {r['throughput'][1]:.3f} "
        f"mb/unit ({loss:.1%} lower)",
    )


def verify(result):
    return [answer_check(result), throughput_check(result)]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
