"""Exercise 4 — the MLP holds 64% of block FLOPs but only 51-59% of wall time.

    Profile a forward pass at batch sizes 1, 8, 64 with `torch.profiler`. The MLP layer dominates wall time, not attention.

Reading of the exercise: the model is the lesson's full ViT-Base
`VisionEncoder` (seed 0, eval mode, CPU, no grad) on random 224x224 images.
Each block's `attn` and `ffn` modules, and the `qkv` and `out` linears
inside `attn`, are wrapped in `record_function` labels. One warm-up forward
runs, then one profiled forward with `with_flops=True`, at batch 1, 8 and
64. Wall time per label is the profiler's `cpu_time_total`. "Attention
core" is attention time minus its two projections: the score matmul,
softmax, the value matmul and the reshapes. FLOPs are the profiler's own
per-op counts, attributed to the label that encloses each op.

**ANSWER: the MLP is the larger half of each block, but it does not
dominate.** The profiler counts 1.859 GFLOP of MLP work per image per block
against 1.049 for attention (projections, score and value matmuls, the
scale): 63.9% of block FLOPs, the same at every batch size. The MLP's share
of block wall time is smaller: 0.51-0.59 over batch 1, 8 and 64 in two runs
on the M-series CPU this was written on, falling as the batch grows.
Timings vary between runs and with machine load, so the check only asks for
a share between 0.45 and 0.75, not for the MLP to win. The FLOPs grow
exactly 8x and 64x with the batch; wall time grew less (5.5x and 46x in one
run), so batching helps.

**FINDING: the attention core costs far more time than its FLOPs.** The
score and value matmuls are 4.1% of block FLOPs (0.119 GFLOP per image per
block), but the attention core takes 17-26% of block wall time. They are
small batched matmuls plus softmax and memory copies, so they run far below
the throughput of the large `addmm` calls in the MLP and projections. At
197 tokens that core is what keeps attention close to the MLP.

Structure: `tag()` wraps a module's forward in a profiler label;
`profile_batch()` profiles one forward and totals time and FLOPs per label.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "59-vit-transformer"
BATCHES, LABELS = (1, 8, 64), ("attn", "mlp", "attn.qkv", "attn.out")


def tag(module, label):
    forward = module.forward

    def wrapped(*args, **kwargs):
        with record_function(label):
            return forward(*args, **kwargs)

    module.forward = wrapped


def owner(event):
    """The innermost label that encloses a profiler event, or None."""
    while event is not None and event.name not in LABELS:
        event = event.cpu_parent
    return event.name if event is not None else None


def profile_batch(enc, batch):
    img = torch.rand(batch, 3, 224, 224, generator=torch.Generator().manual_seed(batch))
    with torch.no_grad():
        enc(img)
        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            enc(img)
    times = {e.key: e.cpu_time_total / 1e3 for e in prof.key_averages() if e.key in LABELS}
    flops = dict.fromkeys(LABELS, 0)
    for event in prof.events():
        if event.flops and owner(event):
            flops[owner(event)] += event.flops
    gf = {k: v / (12 * batch * 1e9) for k, v in flops.items()}
    mlp, attn = gf["mlp"], gf["attn"] + gf["attn.qkv"] + gf["attn.out"]
    block_ms = times["mlp"] + times["attn"]
    core_ms = times["attn"] - times["attn.qkv"] - times["attn.out"]
    return {"mlp": mlp, "attn": attn, "fl_share": mlp / (mlp + attn), "t_share": times["mlp"] / block_ms,
            "core_fl": gf["attn"] / (mlp + attn), "core_t": core_ms / block_ms, "block_ms": block_ms,
            "flops": sum(flops.values()),
            "exact": all(abs(x - want) < 1e-3 for x, want in (
                (mlp, 1.859), (attn, 1.049), (mlp / (mlp + attn), 0.639), (gf["attn"] / (mlp + attn), 0.041))),
            "band": 0.45 < times["mlp"] / block_ms < 0.75,
            "slow": core_ms / block_ms > 2 * gf["attn"] / (mlp + attn)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.manual_seed(0)
    enc = ref.VisionEncoder(ref.ViTConfig()).eval()
    for block in enc.vit.blocks:
        tag(block.attn, "attn"), tag(block.ffn, "mlp")
        tag(block.attn.qkv, "attn.qkv"), tag(block.attn.out, "attn.out")
    rows = [profile_batch(enc, b) for b in BATCHES]
    col = lambda key: [r[key] for r in rows]  # noqa: E731
    return {k: col(k) for k in rows[0]} | {
        "flops_exact": all(col("exact")), "t_band": all(col("band")), "core_slow": all(col("slow")),
        "scale": [r["flops"] / rows[0]["flops"] for r in rows]}


def verify(result):
    r = result
    fmt = ", ".join
    return [
        practice.Check(
            "ANSWER: the MLP is the larger half of each block, but it does not dominate",
            all([r["flops_exact"], r["t_band"], r["scale"] == [1.0, 8.0, 64.0]]),
            f"GFLOP per image per block MLP {r['mlp'][0]:.3f} vs attention {r['attn'][0]:.3f} "
            f"({r['fl_share'][0]:.1%}); MLP share of block time at batch 1/8/64: "
            f"{fmt(f'{x:.2f}' for x in r['t_share'])}; block ms {fmt(f'{x:.0f}' for x in r['block_ms'])}",
        ),
        practice.Check(
            "FINDING: the attention core costs far more time than its FLOPs",
            r["core_slow"],
            f"core share of block FLOPs {r['core_fl'][0]:.1%}, of block time at batch 1/8/64: "
            f"{fmt(f'{x:.2f}' for x in r['core_t'])}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
