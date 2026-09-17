"""Exercise 5 — max_memory_allocated() returns 0 on a CPU box, and step time is not the FLOP model.

    Benchmark a real PyTorch transformer with and without
    `torch.utils.checkpoint`. Measure memory (via
    `torch.cuda.max_memory_allocated`) and step time.

Reading of the exercise: a real 12-block PyTorch transformer is built and run
both ways with `torch.utils.checkpoint`, at four segment sizes. The prescribed
instrument is called first, and then `torch.autograd.graph.saved_tensors_hooks`
is used instead, because it counts the same bytes on any device.

**ANSWER: `torch.cuda.max_memory_allocated()` returns 0 and does not raise.**
Without CUDA it reports 0 bytes before the step and 0 after, for both arms, so
the exercise's measurement says checkpointing saved nothing -- silently, with no
error to notice.

**FINDING: the bytes autograd actually saves, counted with
`saved_tensors_hooks`:**

    no checkpoint     152.166 MB    121 tensors
    checkpoint k=1     13.631 MB     13 tensors     11.2x
    checkpoint k=2      7.340 MB      7 tensors     20.7x
    checkpoint k=4      4.194 MB      4 tensors     36.3x
    checkpoint k=12     2.097 MB      2 tensors     72.6x

This is the measurement the exercise wants, it is exact rather than a
high-water mark, and it needs no GPU: the pack hook fires on every tensor
autograd retains, so summing `numel * element_size` is the saved-activation
volume by construction.

**MECHANISM: PyTorch's recompute total is one extra forward at every k.** Each
segment is recomputed exactly once, so the FLOP overhead is `fwd / (fwd + bwd)`
= **33.3%** regardless of segment size -- flat, not the `(k-1)/k` curve
`checkpoint_cost` plots. What k changes is only the memory.

**FINDING: measured step time rises with k, which is the wrong direction for
every model in the lesson.** Every arm is slower than no checkpointing, and the
penalty grows from k=1 to k=12 -- +17% to +28% when the run has the machine to
itself, +30% to +104% when it does not -- while the number of `checkpoint` calls
*falls* 12x across that same range. So it is not call overhead, and torch's
recompute FLOPs are flat, so it is not FLOPs either. With `use_reentrant=False`
a segment's recomputed graph is materialised in full and held until that
segment's backward finishes, so a larger k trades saved bytes for a larger
transient. Both numbers the exercise asks for are unusable as stated: one is
identically zero, and the other moves for a reason neither cost model contains.

Structure: `measure` installs the pack hook around one step and then times the
same step; `Block` is the transformer block both arms share.
"""

from __future__ import annotations

import math
import time

import torch

from harness import practice

LAYERS, HIDDEN, INNER, BATCH, SEQ = 12, 256, 1024, 8, 128
SEGMENTS = (1, 2, 4, 12)
REPEATS = 5


class Block(torch.nn.Module):
    """One pre-norm transformer-style block: LayerNorm, GELU MLP, residual."""

    def __init__(self):
        super().__init__()
        self.norm, self.up = torch.nn.LayerNorm(HIDDEN), torch.nn.Linear(HIDDEN, INNER)
        self.down = torch.nn.Linear(INNER, HIDDEN)

    def forward(self, x):
        return x + self.down(torch.nn.functional.gelu(self.up(self.norm(x))))


def step(blocks, x, segment):
    """One forward and backward, checkpointed at `segment` blocks per segment."""
    for param in blocks.parameters():
        param.grad = None
    h, width = x, segment or LAYERS
    for start in range(0, LAYERS, width):
        piece = torch.nn.Sequential(*blocks[start:start + width])
        h = (torch.utils.checkpoint.checkpoint(piece, h, use_reentrant=False)
             if segment else piece(h))
    h.square().mean().backward()
    return [param.grad.clone() for param in blocks.parameters()]


def measure(blocks, x, segment):
    """Bytes autograd retains for the backward pass, and the fastest step time."""
    sizes, fastest = [], math.inf
    with torch.autograd.graph.saved_tensors_hooks(
            lambda t: sizes.append(t.numel() * t.element_size()) or t, lambda t: t):
        step(blocks, x, segment)
    for _ in range(REPEATS):
        start = time.perf_counter()
        grads = step(blocks, x, segment)
        fastest = min(fastest, time.perf_counter() - start)
    return {"bytes": sum(sizes), "tensors": len(sizes), "time": fastest, "grads": grads}


def cuda_probe(blocks, x):
    """The instrument the exercise names, called exactly as it prescribes."""
    before = torch.cuda.max_memory_allocated()
    step(blocks, x, None)
    plain = torch.cuda.max_memory_allocated()
    step(blocks, x, 1)
    return {"before": before, "plain": plain, "available": torch.cuda.is_available(),
            "checkpointed": torch.cuda.max_memory_allocated()}


def solve():
    torch.manual_seed(0)
    blocks = torch.nn.ModuleList([Block() for _ in range(LAYERS)])
    x = torch.randn(BATCH, SEQ, HIDDEN)
    probe = cuda_probe(blocks, x)
    baseline = measure(blocks, x, None)
    arms = {k: measure(blocks, x, k) for k in SEGMENTS}
    return {
        "probe": probe,
        "baseline": baseline,
        "arms": arms,
        "times": {k: row["time"] / baseline["time"] - 1 for k, row in arms.items()},
        "grad_diff": {k: max(float((a - b).abs().max())
                             for a, b in zip(baseline["grads"], row["grads"]))
                      for k, row in arms.items()},
        "calls": {k: math.ceil(LAYERS / k) for k in SEGMENTS},
        "flat_overhead": 1 / 3,
    }


def table(arms, baseline):
    return ", ".join(f"k={k} {row['bytes'] / 1e6:.3f} MB in {row['tensors']} tensors "
                     f"({baseline / row['bytes']:.1f}x)" for k, row in arms.items())


def listed(values, fmt):
    return ", ".join(f"k={k} {format(value, fmt)}" for k, value in values.items())


def silent_zero(probe):
    """The prescribed instrument read three times, all zero, with no exception."""
    return not probe["available"] and probe["plain"] == 0 and probe["checkpointed"] == 0


def shrinks(arms, baseline):
    largest = arms[max(SEGMENTS)]
    return (arms[1]["bytes"] < baseline["bytes"] / 10
            and largest["bytes"] < arms[1]["bytes"] / 5 and largest["tensors"] == 2)


def verify(result):
    probe, arms, baseline = result["probe"], result["arms"], result["baseline"]
    times, flat = result["times"], result["flat_overhead"]
    return [
        practice.Check(
            "ANSWER: torch.cuda.max_memory_allocated() returns 0 and does not raise",
            silent_zero(probe),
            f"cuda.is_available() is {probe['available']} and the counter reads "
            f"{probe['before']} before the step, {probe['plain']} after the full one and "
            f"{probe['checkpointed']} after the checkpointed one -- so the exercise's "
            "measurement reports that checkpointing saved nothing, with no error to notice",
        ),
        practice.Check(
            "FINDING: saved_tensors_hooks counts it exactly, on any device",
            shrinks(arms, baseline),
            f"autograd retains {baseline['bytes'] / 1e6:.3f} MB in {baseline['tensors']} tensors "
            "with no checkpointing, against " + table(arms, baseline["bytes"])
            + ". The pack hook fires on every tensor autograd keeps, so summing "
            "numel * element_size is the saved volume by construction -- exact rather than a "
            "high-water mark, and needing no GPU",
        ),
        practice.Check(
            "MECHANISM: PyTorch's recompute is one extra forward at every k, so 33.3% flat",
            set(result["grad_diff"].values()) == {0.0},
            "each segment is recomputed exactly once whatever its size, so the overhead is "
            f"fwd / (fwd + bwd) = {flat:.1%} regardless of k -- flat, not the (k-1)/k curve "
            "checkpoint_cost plots. What k changes is only the memory, and the gradients are "
            "identical at every k: " + listed(result["grad_diff"], ""),
        ),
        practice.Check(
            "FINDING: step time rises with k while the call count falls 12x",
            (min(times.values()) > 0.05 and times[max(SEGMENTS)] > times[1]
             and result["calls"][1] == LAYERS * result["calls"][max(SEGMENTS)]),
            "the arms measure " + listed(times, "+.0%") + f" against a flat {flat:.1%} of real "
            "recompute, while the checkpoint calls fall " + listed(result["calls"], "")
            + ". So it is neither call overhead nor FLOPs: with use_reentrant=False a segment's "
            "recomputed graph is materialised in full and held until that segment's backward "
            "finishes, so a larger k trades saved bytes for a larger transient",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
