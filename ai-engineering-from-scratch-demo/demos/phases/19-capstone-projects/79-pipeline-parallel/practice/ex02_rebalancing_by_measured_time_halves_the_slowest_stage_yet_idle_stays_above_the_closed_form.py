"""Exercise 2 -- rebalancing by measured time halves the slowest stage, yet idle stays above the closed form.

    Profile real per-stage time on a deeper model and rebalance stages by measured wall-clock.

Reading of the exercise: the "deeper model" is 16 layers built from the
lesson's own `StageMLP`: an `nn.Embedding(4000, 64)`, eleven narrow
`StageMLP(64, 32, 64)` blocks, then four wide `StageMLP(64, 2048, 64)`
blocks, batch 64, CPU, one thread. Each layer's forward+backward is timed
(fastest of 41 runs, to shed machine noise), and the 16 layers are cut into
4 contiguous stages three ways: 4 layers each (the default the lesson
implies), balanced parameter count, and balanced measured time (best of all
455 cuts). Each partition is then rebuilt as 4 real stage modules and timed
again. Idle time is the GPipe makespan for M=8 with those stage times,
sum(t) + (M-1)max(t).

**ANSWER: rebalancing by measured time cuts the slowest stage 2.0x against
4-layers-per-stage**, and timing the rebuilt stages again gives about 2.1x.
The 4-per-stage cut puts all four wide blocks on the last stage, so the
pipeline idles about 64% of the step at M=8. The time-balanced cut idles
about 39%: still above the lesson's closed form of 27.3%, because one wide
block is a quarter of the model and cannot be split. Timings vary with load,
so only 1.4x (1.3x re-timed) is asserted; the numbers are in the check
detail.

**FINDING: an embedding's cost grows with its vocabulary, which a FLOP count
scores as zero.** On CPU the backward writes a dense gradient for the whole
table. An `Embedding(16000, 64)` takes 5-6x as long as an `Embedding(4000,
64)` on the same 64 tokens (at least 2x is asserted). The doc's rule,
"equalise FLOPs per stage, not weights", would miss that; profiling does
not. Balancing parameters is about 1.2x worse than balancing time here, as
the doc predicts.

Expected output: two PASS checks.
"""

from __future__ import annotations

import itertools
import time

from harness import parity, practice

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "79-pipeline-parallel"
VOCAB, DIM, BATCH, REPS, STAGES, M = 4000, 64, 64, 41, 4, 8
HIDDEN = [32] * 11 + [2048] * 4


def build(ref):
    torch.manual_seed(0)
    return [nn.Embedding(VOCAB, DIM)] + [ref.StageMLP(DIM, h, DIM) for h in HIDDEN]


def fastest(module, first, vocab=VOCAB):
    """Fastest of REPS forward+backward runs of one module on one microbatch."""
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, vocab, (BATCH,), generator=g) if first else torch.randn(BATCH, DIM, generator=g)
    best = float("inf")
    for _ in range(REPS):
        t0 = time.perf_counter()
        module(x).sum().backward()
        best = min(best, time.perf_counter() - t0)
    return best


def best_cut(cost):
    cuts = ((0, *c, len(cost)) for c in itertools.combinations(range(1, len(cost)), STAGES - 1))
    return min(cuts, key=lambda c: max(stage_sums(c, cost)))


def stage_sums(cut, cost):
    return [sum(cost[a:b]) for a, b in zip(cut, cut[1:])]


def idle(stage_times):
    span = sum(stage_times) + (M - 1) * max(stage_times)
    return 1 - M * sum(stage_times) / (STAGES * span)


def measure(ref):
    """Time every layer, cut three ways, and re-time each partition's rebuilt stages (one thread)."""
    layers = build(ref)
    times = [fastest(layer, i == 0) for i, layer in enumerate(layers)]
    params = [sum(p.numel() for p in layer.parameters()) for layer in layers]
    cuts = {"count": (0, 4, 8, 12, 16), "params": best_cut(params), "time": best_cut(times)}
    staged = {k: [fastest(nn.Sequential(*layers[a:b]), a == 0) for a, b in zip(c, c[1:])]
              for k, c in cuts.items()}
    vocab = {v: fastest(nn.Embedding(v, DIM), True, v) for v in (VOCAB, 4 * VOCAB)}
    return times, cuts, staged, vocab


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        times, cuts, staged, vocab = measure(ref)
    finally:
        torch.set_num_threads(threads)
    return {
        "cuts": cuts, "times_us": [round(t * 1e6) for t in times],
        "profiled": {k: max(stage_sums(c, times)) for k, c in cuts.items()},
        "staged": staged, "idle": {k: idle(v) for k, v in staged.items()},
        "closed": ref.bubble_fraction(STAGES, M),
        "vocab_ratio": vocab[4 * VOCAB] / vocab[VOCAB],
    }


def verify(result):
    p, s = result["profiled"], {k: max(v) for k, v in result["staged"].items()}
    t = result["times_us"]
    narrow = sorted(t[1:12])[5]
    return [
        practice.Check(
            "ANSWER: rebalancing by measured time cuts the slowest stage about 2x against 4-layers-per-stage",
            p["count"] >= 1.4 * p["time"] and s["count"] >= 1.3 * s["time"]
            and result["idle"]["count"] > 2 * result["closed"] and result["idle"]["time"] > result["closed"],
            f"cuts {result['cuts']}; profiled slowest stage count/time {p['count'] / p['time']:.2f}x, "
            f"re-timed {s['count'] / s['time']:.2f}x; M=8 idle count {result['idle']['count']:.1%}, "
            f"time {result['idle']['time']:.1%}, closed form {result['closed']:.1%}",
        ),
        practice.Check(
            "FINDING: an embedding's cost grows with its vocabulary, which a FLOP count calls free",
            result["vocab_ratio"] >= 2 and p["params"] > p["time"],
            f"Embedding({4 * VOCAB}) fwd+bwd {result['vocab_ratio']:.1f}x Embedding({VOCAB}) on the same "
            f"{BATCH} tokens (identical lookups); in the model the embedding costs {t[0] / narrow:.1f}x a "
            f"narrow block; parameter-balanced slowest stage {p['params'] / p['time']:.2f}x the time-balanced one",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
