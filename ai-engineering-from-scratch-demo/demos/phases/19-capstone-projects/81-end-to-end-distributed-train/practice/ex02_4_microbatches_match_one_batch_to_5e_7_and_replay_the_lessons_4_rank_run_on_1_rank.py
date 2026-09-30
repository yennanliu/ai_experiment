"""Exercise 2 — 4 accumulated microbatches match one batch to 5e-7, and replay the lesson's 4-rank run on 1.

    Add gradient accumulation across 4 microbatches and prove the gradient equals the gradient of one big batch.

Reading of the exercise: the big batch is the lesson's step-0 data, the 16
sequences its 4 ranks each take 4 of, and the model is the lesson's seeded
MiniGPT. Accumulation runs 4 backward passes of 4 sequences into `.grad`,
each loss scaled by 1/4, and is compared with one backward pass over all 16.
"Equals" is checked in float32, as the largest element-wise gap relative to
the largest gradient element. The proof is also run end to end: one rank
with the lesson's `ZeroOptimizer` accumulates the 4 ranks' batches for 20
steps and is compared with the lesson's own unmodified 4-rank run.

**ANSWER: the accumulated gradient equals the big-batch gradient to 4.8e-7
(relative), over all 29,824 elements.** It is not bit-equal: the sums run in
a different order. Leaving out the 1/4 scaling gives exactly 4.0x the norm.

**FINDING: the lesson's 4-rank run is gradient accumulation.** One rank
accumulating the same 4 batches reproduces its 20 rank-0 losses to 4.8e-7
and its final parameter norm 54.513171 to 1e-8. reduce_scatter plus the
division by world size is the same sum-then-average.

**FINDING: the equality needs equal microbatches.** Split the 16 sequences
as 1, 3, 5 and 7 and scale each loss by 1/4, and the gradient is off by
124% of its largest element. Weighting each loss by its share of the tokens
(n/16) makes it exact again (3.8e-7). The lesson's `ZeroOptimizer` divides
by world size the same way, so it assumes equal per-rank batches too.

Structure: `grads()` runs the in-process comparisons; `launch()` starts the
lesson's own `_train_worker` as 4 gloo ranks and the accumulating run as 1,
each as subprocesses with a 150 s timeout, and kills stragglers. Expected
output: three PASS checks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from harness import parity, practice

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "81-end-to-end-distributed-train"
MICRO = 4


def launch(ws, *args, timeout=150):
    """Run this file as `ws` gloo ranks in subprocesses; return each rank's JSON, kill stragglers."""
    argv = [sys.executable, __file__, "--rank", str(ws), *map(str, args)]
    pipe = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
    procs = [subprocess.Popen(argv, env=os.environ | {"RANK": str(r)}, **pipe) for r in range(ws)]
    try:
        outs = [p.communicate(timeout=timeout) for p in procs]
    finally:
        for p in procs:
            p.kill()
    if any(p.returncode for p in procs):
        raise RuntimeError(f"a rank failed: {[err[-400:] for _, err in outs]}")
    return [json.loads(out.splitlines()[-1]) for out, _ in outs]


PRINT_QUEUE = type("PrintQueue", (), {  # the lesson's mp.Queue; `_train_worker` ends in os._exit, so print first
    "put": lambda self, item: print(json.dumps(dict(zip(("rank", "losses", "norm"), item))), flush=True),
    "close": lambda self: None, "join_thread": lambda self: None,
})()


def lm_loss(ref, model, block):
    return F.cross_entropy(model(block[:, :-1]).reshape(-1, ref.VOCAB), block[:, 1:].reshape(-1))


def sequences(ref, steps):
    """The lesson's data as [step, 16, SEQ_LEN + 1]: rank r's batch is rows 4r..4r+3 of its step."""
    corpus = ref.make_corpus(ref.SEED + 7, MICRO * ref.BATCH * (ref.SEQ_LEN + 1) * steps)
    return corpus.reshape(steps, MICRO * ref.BATCH, ref.SEQ_LEN + 1)


def accumulated(ref, model, parts, scales):
    """Backward each microbatch with its loss scaled; the grads sum in .grad. Returns microbatch 0's loss."""
    losses = []
    for part, scale in zip(parts, scales):
        loss = lm_loss(ref, model, part)
        (loss * scale).backward()
        losses.append(loss.item())
    return losses[0]


def accum_worker(ref, init_file):
    """One rank running the lesson's ZeroOptimizer, with the 4 ranks' batches as 4 microbatches."""
    ref.init_distributed(0, 1, init_file, ref._loopback_iface())
    torch.manual_seed(ref.SEED)
    model = ref.MiniGPT()
    optim = ref.ZeroOptimizer(model, world_size=1, rank=0, lr=ref.LR)
    losses = []
    for seqs in sequences(ref, ref.STEPS):
        optim.zero_grad()
        losses.append(accumulated(ref, model, seqs.split(ref.BATCH), [1 / MICRO] * MICRO))
        optim.step()
    norm = sum(p.detach().pow(2).sum().item() for p in model.parameters()) ** 0.5
    return {"rank": 0, "losses": losses, "norm": norm}


def worker(rank, ws, mode, init_file, ckpt):
    ref = parity.load_reference(PHASE, LESSON, "main")
    if mode == "lesson":  # the lesson's own 4-rank worker, unmodified; it exits the process itself
        ref._train_worker(int(rank), int(ws), init_file, ref._loopback_iface(), ckpt, ref.STEPS, PRINT_QUEUE)
    return accum_worker(ref, init_file)


def grads(ref, scales=None, sizes=(4, 4, 4, 4)):
    """Flat gradient of a fresh lesson model on the lesson's step-0 16 sequences: one batch, or split."""
    torch.manual_seed(ref.SEED)
    model = ref.MiniGPT()
    seqs = sequences(ref, 1)[0]
    accumulated(ref, model, torch.split(seqs, list(sizes)) if scales else [seqs], scales or [1])
    return ref.gather_flat_grads(model)


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    big, quarter, uneven = grads(ref), [1 / MICRO] * MICRO, (1, 3, 5, 7)
    acc, raw = grads(ref, quarter), grads(ref, [1] * MICRO)
    with tempfile.TemporaryDirectory(prefix="ex02_rdv_") as tmp:
        runs = {m: launch(ws, m, str(Path(tmp) / f"rdv_{m}"), str(Path(tmp) / "ckpt")) for m, ws in (("lesson", 4), ("accum", 1))}
    return {
        "numel": big.numel(), "rel": rel(acc, big), "bit_equal": torch.equal(acc, big),
        "unscaled_ratio": (raw.norm() / big.norm()).item(),
        "uneven_mean": rel(grads(ref, quarter, uneven), big),
        "uneven_weighted": rel(grads(ref, [n / 16 for n in uneven], uneven), big),
        "lesson": next(r for r in runs["lesson"] if r["rank"] == 0), "accum": runs["accum"][0],
    }


def replay(r):
    losses = zip(r["lesson"]["losses"], r["accum"]["losses"])
    return max(abs(a - b) for a, b in losses), abs(r["lesson"]["norm"] - r["accum"]["norm"])


def verify(result):
    r, (loss_gap, norm_gap) = result, replay(result)
    return [
        practice.Check(
            "ANSWER: the accumulated gradient equals the big-batch gradient to float32 noise",
            r["rel"] < 1e-5 and r["numel"] == 29824 and round(r["unscaled_ratio"], 4) == 4.0,
            f"relative gap {r['rel']:.2g} over {r['numel']} elements (bit-equal: {r['bit_equal']}); unscaled {r['unscaled_ratio']:.4f}x",
        ),
        practice.Check(
            "FINDING: one rank accumulating 4 microbatches replays the lesson's 4-rank run",
            loss_gap < 1e-5 and norm_gap < 1e-5 and len(r["accum"]["losses"]) == 20
            and abs(r["lesson"]["norm"] - 54.513171) < 1e-3,
            f"20 rank-0 losses agree to {loss_gap:.2g}; final norm {r['accum']['norm']:.6f} (gap {norm_gap:.2g})",
        ),
        practice.Check(
            "FINDING: with uneven microbatches, scaling each loss by 1/4 is wrong and token weighting fixes it",
            r["uneven_mean"] > 0.5 and r["uneven_weighted"] < 1e-5,
            f"sizes 1/3/5/7: 1/4 each is off by {r['uneven_mean']:.0%}, n/16 each by {r['uneven_weighted']:.2g}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        raise SystemExit(print(json.dumps(worker(os.environ["RANK"], *sys.argv[2:]))))
    raise SystemExit(practice.selfcheck(globals()))
