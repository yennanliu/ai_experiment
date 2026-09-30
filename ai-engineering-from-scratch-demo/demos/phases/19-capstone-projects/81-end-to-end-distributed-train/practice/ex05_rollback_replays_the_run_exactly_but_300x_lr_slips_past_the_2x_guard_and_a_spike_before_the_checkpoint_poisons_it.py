"""Exercise 5 — the rollback replays the run exactly, but no LR spike makes a NaN and one before a checkpoint poisons it.

    Add a NaN guard that rolls back to the previous checkpoint on a loss spike, and force a spike with a one-step LR multiplier to exercise the rollback.

Reading of the exercise: the loop is the lesson's (4 gloo ranks, MiniGPT,
`ZeroOptimizer`, the lesson's data slices and its step-10 sharded checkpoint
written with `save_sharded`), rewritten only as far as a rollback needs: a
step counter that can move back. The guard uses the doc's own rule, a loss
more than 2x the previous step's, or a non-finite one. It checks the loss
averaged over the ranks, so every rank takes the same decision, before
backward. On a trip every rank reloads its shard with `load_sharded` and
resumes from step 10. The spike is one optimizer step with `lr` multiplied
by M. It shows in the next step's loss and is not repeated after a rollback.
M = 1000 is used for the rollback, and a sweep of M without the guard
measures what each multiplier does.

**ANSWER: a 1000x LR step at step 12 trips the guard at step 13, and the
rollback replays the clean run exactly.** The global loss jumps from 4.1843
to 70.4817 (16.8x). The guard reloads the step-10 checkpoint once, and
steps 10-19 then match the clean run bit for bit: final loss 4.2111 and
parameter norm 54.51317.

**FINDING: the guard is a spike guard; no multiplier makes a NaN.** Adam
normalises the step and LayerNorm the activations, so the loss stays finite
up to M = 1e6 (8.3e7). The 2x rule also needs a large M. 100x raises the
loss 1.09x and 300x 1.37x, and neither trips it, yet unguarded 300x still
ends at loss 5.38 instead of 4.21. 500x (3.5x) is the smallest multiplier
tested that trips it. The doc says "the guard is unused but the hook stays",
but `main.py` has no guard or hook at all.

**FINDING: a spike in the step just before the checkpoint poisons it.** With
1000x at step 9, the step-10 checkpoint is written from the spiked weights
before any loss can show the spike. Every rollback reloads loss 76.933 and
trips again. Without the cap of 3 retries used here it would loop forever.
Rolling back safely needs an older checkpoint, or a checkpoint that is only
trusted after the next step's loss has been checked.

Structure: `launch()` starts this file as 4 gloo subprocess ranks with a
150 s timeout and kills stragglers; one process group runs all 11
configurations; checkpoints go to a temporary directory. Expected output:
three PASS checks.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from harness import parity, practice

try:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "81-end-to-end-distributed-train"
WS, MAX_ROLLBACKS, MULTS = 4, 3, (1, 10, 100, 200, 300, 500, 1000, 1e6)
# name -> (step whose update gets lr * multiplier, multiplier, guard on); sweep_1 is the clean run
RUNS = {"spike_12_guarded": (12, 1000, True), "spike_9_guarded": (9, 1000, True), **{f"sweep_{m:g}": (12, m, False) for m in MULTS}}


def launch(ws, *args, timeout=150):
    """Run this file as `ws` gloo ranks in subprocesses; return each rank's JSON, kill stragglers."""
    argv, pipe = [sys.executable, __file__, "--rank", str(ws), *map(str, args)], subprocess.PIPE
    procs = [subprocess.Popen(argv, env=os.environ | {"RANK": str(r)}, stdout=pipe, stderr=pipe, text=True) for r in range(ws)]
    try:
        outs = [p.communicate(timeout=timeout) for p in procs]
    finally:
        for p in procs:
            p.kill()
    if any(p.returncode for p in procs):
        raise RuntimeError(f"a rank failed: {[err[-400:] for _, err in outs]}")
    return [json.loads(out.splitlines()[-1]) for out, _ in outs]


def save(ref, model, optim, rank, ws, ckpt):
    """The lesson's step-10 checkpoint path: serialize, gather to rank 0, save_sharded."""
    state = {"model_state": {k: v.clone() for k, v in model.state_dict().items()},
             "optim_state": optim.state_dict(), "rank": rank}
    payloads = ref._gather_payloads_to_rank0(ref._serialize(state), ws)
    if rank == 0:
        ref.save_sharded([ref._deserialize(p) for p in payloads], ckpt, step=ref.CHECKPOINT_STEP)
    dist.barrier()


def setup(ref, rank, ws):
    """The lesson's init: seeded MiniGPT, broadcast from rank 0, ZeroOptimizer, this rank's data slices."""
    torch.manual_seed(ref.SEED)
    model = ref.MiniGPT()
    [dist.broadcast(p.data, src=0) for p in model.parameters()]
    corpus = ref.make_corpus(ref.SEED + 7, ws * ref.BATCH * (ref.SEQ_LEN + 1) * ref.STEPS)
    data = corpus.reshape(ref.STEPS, ws, ref.BATCH, ref.SEQ_LEN + 1)[:, rank]
    return model, ref.ZeroOptimizer(model, world_size=ws, rank=rank, lr=ref.LR), data


def run(ref, rank, ws, ckpt, spike, mult, guard):
    """The lesson's loop, plus the guard: a >2x or non-finite global loss reloads the step-10 checkpoint."""
    model, optim, data = setup(ref, rank, ws)
    step, seen, trace, rollbacks = 0, {}, [], []
    while step < ref.STEPS and len(rollbacks) <= MAX_ROLLBACKS:
        optim.zero_grad()
        loss = F.cross_entropy(model(data[step][:, :-1]).reshape(-1, ref.VOCAB), data[step][:, 1:].reshape(-1))
        total = loss.detach().clone()
        dist.all_reduce(total)
        g = total.item() / ws
        trace.append([step, g])
        if guard and (not math.isfinite(g) or g > 2 * seen.get(step - 1, math.inf)):
            state = ref.load_sharded(ckpt, expected_world_size=ws)[rank]
            model.load_state_dict(state["model_state"])
            optim.load_state_dict(state["optim_state"])
            rollbacks, step, spike = [*rollbacks, step], ref.CHECKPOINT_STEP, None  # the spike was one step
            continue
        loss.backward()
        optim.lr = ref.LR * (mult if step == spike else 1)
        optim.step()
        optim.lr, seen[step], step = ref.LR, g, step + 1
        if step == ref.CHECKPOINT_STEP:
            save(ref, model, optim, rank, ws, ckpt)
    norm = ref.gather_flat_params(model).double().norm().item()
    return {"trace": trace, "rollbacks": rollbacks, "done": step == ref.STEPS, "norm": norm}


def worker(rank, ws, init_file, ckpt_root):
    ref = parity.load_reference(PHASE, LESSON, "main")
    rank, ws = int(rank), int(ws)
    ref.init_distributed(rank, ws, init_file, ref._loopback_iface())
    out = {name: run(ref, rank, ws, str(Path(ckpt_root) / name), *cfg) for name, cfg in RUNS.items()}
    dist.destroy_process_group()
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory(prefix="ex05_guard_") as tmp:
        runs, source = launch(WS, str(Path(tmp) / "rdv"), tmp)[0], Path(ref.__file__).read_text(encoding="utf-8")
    at = {f"{m:g}": dict(map(tuple, runs[f"sweep_{m:g}"]["trace"])) for m in MULTS}
    return {
        "clean": runs["sweep_1"], "guarded": runs["spike_12_guarded"], "poisoned": runs["spike_9_guarded"],
        "jump": {m: t[13] / t[12] for m, t in at.items()}, "final": {m: t[19] for m, t in at.items()},
        "finite": all(math.isfinite(v) for t in at.values() for v in t.values()),
        "lesson_guard": [w for w in ("isnan", "isfinite", "rollback", "spike") if w in source],
    }


def rolled_back_clean(c, g):
    tail = [t for t in g["trace"] if t[0] >= 10]
    return g["rollbacks"] == [13] and g["done"] and g["norm"] == c["norm"] and tail[4:] == c["trace"][10:]


def missed_and_looped(r, j, p):
    missed = r["finite"] and j["300"] < 2 < j["500"] and r["final"]["300"] > 1.2 * r["final"]["1"]
    looped = p["rollbacks"] == [10] * (MAX_ROLLBACKS + 1) and len({t[1] for t in p["trace"][10:]}) == 1
    return missed and not r["lesson_guard"], looped and not p["done"]


def verify(result):
    r, j, g, p = result, result["jump"], result["guarded"], result["poisoned"]
    missed, looped = missed_and_looped(r, j, p)
    return [
        practice.Check(
            "ANSWER: a 1000x LR step at step 12 trips the guard at step 13, and the rollback replays the clean run exactly",
            rolled_back_clean(r["clean"], g) and j["1000"] > 10,
            f"loss {g['trace'][12][1]:.4f} -> {g['trace'][13][1]:.4f} ({j['1000']:.1f}x), rolled back at {g['rollbacks']}; "
            f"final loss {g['trace'][-1][1]:.4f}, norm {g['norm']:.6f}, clean {r['clean']['norm']:.6f}",
        ),
        practice.Check(
            "FINDING: no LR multiplier up to 1e6 makes a NaN, and the 2x spike test misses 300x",
            missed,
            f"loss jump by multiplier {({k: round(v, 2) for k, v in j.items()})}; unguarded final loss at 300x "
            f"{r['final']['300']:.2f} vs {r['final']['1']:.2f}; guard words in main.py: {r['lesson_guard']}",
        ),
        practice.Check(
            "FINDING: a spike in the step before the checkpoint poisons it, and the guard would roll back forever",
            looped,
            f"1000x at step 9: loss {p['trace'][10][1]:.3f} at all {len(p['rollbacks'])} rollbacks to the step-10 checkpoint",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        raise SystemExit(print(json.dumps(worker(os.environ["RANK"], *sys.argv[2:]))))
    raise SystemExit(practice.selfcheck(globals()))
