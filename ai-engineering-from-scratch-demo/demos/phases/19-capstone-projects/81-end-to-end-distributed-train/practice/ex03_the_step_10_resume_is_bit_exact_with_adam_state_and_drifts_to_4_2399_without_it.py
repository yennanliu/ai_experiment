"""Exercise 3 — resuming from the step-10 checkpoint reproduces the run bit for bit, but only with Adam's state.

    Add a resume-from-step-10 path that actually continues training to step 20 and produces the same final loss as the original run.

Reading of the exercise: "the original run" is the lesson's own
`_train_worker`, unmodified, as 4 gloo ranks for 20 steps; it writes the
step-10 sharded checkpoint. The resume path is a fresh set of 4 ranks that
loads that checkpoint with the lesson's `load_sharded`, restores each rank's
model and `ZeroOptimizer` state, and runs steps 10-19 on the same data
slices. "The same final loss" is checked as exact equality of every resumed
loss and of the final parameter norm. As a control, a second resume
restores only the weights and starts Adam afresh, a common shortcut.

**ANSWER: the resumed run reproduces steps 10-19 bit for bit.** The final
rank-0 loss is 4.222419 in both runs, all 10 resumed losses are identical,
and so is the final parameter norm (54.513171 on every rank). The resumed
Adam step count is 20, as in the original.

**FINDING: restoring the weights without the optimizer state does not.**
Step 10's loss still matches (same weights, same batch), but from step 11
on the fresh Adam moments and bias correction take different steps. The
final loss is 4.2399 instead of 4.2224.

**FINDING: the lesson never resumes, and its checkpoint is 55% redundant
copies of the model.** `verify_resume` only compares saved bytes with an
in-memory snapshot; no code continues training. Each of the 4 shard files
carries the full 119,296-byte `model_state`, identical on every rank. The
concatenated ZeRO master shards already hold exactly those values, so 4
model copies are 55% of the checkpoint's shard bytes. Loading at world size 2 is
refused: "world_size mismatch: manifest=4, expected=2".

Structure: `launch()` starts this file as 4 gloo ranks per run (original,
full resume, weights-only resume), as subprocesses with a 150 s timeout, and
kills stragglers; checkpoints go to a temporary directory. Expected output:
three PASS checks.
"""

from __future__ import annotations

import inspect
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


def resume(ref, rank, ws, init_file, ckpt_dir, mode):
    """Reload the lesson's step-10 checkpoint and run steps 10..19 exactly as `_train_worker` does."""
    ref.init_distributed(rank, ws, init_file, ref._loopback_iface())
    torch.manual_seed(ref.SEED)
    model, state = ref.MiniGPT(), ref.load_sharded(ckpt_dir, expected_world_size=ws)[rank]
    model.load_state_dict(state["model_state"])
    optim = ref.ZeroOptimizer(model, world_size=ws, rank=rank, lr=ref.LR)  # master copy rebuilt from the weights
    if mode == "full":  # otherwise the common shortcut: weights only, fresh Adam moments
        optim.load_state_dict(state["optim_state"])
    corpus = ref.make_corpus(ref.SEED + 7, ws * ref.BATCH * (ref.SEQ_LEN + 1) * ref.STEPS)
    losses = []
    for block in corpus.reshape(ref.STEPS, ws, ref.BATCH, ref.SEQ_LEN + 1)[ref.CHECKPOINT_STEP:, rank]:
        optim.zero_grad()
        loss = F.cross_entropy(model(block[:, :-1]).reshape(-1, ref.VOCAB), block[:, 1:].reshape(-1))
        loss.backward()
        optim.step()
        losses.append(loss.item())
    norm = sum(p.detach().pow(2).sum().item() for p in model.parameters()) ** 0.5
    return {"rank": rank, "losses": losses, "norm": norm, "step_count": optim.step_count}


def worker(rank, ws, mode, init_file, ckpt_dir):
    ref = parity.load_reference(PHASE, LESSON, "main")
    rank, ws = int(rank), int(ws)
    if mode == "orig":  # the lesson's own worker, unmodified; it writes the step-10 checkpoint
        ref._train_worker(rank, ws, init_file, ref._loopback_iface(), ckpt_dir, ref.STEPS, PRINT_QUEUE)
    out = resume(ref, rank, ws, init_file, ckpt_dir, mode)
    torch.distributed.destroy_process_group()
    return out


def inspect_checkpoint(ref, ckpt):
    """What the lesson's step-10 checkpoint holds, and whether it loads at another world size."""
    try:
        ref.load_sharded(ckpt, expected_world_size=2)
        refused = ""
    except RuntimeError as exc:
        refused = str(exc)
    states = ref.load_sharded(ckpt, expected_world_size=4)
    loaded, master = [s["model_state"] for s in states], torch.cat([s["optim_state"]["master_shard"] for s in states])
    flat = torch.cat([v.flatten() for v in loaded[0].values()])
    return {
        "refused": refused, "files": {p.name: p.stat().st_size for p in sorted(Path(ckpt).iterdir())},
        "model_bytes": flat.numel() * flat.element_size(), "master_is_model": torch.equal(master[:flat.numel()], flat),
        "same_model": all(torch.equal(loaded[0][k], s[k]) for s in loaded for k in s),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory(prefix="ex03_resume_") as tmp:
        ckpt = str(Path(tmp) / "step_0010")
        runs = {m: launch(4, m, str(Path(tmp) / f"rdv_{m}"), ckpt) for m in ("orig", "full", "weights")}
        facts = inspect_checkpoint(ref, ckpt)
    r0 = {m: next(r for r in rs if r["rank"] == 0) for m, rs in runs.items()}
    return {
        **facts, "orig_tail": r0["orig"]["losses"][10:],
        "full_tail": r0["full"]["losses"], "weights_tail": r0["weights"]["losses"],
        "norms": {m: sorted({r["norm"] for r in rs}) for m, rs in runs.items()}, "step_count": r0["full"]["step_count"],
        "lesson_resumes": any("load_state_dict(" in inspect.getsource(f) for f in (ref.main, ref.run_e2e, ref.verify_resume)),
    }


def resumes(r):
    """(full resume is bit-exact, weights-only resume matches step 10 and then drifts)."""
    w, o = r["weights_tail"], r["orig_tail"]
    same = r["full_tail"] == o and r["norms"]["full"] == r["norms"]["orig"] and len(r["norms"]["orig"]) == 1
    drift = w[0] == o[0] and all(a != b for a, b in zip(w[1:], o[1:])) and r["norms"]["weights"] != r["norms"]["orig"]
    return same and r["step_count"] == 20 and abs(o[-1] - 4.2224) < 1e-3, drift


def redundant(r):
    share = 4 * r["model_bytes"] / sum(v for k, v in r["files"].items() if k.endswith(".bin"))
    ok = r["refused"] == "world_size mismatch: manifest=4, expected=2" and r["same_model"]
    return ok and r["master_is_model"] and r["model_bytes"] == 119296 and not r["lesson_resumes"], share


def verify(result):
    r, (ok, share), (same, drift) = result, redundant(result), resumes(result)
    return [
        practice.Check(
            "ANSWER: resuming from the lesson's step-10 checkpoint reproduces steps 10-19 bit for bit",
            same,
            f"final rank-0 loss {r['full_tail'][-1]!r} resumed vs {r['orig_tail'][-1]!r} original; "
            f"final param norm {r['norms']['full']} vs {r['norms']['orig']}; Adam step_count {r['step_count']}",
        ),
        practice.Check(
            "FINDING: restoring the weights without the Adam state does not give the same final loss",
            drift and abs(r["weights_tail"][-1] - r["orig_tail"][-1]) > 1e-3,
            f"weights-only final loss {r['weights_tail'][-1]:.4f}, norm {r['norms']['weights'][0]:.6f}",
        ),
        practice.Check(
            "FINDING: the lesson never resumes, and its checkpoint is 55% redundant copies of the model",
            ok and 0.5 < share < 0.6,
            f"files {r['files']}; each has the same {r['model_bytes']}-byte model_state the master shards hold "
            f"({share:.0%}); world size 2: '{r['refused']}'; the lesson calls load_state_dict: {r['lesson_resumes']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        raise SystemExit(print(json.dumps(worker(os.environ["RANK"], *sys.argv[2:]))))
    raise SystemExit(practice.selfcheck(globals()))
