"""Exercise 2 — keeping K checkpoints needs K+1 on disk, and a kill mid-save leaves a .tmp that no pruner ever deletes.

    Extend the save loop to keep the last K checkpoints and prune older ones. What is the right K when the disk is small?

Reading of the exercise: the lesson's demo run (seed 11, 24 steps, 5
batches per epoch) is driven through `train_until` in blocks of 4 steps,
with the lesson's `save_checkpoint` writing `ckpt-<step>.pt` after each
block and a pruner deleting all but the newest K, for K = 1, 2, 3. Disk use
is measured at its peak -- after the new file lands and before pruning --
and a real crash is simulated by a child process that is killed with
`os._exit` inside `atomic_save`, between the temp write and `os.replace`.

**ANSWER: K = (disk / checkpoint size) - 1, and K = 1 when the disk holds
only two.** Keeping K costs K+1 checkpoints at the peak, because the new
file must land before the oldest may go: 2 / 3 / 4 files (79,422 /
119,101 / 158,716 bytes) for K = 1 / 2 / 3, each checkpoint about 39.7 KB.
K = 1 is already crash-safe, because the lesson's save is write-then-rename:
after three kills mid-save the one kept checkpoint (`ckpt-0004.pt`) still
loads and resumes to the uninterrupted losses with a max diff of 0.0. What
K = 1 cannot survive is a kept file that is silently bad (exercise 4), so
K = 2 is the choice as soon as three checkpoints fit.

**FINDING: every kill mid-save leaves a full-size orphan that pruning never
sees.** Each killed child (exit 9) leaves a `ckpt-0008.pt.<random>.tmp`;
the lesson's `finally: unlink` never runs on a kill. Three kills leave
118,845 bytes of temp files, three checkpoints' worth, more than K = 1 is
allowed, and a pruner globbing `ckpt-*.pt` matches only `ckpt-0004.pt`.
On a small disk the save loop must also sweep `*.tmp` at startup. The
lesson's code calls no `fsync` either, although its reading list cites the
fsync documentation "for the durability guarantee behind atomic rename".

Expected output: two PASS checks.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import tempfile
from pathlib import Path

import harness
from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "47-checkpoint-save-resume"
DIMS = {"batches_per_epoch": 5, "batch_size": 4, "in_dim": 16, "out_dim": 4}
KILL = """
import os, sys, torch
from pathlib import Path
from harness import parity
ref = parity.load_reference(%r, %r, "main")
ref.seed_everything(11)
m = ref.make_model(16, 24, 4); o, s = ref.make_optimizer_and_scheduler(m, lr=0.01, total_steps=24)
st = ref.train_until(m, o, s, torch.nn.CrossEntropyLoss(), ref.TrainState(0, 0, 0), stop_step=8, **%r)
os.replace = lambda *a: os._exit(9)          # the kill lands between temp write and rename
ref.save_checkpoint(m, o, s, st, Path(sys.argv[1]) / "ckpt-0008.pt")
"""


def fresh(ref):
    ref.seed_everything(11)
    model = ref.make_model(16, 24, 4)
    return (model, *ref.make_optimizer_and_scheduler(model, lr=0.01, total_steps=24))


def keep_last(ref, folder, k, every=4, total=24):
    """The save loop with pruning; returns the peak (files, bytes) and what survives."""
    model, opt, sched = fresh(ref)
    state, peak = ref.TrainState(0, 0, 0), (0, 0)
    while state.step < total:
        ref.train_until(model, opt, sched, torch.nn.CrossEntropyLoss(), state, stop_step=state.step + every, **DIMS)
        ref.save_checkpoint(model, opt, sched, state, folder / f"ckpt-{state.step:04d}.pt")
        files = sorted(folder.glob("ckpt-*.pt"))
        peak = max(peak, (len(files), sum(f.stat().st_size for f in folder.iterdir())))
        for old in files[:-k]:
            old.unlink()
    return {"peak": peak, "kept": [f.name for f in sorted(folder.glob("ckpt-*.pt"))], "losses": state.losses}


def resume_diff(ref, path, full):
    model, opt, sched = fresh(ref)
    state = ref.load_checkpoint(path, model, opt, sched)
    start = state.step
    ref.train_until(model, opt, sched, torch.nn.CrossEntropyLoss(), state, stop_step=24, **DIMS)
    return max(abs(a - b) for a, b in zip(full[start:], state.losses[start:], strict=True))


def killed_saves(ref, folder, kills):
    code = KILL % (PHASE, LESSON, DIMS)
    root = str(Path(harness.__file__).resolve().parents[1])
    codes = [subprocess.run([sys.executable, "-c", code, str(folder)], cwd=root, check=False).returncode
             for _ in range(kills)]
    return codes, sorted(p.name.split(".")[-1] for p in folder.iterdir())


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    out = {}
    with tempfile.TemporaryDirectory(prefix="ex02-") as tmp:
        for k in (1, 2, 3):
            (Path(tmp) / f"k{k}").mkdir()
            out[k] = keep_last(ref, Path(tmp) / f"k{k}", k)
        single = (Path(tmp) / "k1" / "ckpt-0024.pt").stat().st_size
        crash = Path(tmp) / "crash"
        crash.mkdir()
        keep_last(ref, crash, 1, total=4)
        codes, suffixes = killed_saves(ref, crash, 3)
        orphans = sum(p.stat().st_size for p in crash.glob("*.tmp"))
        survivor = resume_diff(ref, crash / "ckpt-0004.pt", out[1]["losses"])
        pruned_by_glob = [p.name for p in crash.glob("ckpt-*.pt")]
    return {"runs": {k: {"peak": v["peak"], "kept": v["kept"]} for k, v in out.items()}, "single": single,
            "codes": codes, "suffixes": suffixes, "orphan_bytes": orphans, "survivor_diff": survivor,
            "visible_to_pruner": pruned_by_glob, "fsync": "fsync" in inspect.getsource(ref)}


def verify(result):
    runs, size = result["runs"], result["single"]
    peaks = [runs[1]["peak"], runs[2]["peak"], runs[3]["peak"]]
    return [
        practice.Check(
            "ANSWER: keeping K needs K+1 on disk, and K=1 survives a kill mid-save",
            [files for files, _ in peaks] == [2, 3, 4]
            and max(abs(b - n * size) for n, (_, b) in enumerate(peaks, 2)) < 0.02 * size
            and (runs[2]["kept"], result["survivor_diff"]) == (["ckpt-0020.pt", "ckpt-0024.pt"], 0.0),
            f"peak files/bytes {peaks} for K=1,2,3 with a {size} B checkpoint; "
            f"after the kills ckpt-0004.pt resumes with max loss diff {result['survivor_diff']}",
        ),
        practice.Check(
            "FINDING: every kill mid-save leaves a full-size .tmp that a ckpt-*.pt pruner never sees",
            (result["codes"], result["suffixes"], result["visible_to_pruner"], result["fsync"])
            == ([9, 9, 9], ["pt", "tmp", "tmp", "tmp"], ["ckpt-0004.pt"], False)
            and result["orphan_bytes"] > 2.9 * size,
            f"exit codes {result['codes']}; files {result['suffixes']}; {result['orphan_bytes']} B of orphans; "
            f"pruner sees {result['visible_to_pruner']}; fsync in the lesson's code: {result['fsync']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
