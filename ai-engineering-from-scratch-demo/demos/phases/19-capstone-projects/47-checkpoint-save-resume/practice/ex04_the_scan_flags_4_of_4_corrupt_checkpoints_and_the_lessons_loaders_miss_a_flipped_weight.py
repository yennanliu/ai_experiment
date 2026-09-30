"""Exercise 4 — the startup scan flags 4 of 4 corrupt checkpoints; the lesson's loaders miss a flipped weight, and a flipped shard under python -O.

    Add a checksum verification path that runs at startup, scans every checkpoint in the directory, and reports which ones are corrupt.

Reading of the exercise: a checkpoint directory is filled by the lesson's
own savers from its demo run (seed 11) -- four single files and three
sharded directories -- and four of the seven are damaged the way disks and
crashes damage them: a single file truncated to half, a single file with one
byte of one weight tensor flipped, a shard with one byte flipped, and a sharded
directory torn by a crash in the middle of re-saving over it. The scan
checks each single file against a sha256 sidecar written right after the
save, and each sharded directory against the hashes in its `index.json`.
It is compared with what the lesson's own loaders notice.

**ANSWER: `scan()` below reports all 4 damaged checkpoints and passes the
other 3.** The truncated and the flipped single files fail their sidecar
hash; `sharded-0024` fails at `model.shard-001.pt`, and the torn directory
fails at `model.shard-000.pt`, the one shard the crash had already replaced.

| checkpoint | damage | scan | lesson's loader |
|---|---|---|---|
| ckpt-0004.pt, ckpt-0008.pt | none | ok | loads |
| ckpt-0012.pt | truncated to 20,000 bytes | mismatch | OSError |
| ckpt-0016.pt | 1 weight byte flipped | mismatch | **loads step 16** |
| sharded-0020 | none | ok | loads |
| sharded-0024 | 1 shard byte flipped | mismatch | AssertionError |
| sharded-torn | killed mid re-save | mismatch | AssertionError |

**FINDING: the lesson's single-file checkpoint carries no checksum, and
`torch.load` does not check the zip CRC.** A flipped byte in a weight
tensor loads cleanly into the model. Only the sharded layout records
hashes, so the single-file path needs the sidecar this scan writes.

**FINDING: the sharded loader's hash checks are `assert` statements.** Under
`python -O` they are compiled out: the same flipped `sharded-0024` loads
(exit 0, step 24) instead of "failing loudly", as the doc promises.

**FINDING: re-saving shards over the same directory is not atomic as a
whole.** Each file is swapped atomically, but a crash after the first shard
leaves new shard 000 beside the old meta and shards. The scan and the loader
both catch it, and the good step-20 checkpoint that was in the directory is
gone; a sharded save needs a fresh directory per step, renamed into place.

Expected output: two PASS checks.
"""

from __future__ import annotations

import contextlib
import json
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest import mock

import harness
from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "47-checkpoint-save-resume"
DIMS = {"batches_per_epoch": 5, "batch_size": 4, "in_dim": 16, "out_dim": 4}
LOAD_O = """
import sys; from pathlib import Path; from harness import parity
ref = parity.load_reference(%r, %r, "main")
m = ref.make_model(16, 24, 4); o, s = ref.make_optimizer_and_scheduler(m, lr=0.01, total_steps=24)
print(ref.load_sharded_checkpoint(Path(sys.argv[1]), m, o, s).step)
"""


def fresh(ref, steps=0):
    ref.seed_everything(11)
    model = ref.make_model(16, 24, 4)
    opt, sched = ref.make_optimizer_and_scheduler(model, lr=0.01, total_steps=24)
    loss = torch.nn.CrossEntropyLoss()
    return model, opt, sched, ref.train_until(model, opt, sched, loss, ref.TrainState(0, 0, 0), stop_step=steps, **DIMS)


def flip(path, member="data/0", at=100):
    """Flip one byte inside a zip member's stored bytes (torch.save does not compress)."""
    info = next(i for i in zipfile.ZipFile(path).infolist() if i.filename.endswith(member))
    raw = bytearray(path.read_bytes())
    name_len, extra_len = struct.unpack("<HH", raw[info.header_offset + 26:info.header_offset + 30])
    raw[info.header_offset + 30 + name_len + extra_len + at] ^= 0xFF
    path.write_bytes(bytes(raw))


def build(ref, root):
    for step in (4, 8, 12, 16):
        path = root / f"ckpt-{step:04d}.pt"
        ref.save_checkpoint(*fresh(ref, step), path)
        ref.atomic_write_json({"sha256": ref.file_sha256(path)}, path.with_suffix(".sha256.json"))
    for name, step in (("sharded-0020", 20), ("sharded-0024", 24), ("sharded-torn", 20)):
        ref.save_sharded_checkpoint(*fresh(ref, step), root / name, num_shards=3)
    (root / "ckpt-0012.pt").write_bytes((root / "ckpt-0012.pt").read_bytes()[:20000])
    flip(root / "ckpt-0016.pt")
    flip(root / "sharded-0024" / "model.shard-001.pt")
    with mock.patch.object(ref, "file_sha256", side_effect=RuntimeError("killed")), contextlib.suppress(RuntimeError):
        ref.save_sharded_checkpoint(*fresh(ref, 24), root / "sharded-torn", num_shards=3)


def scan(ref, root):
    """The startup verification path: name -> 'ok' or the first thing that fails its hash."""
    report = {}
    for path in sorted(root.glob("*.pt")):
        want = json.loads(path.with_suffix(".sha256.json").read_text())["sha256"]
        report[path.name] = "ok" if ref.file_sha256(path) == want else "sha256 mismatch"
    for index in sorted(root.glob("*/index.json")):
        idx, folder = json.loads(index.read_text()), index.parent
        files = [("meta.pt", idx["meta_sha256"])] + [(s["path"], s["sha256"]) for s in idx["shards"]]
        bad = [n for n, sha in files if not (folder / n).is_file() or ref.file_sha256(folder / n) != sha]
        report[folder.name] = "ok" if not bad else f"sha256 mismatch: {', '.join(bad)}"
    return report


def lesson_loads(ref, path):
    model, opt, sched, _ = fresh(ref)
    loader = ref.load_sharded_checkpoint if path.is_dir() else ref.load_checkpoint
    try:
        return f"loaded step {loader(path, model, opt, sched).step}"
    except Exception as exc:  # noqa: BLE001 -- reporting what the lesson raises is the point
        return type(exc).__name__


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory(prefix="ex04-") as tmp:
        root = Path(tmp)
        build(ref, root)
        report = scan(ref, root)
        lesson = {name: lesson_loads(ref, root / name) for name in report}
        child = subprocess.run([sys.executable, "-O", "-c", LOAD_O % (PHASE, LESSON), str(root / "sharded-0024")],
                               cwd=str(Path(harness.__file__).resolve().parents[1]), capture_output=True, text=True)
    return {"scan": report, "lesson": lesson, "optimized": [child.returncode, child.stdout.strip()]}


def verify(result):
    scan, lesson = result["scan"], result["lesson"]
    bad = {"ckpt-0012.pt", "ckpt-0016.pt", "sharded-0024", "sharded-torn"}
    return [
        practice.Check(
            "ANSWER: the startup scan flags exactly the 4 damaged checkpoints of 7",
            len(scan) == 7 and {n for n, v in scan.items() if v != "ok"} == bad
            and scan["sharded-0024"].endswith("model.shard-001.pt") and scan["sharded-torn"].endswith("model.shard-000.pt"),
            f"scan {scan}",
        ),
        practice.Check(
            "FINDING: the lesson's loaders load a flipped weight silently, and a flipped shard under python -O",
            (lesson["ckpt-0012.pt"], lesson["ckpt-0016.pt"], lesson["sharded-0024"], lesson["sharded-torn"])
            == ("OSError", "loaded step 16", "AssertionError", "AssertionError") and result["optimized"] == [0, "24"],
            f"lesson loaders {lesson}; python -O on sharded-0024: exit/step {result['optimized']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
