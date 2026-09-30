"""Exercise 5 — with an upload link at half the save cadence, the lesson's rotation deletes 6 of 20 checkpoints before they reach S3.

    Add an upload to a fake S3 (a second directory) and write the upload manifest. Defend the two-tier storage policy.

Reading of the exercise: the fake S3 is a second temporary directory.
`upload` copies one of the lesson's `save_sharded` checkpoints into it, one
object at a time (copy to `.tmp`, then rename). The shards go first,
`manifest.json` second, and `upload_manifest.json` last. The upload manifest
records the step, the local path, the remote path, and each object's bytes
and sha256. Each remote sha256 is recomputed and compared with the lesson's
manifest before the next object is sent. The upload manifest is the commit
marker: a resume from S3 takes the newest remote directory that has one, and
loads it with the lesson's `load_sharded`. "Defend the two-tier policy" is
read as: show what each tier buys and when the pair fails. The state is the
lesson's 4 demo ranks. A 20-save run with the lesson's `rotate_checkpoints`
(5 local checkpoints, rotated before each save) is simulated at two upload
speeds, measured in save intervals rather than wall time.

**ANSWER: the remote tier is what survives node loss, and the policy holds
while one upload fits in one save interval.** After `rmtree` of the whole
local tier, the step-100 checkpoint resumes from S3 byte-equal on all 4
ranks. An upload killed after 2 of 5 objects leaves `rank0.bin` and
`rank1.bin` with no marker, so resume falls back to step 100 instead of
reading a half-uploaded checkpoint. With one upload per save, 20 of 20
checkpoints reach S3, local disk never holds more than 5, and the remote
copy is at most one save behind (19). The local tier is the fast path for
the common case, a process restart on a healthy node. The remote tier covers
node loss. Neither tier alone covers both.

**FINDING: a slow link and the lesson's rotation silently lose archive
copies.** At one upload per two saves, `rotate_checkpoints` deletes steps 4,
6, 8, 10, 12 and 14 before they are uploaded, 6 of 20. It knows nothing
about uploads, and nothing reports the loss. After 20 saves, S3's newest
checkpoint is step 15, and 4 checkpoints are still waiting.

**FINDING: pinning un-uploaded checkpoints moves the failure onto the local
disk.** A rotation that skips checkpoints not yet uploaded loses none, but
local disk peaks at 11 checkpoints against a budget of 5, and grows by one
every two saves. S3 also falls further behind (newest step 9, backlog 10),
because the uploader stays on the oldest ones. With a link slower than the
save cadence, no rotation rule fixes the policy. The save interval has to be
at least the upload time.

Expected output: three PASS checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "80-checkpoint-sharded-resume"
UPLOAD = "upload_manifest.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def upload(ref, local, bucket, crash_after=None):
    """Shards first, manifest.json second, upload_manifest.json last as the commit marker."""
    local, manifest = Path(local), ref.ShardManifest.from_json(Path(local, ref.MANIFEST_NAME).read_text())
    remote, objects = Path(bucket, local.name), []
    remote.mkdir(parents=True, exist_ok=True)
    want = {**{e.path: e.sha256 for e in manifest.shards}, ref.MANIFEST_NAME: sha(local / ref.MANIFEST_NAME)}
    for i, (name, digest) in enumerate(want.items()):
        if i == crash_after:
            raise OSError("link lost mid-upload")
        shutil.copyfile(local / name, remote / f"{name}.tmp")  # one PUT: never a half object under the real name
        os.replace(remote / f"{name}.tmp", remote / name)
        if sha(remote / name) != digest:
            raise ValueError(f"remote {name} does not match its sha256 {digest[:12]}")
        objects.append({"name": name, "bytes": (remote / name).stat().st_size, "sha256": digest})
    doc = {"step": manifest.step, "local_path": str(local), "remote_path": str(remote), "objects": objects}
    Path(remote, UPLOAD + ".tmp").write_text(json.dumps(doc, indent=2))
    os.replace(Path(remote, UPLOAD + ".tmp"), Path(remote, UPLOAD))
    return doc


def latest_remote(ref, bucket, world):
    """Resume after node loss: newest remote step with a commit marker that loads."""
    for d in sorted(Path(bucket).glob("step_*"), reverse=True):
        if (d / UPLOAD).exists():
            return ref.load_sharded(str(d), world)
    return None


def pinned_rotate(parent, uploaded, keep_last=4):
    """Delete the oldest uploaded checkpoints beyond keep_last; never one still waiting."""
    dirs = sorted(Path(parent).glob("step_*"))
    for d in [d for d in dirs[: max(len(dirs) - keep_last, 0)] if d.name in uploaded]:
        shutil.rmtree(d)


def next_upload(ref, queue, local, bucket, uploaded, lost):
    """One upload slot: skip checkpoints rotation already deleted, upload the oldest that remains."""
    while queue:
        name = queue.pop(0)
        if Path(local, name).exists():
            upload(ref, Path(local, name), bucket)
            return uploaded.add(name)
        lost.append(int(name[5:]))


def backlog_run(ref, states, pinned, every, saves=20):
    """Save every interval, rotate to 5 before each save; the link uploads one checkpoint per `every` intervals."""
    queue, uploaded, lost, peak = [], set(), [], 0
    with tempfile.TemporaryDirectory(prefix="ex05-local-") as local, tempfile.TemporaryDirectory(prefix="ex05-s3-") as s3:
        for step in range(saves):
            if pinned:
                pinned_rotate(local, uploaded)
            else:
                ref.rotate_checkpoints(local, keep_last=4)
            ref.save_sharded(states, f"{local}/step_{step:04d}", step)
            queue.append(f"step_{step:04d}")
            peak = max(peak, len(list(Path(local).glob("step_*"))))
            if step % every == every - 1:
                next_upload(ref, queue, local, s3, uploaded, lost)
        newest = latest_remote(ref, s3, 4)[0].step
    return {"lost": lost, "uploaded": len(uploaded), "backlog": len(queue), "peak_local": peak, "remote_latest": newest}


def round_trip(ref, tmp, states):
    ref.save_sharded(states, f"{tmp}/local/step_0100", 100)
    doc = upload(ref, f"{tmp}/local/step_0100", f"{tmp}/s3/run")
    shutil.rmtree(f"{tmp}/local")  # node loss: the local tier is gone
    m, loaded = latest_remote(ref, f"{tmp}/s3/run", 4)
    same = all(torch.equal(loaded[r][k], states[r][k]) for r in range(4) for k in ("param_shard", "m_shard", "v_shard"))
    ref.save_sharded(states, f"{tmp}/local/step_0101", 101)
    try:
        upload(ref, f"{tmp}/local/step_0101", f"{tmp}/s3/run", crash_after=2)
    except OSError:
        pass
    return {"objects": [o["name"] for o in doc["objects"]], "remote": doc["remote_path"].endswith("s3/run/step_0100"),
            "restored": (m.step, same), "partial": sorted(p.name for p in Path(f"{tmp}/s3/run/step_0101").iterdir()),
            "after_partial": latest_remote(ref, f"{tmp}/s3/run", 4)[0].step}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    states = [ref.make_demo_state(r, 4) for r in range(4)]
    with tempfile.TemporaryDirectory(prefix="ex05-") as tmp:
        out = round_trip(ref, tmp, states)
    runs = {"matched": (False, 1), "lesson": (False, 2), "pinned": (True, 2)}
    return {**out, **{name: backlog_run(ref, states, pin, every) for name, (pin, every) in runs.items()}}


def verify(result):
    r, fast, slow, pin = result, result["matched"], result["lesson"], result["pinned"]
    return [
        practice.Check(
            "ANSWER: the remote copy survives node loss, and a link as fast as the save cadence keeps both tiers whole",
            r["objects"] == ["rank0.bin", "rank1.bin", "rank2.bin", "rank3.bin", "manifest.json"] and r["remote"]
            and r["restored"] == (100, True) and r["partial"] == ["rank0.bin", "rank1.bin"] and r["after_partial"] == 100
            and fast == {"lost": [], "uploaded": 20, "backlog": 0, "peak_local": 5, "remote_latest": 19},
            f"upload manifest objects {r['objects']}; after local rmtree resumed step, byte-equal {r['restored']}; "
            f"crashed upload left {r['partial']} and no marker, resume falls back to {r['after_partial']}; "
            f"1 upload per save: {fast}",
        ),
        practice.Check(
            "FINDING: at half the save cadence, the lesson's rotation deletes 6 of 20 checkpoints before upload",
            slow == {"lost": [4, 6, 8, 10, 12, 14], "uploaded": 10, "backlog": 4, "peak_local": 5, "remote_latest": 15},
            f"1 upload per 2 saves, rotate_checkpoints(keep_last=4) before each save: {slow}",
        ),
        practice.Check(
            "FINDING: pinning un-uploaded checkpoints loses none but grows local disk past 2x and staler remote",
            pin == {"lost": [], "uploaded": 10, "backlog": 10, "peak_local": 11, "remote_latest": 9},
            f"same link, rotation skips checkpoints not yet uploaded: {pin}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
