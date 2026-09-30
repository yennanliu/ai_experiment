"""Exercise 5 — forging manifest and lock together passes the check, and a rerun one second later changes the hash.

    Add a manifest sha256 check on startup. The downloader should fail closed if the manifest on disk disagrees with the manifest hash in `manifest.lock`.

Reading of the exercise: "on startup" is a gate run before any download:
`startup_check()` recomputes sha256 over the manifest's bytes and compares it
with the lock the reference `ManifestWriter.write` produced. "Fail closed"
means every state that cannot prove agreement refuses -- a missing lock, a
missing manifest, an unreadable lock -- and only a cache with neither file
counts as a fresh start. The manifest is made by the reference pipeline over
the lesson's own demo corpus, then copied and damaged nine ways. An optional
second pass re-hashes each cached shard against its manifest row, the check
the doc's "Shard manifest as a contract" section describes.

**ANSWER: `startup_check()` below; it passes the intact manifest and refuses
all five broken states.**

| state | manifest-hash check | + per-shard check |
|---|---|---|
| intact | ok | ok |
| one verdict edited | refused | refused |
| same JSON, reformatted | refused | refused |
| lock deleted | refused | refused |
| manifest deleted | refused | refused |
| lock not JSON | refused | refused |
| manifest edited, lock recomputed | **ok** | **ok** |
| one cached shard byte changed | **ok** | refused |
| empty cache | fresh | fresh |

The hash covers bytes, not meaning, so a reformat with identical content is
refused as well; that is the fail-closed direction.

**FINDING: the lock is `manifest.json.lock`, not `manifest.lock`.** `write`
builds it with `with_suffix(".json.lock")`, so a check that opens the file
named in the exercise finds nothing.

**FINDING: the lock is a checksum, not a seal.** It sits beside the
manifest, unkeyed, so an edit plus a recomputed lock passes. That is the
"attacker who can edit a single file" case the doc says the manifest hash
closes. The hash also says nothing about the shards: a corrupted cached
shard passes until each shard's sha256 is checked too.

**FINDING: identical data reproduces the manifest hash only within the same second.** `write`
stamps `generated_at = int(time.time())`. A rerun in the same second
reproduces the hash; one second later it does not. Nor does the same corpus
served from another directory, since the URL is in every row. A sha256
"pinned from a commit" (doc, Use It) changes on every rerun.

Structure: `startup_check()` is the gate; `build()` runs the reference
pipeline with a pinned clock; `TAMPER` holds the nine cache states.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import tempfile
import types

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "42-large-corpus-downloader"


class ManifestMismatch(RuntimeError):
    """Startup refuses: the manifest cannot be trusted."""


def startup_check(cache, check_shards=False):
    """Fail closed unless manifest.json hashes to the lock's manifest_sha256 (or neither exists yet)."""
    manifest, lock = cache / "manifest.json", cache / "manifest.json.lock"
    if (manifest.exists(), lock.exists()) == (False, False):
        return "fresh"
    if not (manifest.exists() and lock.exists()):
        raise ManifestMismatch(f"only one of {manifest.name} and {lock.name} exists")
    try:
        pinned = json.loads(lock.read_text("utf-8"))["manifest_sha256"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ManifestMismatch(f"unreadable lock: {type(exc).__name__}") from exc
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if actual != pinned:
        raise ManifestMismatch(f"manifest sha256 {actual[:12]} != lock {str(pinned)[:12]}")
    return shard_check(cache) if check_shards else "ok"


def shard_check(cache):
    """The optional second pass: every cached shard must still hash to its manifest row."""
    for row in json.loads((cache / "manifest.json").read_text("utf-8"))["shards"]:
        shard = cache / f"{row['shard_id']}.zst"
        if not shard.exists() or hashlib.sha256(shard.read_bytes()).hexdigest() != row["sha256"]:
            raise ManifestMismatch(f"{row['shard_id']} missing or sha256 differs from manifest")
    return "ok"


def build(ref, src, cache, clock=1_700_000_000):
    """Run the reference pipeline over the lesson's demo corpus; the manifest lands in `cache`."""
    ref.time = types.SimpleNamespace(time=lambda: clock)  # the writer stamps int(time.time())
    urls = ref.build_demo_corpus(src)
    loader = ref.StreamingDownloader(cache)
    dedup = ref.Dedup(ref.MinHasher(128, 3), ref.LSHIndex(128, 32))
    manifest = ref.ManifestWriter()
    for plan in ref.ShardPlanner.from_urls(urls):
        ref.process_shard(plan, loader, dedup, manifest)
    return manifest.write(cache / "manifest.json")


def edit_verdict(cache):
    m = cache / "manifest.json"
    m.write_text(m.read_text().replace('"verdict": "near_duplicate"', '"verdict": "keep"', 1))


def forge_both(cache):
    edit_verdict(cache)
    sha = hashlib.sha256((cache / "manifest.json").read_bytes()).hexdigest()
    (cache / "manifest.json.lock").write_text(json.dumps({"manifest_sha256": sha}))


def rewrite(path, fn):
    path.write_bytes(fn(path.read_bytes()))


TAMPER = {
    "intact": lambda c: None,
    "edited_verdict": edit_verdict,
    "reformatted_json": lambda c: rewrite(c / "manifest.json", lambda b: json.dumps(json.loads(b)).encode()),
    "lock_deleted": lambda c: (c / "manifest.json.lock").unlink(),
    "manifest_deleted": lambda c: (c / "manifest.json").unlink(),
    "lock_garbage": lambda c: (c / "manifest.json.lock").write_text("not json"),
    "both_forged": forge_both,
    "shard_corrupted": lambda c: rewrite(c / "shard-0001.zst", lambda b: b[:-1] + b"\x00"),
    "empty_cache": lambda c: [p.unlink() for p in c.iterdir()],
}


def outcome(cache, check_shards):
    try:
        return startup_check(cache, check_shards)
    except ManifestMismatch as exc:
        return f"refused: {exc}"


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as d:
        src, golden = pathlib.Path(d) / "src", pathlib.Path(d) / "golden"
        sha = build(ref, src, golden)
        out = {"lock_files": sorted(p.name for p in golden.iterdir() if "lock" in p.name), "sha": sha}
        for name, tamper in TAMPER.items():
            for shards in (False, True):
                cache = pathlib.Path(d) / f"{name}{shards}"
                shutil.copytree(golden, cache)
                tamper(cache)
                out[(name, shards)] = outcome(cache, shards)
        out["rerun_same_second"] = build(ref, src, pathlib.Path(d) / "r1") == sha
        out["rerun_next_second"] = build(ref, src, pathlib.Path(d) / "r2", clock=1_700_000_001) == sha
        out["moved_source"] = build(ref, pathlib.Path(d) / "elsewhere", pathlib.Path(d) / "r3") == sha
    return out


def verify(result):
    r = result
    names = list(TAMPER)
    plain = {n: r[(n, False)].split(":")[0] for n in names}
    deep = {n: r[(n, True)].split(":")[0] for n in names}
    broken = ["edited_verdict", "reformatted_json", "lock_deleted", "manifest_deleted", "lock_garbage"]
    return [
        practice.Check(
            "ANSWER: the startup check passes the intact manifest and refuses all five broken states",
            [plain[n] for n in broken] == [deep[n] for n in broken] == ["refused"] * 5
            and (plain["intact"], deep["intact"], plain["empty_cache"]) == ("ok", "ok", "fresh"),
            f"manifest-hash check {plain}",
        ),
        practice.Check(
            "FINDING: the lock is manifest.json.lock, not manifest.lock",
            r["lock_files"] == ["manifest.json.lock"] and "`manifest.lock`" in parity.doc_text(PHASE, LESSON),
            f"ManifestWriter.write produced {r['lock_files']}",
        ),
        practice.Check(
            "FINDING: the lock is a checksum, not a seal",
            (plain["both_forged"], deep["both_forged"], plain["shard_corrupted"], deep["shard_corrupted"])
            == ("ok", "ok", "ok", "refused"),
            f"forged manifest + lock -> {r[('both_forged', True)]}; corrupted shard -> "
            f"{plain['shard_corrupted']} / with shard check: {r[('shard_corrupted', True)]}",
        ),
        practice.Check(
            "FINDING: identical data reproduces the manifest hash only within the same second",
            (r["rerun_same_second"], r["rerun_next_second"], r["moved_source"]) == (True, False, False),
            f"same second {r['rerun_same_second']}, next second {r['rerun_next_second']}, "
            f"source moved {r['moved_source']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
