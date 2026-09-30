"""Exercise 3 — the reference validator passes a shifted global_start that silently corrupts 49% of windows.

    Add a `--validate` mode that reads every shard, recomputes the sha256 over its tokens, and compares against `shards.json`. CI should run this before training starts.

Reading of the exercise: `--validate INDEX` loads `shards.json` and returns
a CI exit code: 0 clean, 1 on any problem. It checks four things. For each
shard, it recomputes the sha256 over the file's own `token_count` tokens and
compares it with the index. It checks that the file's count equals the
index's count. It checks that the `global_start` values are contiguous. And
it resolves each shard path next to `shards.json` when the stored path is
gone. The mode and the lesson's own `validate_corpus` are run against the
shipped demo corpus (2 shards, 12,912 tokens): once clean, once after each
of three single edits a CI gate should catch, and once after moving the
corpus directory, which it should survive.

**ANSWER: `--validate` exits 1 on each of the three edits and 0 on the clean
and the moved corpus.** The three edits are a flipped token in shard-0001, an
index `token_count` lowered by 1, and an index `global_start` moved from
6,456 to 6,400. Each is named in the problem list. The clean corpus passes,
and the moved one passes because each shard is found next to `shards.json`.

**FINDING: the reference `validate_corpus` sees only the flipped token.** It
truncates at the file's own attribute and never reads the index's
`token_count` or `global_start`. So it returns no failures for the lowered
count, which then makes the store raise `RuntimeError` on 65 of the 12,847
window positions at training time. It also returns no failures for the moved
`global_start`, which is worse. With that edit the store silently returns
wrong tokens for 6,336 of 12,848 window positions (49.3%) and raises on 176
more. The lesson's "a wrong sha256 fails the run early" holds only for the
bytes the sha covers.

**FINDING: `shards.json` pins absolute paths, and the shipped corpus is one
shard written twice.** `write_corpus` stores `str(output_dir / name)`, so
moving the directory makes `validate_corpus` raise `FileNotFoundError`
instead of reporting a failure. The two demo shards have the same sha256
(5c811f78961b...), and a per-shard hash check cannot see that. The demo
documents themselves say "deduplication is upstream of tokenization".

Structure: `validate_mode()` is the flag's body; `tamper()` applies one edit
to a fresh corpus; `windows()` counts what the store returns at every start.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import shutil
import tempfile

import h5py

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "43-hdf5-tokenized-corpus"
EDITS = {"count": (0, "token_count", 6455), "start": (1, "global_start", 6400)}
PARSER = argparse.ArgumentParser(prog="ex03")
PARSER.add_argument("--validate", metavar="SHARDS_JSON", type=pathlib.Path)


def check_shard(entry, path):  # problems with one shard file against its index row
    if not path.is_file():
        return [f"{entry.shard_id}: missing {path}"]
    with h5py.File(path, "r") as fh:
        count = int(fh["tokens"].attrs.get("token_count", -1))
        digest = hashlib.sha256(fh["tokens"][:count].astype("<u2").tobytes()).hexdigest()
    if count != entry.token_count:
        return [f"{entry.shard_id}: file has {count} tokens, index says {entry.token_count}"]
    return [] if digest == entry.sha256 else [f"{entry.shard_id}: sha256 mismatch"]


def validate_mode(ref, argv):
    """(exit code, problems) for `--validate SHARDS_JSON`."""
    index = PARSER.parse_args(argv).validate
    problems, offset = [], 0
    for entry in ref.load_index(index):
        path = pathlib.Path(entry.path)
        problems += check_shard(entry, path if path.is_file() else index.parent / path.name)
        if entry.global_start != offset:
            problems.append(f"{entry.shard_id}: global_start {entry.global_start} != {offset}")
        offset += entry.token_count
    return int(bool(problems)), problems


def tamper(root, case):
    if case in EDITS:
        row, key, value = EDITS[case]
        body = json.loads((root / "shards.json").read_text("utf-8"))
        body["shards"][row][key] = value
        (root / "shards.json").write_text(json.dumps(body), encoding="utf-8")
    if case == "flip":
        with h5py.File(root / "shard-0001.h5", "r+") as fh:
            fh["tokens"][100] = 1
    return pathlib.Path(shutil.move(root, f"{root}-moved")) if case == "moved" else root


def windows(ref, root, truth, span=65):  # store's answer at every start: ok, wrong, raised
    seen = collections.Counter()
    with ref.MmapTokenStore(ref.load_index(root / "shards.json")) as store:
        for start in range(store.total_tokens - span + 1):
            try:
                same = (store.get_slice(start, start + span) == truth[start : start + span]).all()
                seen["ok" if same else "wrong"] += 1
            except (RuntimeError, ValueError):
                seen["raised"] += 1
    return dict(seen)


def run_case(ref, case):
    """(--validate result, reference verdict, store impact, shard sha256s) on a fresh corpus."""
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "c"
        entries = ref.ShardedTokenizationPipeline(ref.Tokenizer(), root, 512).write_corpus(ref.build_demo_corpus())
        with ref.MmapTokenStore(entries) as store:
            truth = store.get_slice(0, store.total_tokens)
        root = tamper(root, case)
        try:
            verdict = ref.validate_corpus(ref.load_index(root / "shards.json"))
        except FileNotFoundError:
            verdict = "FileNotFoundError"
        impact = windows(ref, root, truth) if case in EDITS else None
        mode = validate_mode(ref, ["--validate", str(root / "shards.json")])
        return mode, verdict, impact, [e.sha256[:12] for e in entries]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {case: run_case(ref, case) for case in ("clean", "flip", "count", "start", "moved")}


def verify(result):
    r = result
    codes, refs = {c: r[c][0][0] for c in r}, {c: r[c][1] for c in r}
    return [
        practice.Check(
            "ANSWER: --validate exits 1 on each of three edits and 0 on a clean or moved corpus",
            codes == {"clean": 0, "flip": 1, "count": 1, "start": 1, "moved": 0},
            f"exit codes {codes}; problems: {[r[c][0][1][0] for c in r if r[c][0][1]]}",
        ),
        practice.Check(
            "FINDING: the reference validate_corpus sees only the flipped token",
            refs == {"clean": [], "flip": ["shard-0001"], "count": [], "start": [], "moved": "FileNotFoundError"}
            and r["count"][2] == {"ok": 12782, "raised": 65}
            and r["start"][2] == {"ok": 6336, "wrong": 6336, "raised": 176},
            f"reference verdicts {refs}; store with lowered count {r['count'][2]}, "
            f"with moved global_start {r['start'][2]}",
        ),
        practice.Check(
            "FINDING: shards.json pins absolute paths, and the shipped corpus is one shard written twice",
            r["moved"][1] == "FileNotFoundError" and r["clean"][3] == ["5c811f78961b"] * 2,
            f"moved dir: reference raises {r['moved'][1]}; shard sha256s {r['clean'][3]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
