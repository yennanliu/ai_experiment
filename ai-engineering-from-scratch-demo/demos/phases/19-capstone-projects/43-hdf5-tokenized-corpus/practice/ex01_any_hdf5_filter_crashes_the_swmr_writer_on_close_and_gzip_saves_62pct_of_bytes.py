"""Exercise 1 — any HDF5 filter crashes the reference writer on close; gzip saves 62% of bytes for about 1.1x the write time.

    Add a `--compression gzip` flag to the HDF5 writer and measure the throughput cost on the demo corpus. Defend the chosen default.

Reading of the exercise: the flag is `--compression {none,gzip}`, parsed by
argparse. It reaches the reference `HDF5ShardWriter` by adding
`compression=` to its `create_dataset` call, because the writer has no such
parameter. "Throughput cost" means two things, both measured on the shipped
`build_demo_corpus()` (12,912 tokens, 2 shards, chunk 512). The first is the
write time of `write_corpus`. The second is the read time of 250 seed-7
dataloader batches, since the trainer pays for decompression on every read.
Each figure is the best of 7 runs. Bytes on disk come from HDF5's storage
size, which is exact.

**ANSWER: default `none`. gzip costs about 1.1x write time and about 1.0x read
time, and saves 62% of the bytes. The flag has to turn SWMR off to work at
all.** Stored bytes fall from 26,624 to 10,130 with gzip, and to 7,670 with
gzip plus shuffle. The timing ratios are printed and asserted only loosely
(< 3x). The whole corpus writes in about 2 ms, so fixed overhead dominates.
`none` stays the default for two reasons. The reference writer cannot
finalize a filtered shard in SWMR mode (next finding). And on a corpus this
size the savings are 16 KB.

**FINDING: any HDF5 filter makes the reference writer crash on close and
leaves the shard locked.** `__enter__` sets `swmr_mode = True` and only then
does `__exit__` create three attributes. The HDF5 SWMR note says "The writer
cannot add new objects to the file"
(support.hdfgroup.org/documentation/hdf5/latest/_s_w_m_r_t_n.html, read
2026-09-29). Unfiltered, the attributes happen to fit the dataset's object
header. A filter pipeline message uses up that space, so gzip, lzf, shuffle
and fletcher32 each raise `RuntimeError` in `__exit__`. After that the file
cannot be reopened ("file is already open for write"), and a second failed
close in the same process segfaults inside HDF5, so each probe runs in a
child process. The headroom is thin even unfiltered: on a bare SWMR dataset,
three 64-character string attributes already fail the same way. Here the
flag gets a non-SWMR writer, which reads back and validates cleanly.

**FINDING: every other stored byte is a zero.** Tokens are bytes + 1, at most
256, but they are stored as uint16. On the ASCII demo corpus the high byte is
0 for 100% of tokens. That is why shuffle, which groups those zero bytes
together, gets to 29% of the raw size.

Structure: `flags()` patches `create_dataset` (and, for a filter, the
`swmr_mode` setter) and restores both; `run()` writes, sizes, validates and
reads one configuration.
"""

from __future__ import annotations

import argparse
import contextlib
import pathlib
import subprocess
import sys
import tempfile
import time

import h5py

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "43-hdf5-tokenized-corpus"
ROOT = pathlib.Path(practice.__file__).resolve().parent.parent
CHILD = (
    "import os, sys; sys.path.insert(0, {root!r}); from harness import practice; "
    "print(practice.load_module({path!r}).close_in_process({filters!r}), flush=True); os._exit(0)"
)
PARSER = argparse.ArgumentParser(prog="ex01")
PARSER.add_argument("--compression", choices=["none", "gzip"], default="none")


@contextlib.contextmanager
def flags(filters, swmr=True):
    """Pass `filters` to every create_dataset; optionally make swmr_mode a no-op."""
    make, prop = h5py.Group.create_dataset, h5py.File.swmr_mode
    h5py.Group.create_dataset = lambda self, *a, **k: make(self, *a, **k, **filters)
    if not swmr:
        h5py.File.swmr_mode = property(prop.fget, lambda self, value: None)
    try:
        yield
    finally:
        h5py.Group.create_dataset, h5py.File.swmr_mode = make, prop


def run(ref, filters, swmr):
    """(write s, stored bytes, validate failures, read s, zero-high-byte share)."""
    with tempfile.TemporaryDirectory() as tmp, flags(filters, swmr):
        pipe = ref.ShardedTokenizationPipeline(ref.Tokenizer(), pathlib.Path(tmp), chunk_size=512)
        t0 = time.perf_counter()
        entries = pipe.write_corpus(ref.build_demo_corpus())
        wrote = time.perf_counter() - t0
        with ref.MmapTokenStore(entries) as store:
            size = sum(d.id.get_storage_size() for d in store._datasets)
            t0 = time.perf_counter()
            ref.SlidingWindowDataloader(store, 64, 4, seed=7).take(250)
            read = time.perf_counter() - t0
            zero = float((store.get_slice(0, store.total_tokens) >> 8 == 0).mean())
        return wrote, size, ref.validate_corpus(entries), read, zero


def crashes(filters):
    """'raised' or 'closed' for the unmodified SWMR writer, in a child process."""
    code = CHILD.format(root=str(ROOT), path=__file__, filters=filters)
    child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    return child.stdout.strip() or f"exit {child.returncode}"


def close_in_process(filters):
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as tmp, flags(filters):
        try:
            with ref.HDF5ShardWriter(pathlib.Path(tmp) / "x.h5", chunk_size=512) as writer:
                writer.add_document([5] * 1000)
        except RuntimeError:
            return "raised"
        return "closed"


def best(ref, filters, swmr=False, n=7):
    runs = [run(ref, filters, swmr) for _ in range(n)]
    return (min(r[0] for r in runs), *runs[0][1:3], min(r[3] for r in runs), runs[0][4])


def solve(argv=("--compression", "gzip")):
    ref = parity.load_reference(PHASE, LESSON, "main")
    chosen = PARSER.parse_args(list(argv)).compression
    gzip = {"compression": chosen} if chosen != "none" else {}
    filters = ({"compression": "lzf"}, {"shuffle": True}, {"fletcher32": True}, gzip)
    return {
        "default": PARSER.parse_args([]).compression,
        "none": best(ref, {}, swmr=True), "gzip": best(ref, gzip),
        "shuffle": best(ref, {**gzip, "shuffle": True}),
        "crash": [crashes(f) for f in filters], "plain_crash": crashes({}),
    }


def verify(result):
    r, none, gz, sh = result, result["none"], result["gzip"], result["shuffle"]
    w_ratio, r_ratio = gz[0] / none[0], gz[3] / none[3]
    return [
        practice.Check(
            "ANSWER: default none; gzip saves 62% of bytes at a small throughput cost",
            r["default"] == "none" and (none[1], gz[1], sh[1]) == (26624, 10130, 7670)
            and none[2] == gz[2] == sh[2] == [] and w_ratio < 3 and r_ratio < 3,
            f"bytes {none[1]} -> gzip {gz[1]} ({1 - gz[1] / none[1]:.0%} saved) -> +shuffle "
            f"{sh[1]}; write {none[0] * 1e3:.2f} -> {gz[0] * 1e3:.2f} ms ({w_ratio:.2f}x), "
            f"250-batch read {r_ratio:.2f}x; all configs validate clean",
        ),
        practice.Check(
            "FINDING: any HDF5 filter makes the reference SWMR writer crash on close",
            r["crash"] == ["raised"] * 4 and r["plain_crash"] == "closed",
            f"lzf/shuffle/fletcher32/gzip crash: {r['crash']}; unfiltered crashes: {r['plain_crash']}",
        ),
        practice.Check(
            "FINDING: every other stored byte is a zero",
            none[4] == 1.0 and round(sh[1] / none[1], 2) == 0.29,
            f"high byte zero on {none[4]:.0%} of tokens; gzip+shuffle keeps {sh[1] / none[1]:.0%}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
