"""Exercise 4 — chunk equal to the window touches 2 chunks per sample every time; the 26 KB corpus never leaves HDF5's cache.

    Compare the dataloader throughput at chunk sizes equal to, half of, and twice the window size. Report the page-cache effect.

Reading of the exercise: the shipped demo corpus is written three times, with
chunk 32, 64 and 128 around the default window of 64. Each copy is read by
the reference `SlidingWindowDataloader` (batch 4, seed 7) for 500 batches.
Throughput is tokens per second, best of 5, and is printed but asserted only
loosely. Timing on a 26 KB corpus is noise-sized. The mechanism behind any
page-cache effect is deterministic, though: how many HDF5 chunks each window
read touches. It is counted exactly from the start positions the loader drew.
Each copy is also read with HDF5's chunk cache switched off (`rdcc_nbytes=0`),
which separates HDF5's own cache from the OS page cache.

**ANSWER: throughput rises slightly with chunk size (32 < 64 < 128, well
under 2x apart). The page-cache effect cannot show on this corpus.** Every
window read is `window_size + 1` = 65 tokens from a uniformly random start,
and the touch counts follow from that. Averaged over the 2,000 reads, chunk
32 touches 3.0 chunks per read, chunk 64 touches 2.0 and never fewer than 2,
and chunk 128 touches 1.51. Reads that straddle the shard boundary add a
chunk. Bytes decoded per read are 192, 256 and 386. The whole corpus is
25,824 bytes, and HDF5 2.0's default chunk cache is 8 MiB per dataset
(docs.h5py.org/en/stable/high/file.html, read 2026-09-29). So after the first
touch every chunk is served from HDF5's cache, not from the kernel. Turning
that cache off costs about 1.3x at every chunk size.

**FINDING: "chunk equal to the window" is the case the lesson warns about.**
The lesson says to set the chunk "to a multiple of window_size" so reads are
aligned, and that mismatched chunks halve throughput "because every sample
touches two chunks". But the loader reads window + 1 tokens from any start.
So at chunk = window, 100% of samples touch two or more chunks. Alignment would need
starts that are multiples of the chunk and reads of exactly one chunk. Half
the window is not half the throughput: it measures about 0.9x.

**FINDING: `MmapTokenStore` memory-maps nothing.** It reads through
`h5py.Dataset.__getitem__`, and a chunked dataset has no single file offset
to map: `get_offset()` is None at all three chunk sizes. The lesson's
"memory-mapped read" is h5py's chunk cache.

Structure: `touches()` counts chunks per recorded read; `rate()` times the
loader, with the store's h5py swapped for one opened with `rdcc_nbytes=0` when
the cache is off.
"""

from __future__ import annotations

import functools
import pathlib
import tempfile
import time
import types

import h5py

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "43-hdf5-tokenized-corpus"
WINDOW, BATCHES = 64, 500


def touches(entries, reads, chunk):
    """Chunks touched by each recorded (start, stop) global read."""
    counts = []
    for start, stop in reads:
        n = 0
        for e in entries:
            lo, hi = max(start, e.global_start), min(stop, e.global_start + e.token_count)
            if lo < hi:
                n += (hi - 1 - e.global_start) // chunk - (lo - e.global_start) // chunk + 1
        counts.append(n)
    return counts


def rate(ref, entries, cache, reps=5):
    """Best-of-reps tokens/s over BATCHES batches, with or without HDF5's chunk cache."""
    saved = ref.h5py
    if not cache:
        ref.h5py = types.SimpleNamespace(File=functools.partial(h5py.File, rdcc_nbytes=0))
    try:
        best = float("inf")
        for _ in range(reps):
            with ref.MmapTokenStore(entries) as store:
                loader = ref.SlidingWindowDataloader(store, WINDOW, batch_size=4, seed=7)
                t0 = time.perf_counter()
                loader.take(BATCHES)
                best = min(best, time.perf_counter() - t0)
        return BATCHES * 4 * WINDOW / best
    finally:
        ref.h5py = saved


def measure(ref, root, chunk):
    entries = ref.ShardedTokenizationPipeline(ref.Tokenizer(), root / str(chunk), chunk).write_corpus(
        ref.build_demo_corpus())
    reads = []
    with ref.MmapTokenStore(entries) as store:
        read = store.get_slice
        store.get_slice = lambda a, b: (reads.append((a, b)), read(a, b))[1]
        ref.SlidingWindowDataloader(store, WINDOW, batch_size=4, seed=7).take(BATCHES)
    with h5py.File(entries[0].path, "r") as fh:
        offset = fh["tokens"].id.get_offset()
    per = touches(entries, reads, chunk)
    return {
        "mean": round(sum(per) / len(per), 2), "spans": sorted(set(per)), "offset": offset,
        "two": sum(n >= 2 for n in per) / len(per), "bytes": round(sum(per) / len(per) * chunk * 2),
        "on": rate(ref, entries, True), "off": rate(ref, entries, False), "span": reads[0][1] - reads[0][0],
    }


def corpus_bytes(ref):
    """uint16 bytes of the demo corpus: every document plus its boundary token."""
    return sum(len(d) + 1 for docs in ref.build_demo_corpus().values() for d in docs) * 2


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sizes = (WINDOW // 2, WINDOW, WINDOW * 2)
    with tempfile.TemporaryDirectory() as tmp:
        m = [measure(ref, pathlib.Path(tmp), c) for c in sizes]
    on = [x["on"] for x in m]
    return {
        "means": tuple(x["mean"] for x in m), "bytes": tuple(x["bytes"] for x in m),
        "corpus": corpus_bytes(ref), "spread": max(on) / min(on), "half_vs_equal": on[0] / on[1],
        "equal": m[1], "offsets": [x["offset"] for x in m],
        "rates": {c: f"{x['on'] / 1e6:.1f}M tok/s, cache off {x['on'] / x['off']:.2f}x slower"
                  for c, x in zip(sizes, m)},
    }


def verify(result):
    r, equal = result, result["equal"]
    return [
        practice.Check(
            "ANSWER: 3 / 2 / 1.51 chunks per read; the whole corpus sits in HDF5's 8 MiB cache",
            r["means"] == (3.0, 2.0, 1.51) and r["bytes"] == (192, 256, 386)
            and r["corpus"] == 25824 < 8 * 2**20 and r["spread"] < 2,
            f"chunks per read 32/64/128: {r['means']}, bytes {r['bytes']}; corpus "
            f"{r['corpus']} B; throughput spread {r['spread']:.2f}x; {r['rates']}",
        ),
        practice.Check(
            "FINDING: 'chunk equal to the window' is the case the lesson warns about",
            equal["span"] == WINDOW + 1 and equal["two"] == 1.0 and min(equal["spans"]) == 2
            and r["half_vs_equal"] > 0.5,
            f"reads are {equal['span']} tokens; at chunk = window {equal['two']:.0%} of reads touch "
            f">= 2 chunks; half-window throughput is {r['half_vs_equal']:.2f}x, not 0.5x",
        ),
        practice.Check(
            "FINDING: MmapTokenStore memory-maps nothing",
            r["offsets"] == [None] * 3,
            f"get_offset() at chunk 32/64/128: {r['offsets']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
