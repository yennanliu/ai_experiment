"""Exercise 4 — an sqlite LSH index doubles index time but adds 3% end to end, because MinHash dominates.

    Move the LSH index to a shelf or sqlite file and measure throughput vs the in-memory variant.

Reading of the exercise: sqlite, since it gives an index on (band, key) and
a first-inserted order for free. `sqlite_index()` implements the two methods
the reference `Dedup` calls (`query`, `insert`) with the reference's own
`_band_key`, and the unchanged `Dedup` runs over it. Throughput is measured
twice: the index alone (signatures precomputed, best of 3), and end to end
(MinHash + index), because a slower index only matters if the index is what
bounds throughput. Two sqlite variants: one commit at the end, and a commit
per document (a crash loses at most one document). Fixture: 2,000 documents,
1,400 Zipf-word originals and 600 copies with a new footer, default width 5,
k = 128, b = 32.

**ANSWER: identical verdicts (1,400 kept, 600 near-duplicate, same keepers)
from all three; sqlite costs about 2x in the index and almost nothing end to
end.** One run on this machine (the checks assert only wide ratios):

| variant | index s / 2,000 docs | end to end s | vs memory |
|---|---:|---:|---:|
| in-memory `LSHIndex` | 0.065 | 2.73 | 1.00x |
| sqlite, one commit | 0.137 | 2.80 | 1.03x |
| sqlite, commit per doc | 1.74 | 4.40 | 1.61x |

Computing the 2,000 MinHash signatures takes 2.66 s, about 40x the
in-memory index. The in-memory index holds about 11.4 KB of heap per kept
document; the sqlite file is 2.7 MB, about 1.95 KB per kept document.

**FINDING: the "one extra disk read per document" in the doc is 32 for every
new document.** `query` has to try every band before it can say "new", so
the 2,000 documents cost 45,552 lookups: 32 for each of the 1,400 keepers,
and on average 1.25 for each of the 600 duplicates. Partitioning by the
first band hash, as the doc suggests, answers only band 0.

**FINDING: about half the in-memory index is a signature store the pipeline
never reads.** `LSHIndex.insert` keeps every full 128-value signature for
`jaccard_estimate`, which `Dedup` never calls; clearing it frees about 46% of
the index's heap.

Structure: `make_fixture()`; `sqlite_index()` is the on-disk index;
`run()` times the reference `Dedup` over one index; `footprint()` traces the
in-memory index's heap.
"""

from __future__ import annotations

import pathlib
import pickle
import random
import sqlite3
import tempfile
import time
import tracemalloc
import types

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "42-large-corpus-downloader"
REPEATS = 3
SELECT = "SELECT doc FROM buckets WHERE band=? AND key=? ORDER BY rowid LIMIT 1"  # first inserted, as the reference


def make_fixture(seed=3):
    """2,000 docs: 1,400 Zipf-word originals and 600 copies with a new footer, shuffled."""
    rng, vocab = random.Random(seed), [f"w{i}" for i in range(5000)]
    zipf = [1 / (i + 1) for i in range(5000)]
    origs = [" ".join(rng.choices(vocab, zipf, k=rng.randint(40, 120))) for _ in range(1400)]
    docs = origs + [origs[rng.randrange(1400)] + " share this post and subscribe" for _ in range(600)]
    rng.shuffle(docs)
    return docs


def sqlite_index(path, band_key, num_hashes, bands, commit_every_doc):
    """The reference LSHIndex's query/insert contract, backed by one sqlite table."""
    rows = num_hashes // bands
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE buckets (band INTEGER, key BLOB, doc TEXT)")
    db.execute("CREATE INDEX by_key ON buckets (band, key)")
    lookups = [0]

    def keys(sig):
        return [(i, band_key(sig[i * rows : (i + 1) * rows])) for i in range(bands)]

    def query(sig):
        for i, key in keys(sig):
            lookups[0] += 1
            hit = db.execute(SELECT, (i, key)).fetchone()
            if hit:
                return hit[0]
        return None

    def insert(doc_id, sig):
        db.executemany("INSERT INTO buckets VALUES (?, ?, ?)", [(i, k, doc_id) for i, k in keys(sig)])
        if commit_every_doc:
            db.commit()

    return types.SimpleNamespace(query=query, insert=insert, lookups=lookups, close=lambda: (db.commit(), db.close()))


def run(ref, docs, sigs, index):
    """Reference Dedup over precomputed signatures, so only the index is timed."""
    dedup = ref.Dedup(types.SimpleNamespace(signature=sigs.__getitem__), index)
    t0 = time.perf_counter()
    verdicts = [(v.verdict, v.collided_with) for i, t in enumerate(docs) for v in [dedup.evaluate("s", i, t)]]
    return verdicts, time.perf_counter() - t0


def footprint(ref, docs, sigs):
    """Heap bytes the in-memory index holds, and how much of it is the unused signature store."""
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    idx = ref.LSHIndex(ref.DEFAULT_NUM_HASHES, ref.DEFAULT_BANDS)
    fresh = pickle.loads(pickle.dumps(sigs))  # own int objects, so the index's share is counted
    run(ref, docs, fresh, idx)
    del fresh
    total = tracemalloc.get_traced_memory()[0] - base
    idx._signatures.clear()
    without = tracemalloc.get_traced_memory()[0] - base
    tracemalloc.stop()
    return {"heap": total, "sig_store": total - without, "kept": len(idx._buckets[0])}


def timed(ref, name, docs, sigs, path):
    """Best of REPEATS index-only runs for one variant; sqlite also reports lookups and file size."""
    runs = []
    for rep in range(REPEATS):
        db = path.with_name(f"{name}{rep}.db")
        idx = (ref.LSHIndex(128, 32) if name == "memory"
               else sqlite_index(db, ref.LSHIndex._band_key, 128, 32, name == "sqlite_commit"))
        runs.append(run(ref, docs, sigs, idx))
        getattr(idx, "close", lambda: None)()
    extra = {} if name == "memory" else {"lookups": idx.lookups[0], "db_bytes": db.stat().st_size}
    return {"verdicts": runs[-1][0], "secs": min(t for _, t in runs), **extra}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    docs = make_fixture()
    hasher = ref.MinHasher(ref.DEFAULT_NUM_HASHES, ref.DEFAULT_SHINGLE_WIDTH)
    t0 = time.perf_counter()
    sigs = dict(zip(docs, [hasher.signature(d) for d in docs]))
    out = {"sig_s": time.perf_counter() - t0}
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("memory", "sqlite_batch", "sqlite_commit"):
            out[name] = timed(ref, name, docs, sigs, pathlib.Path(tmp) / "x")
    out["memory"].update(footprint(ref, docs, sigs))
    return out


def verify(result):
    mem, batch, commit, sig = result["memory"], result["sqlite_batch"], result["sqlite_commit"], result["sig_s"]
    e2e = {k: round((sig + result[k]["secs"]) / (sig + mem["secs"]), 2) for k in ("sqlite_batch", "sqlite_commit")}
    dups = sum(v == "near_duplicate" for v, _ in mem["verdicts"])
    per_doc, share = mem["heap"] / mem["kept"], mem["sig_store"] / mem["heap"]
    same = mem["verdicts"] == batch["verdicts"] == commit["verdicts"]
    wide = (e2e["sqlite_batch"] < 1.5, commit["secs"] > 5 * mem["secs"], sig > 10 * mem["secs"])
    claim = "one extra disk read per document" in parity.doc_text(PHASE, LESSON)
    return [
        practice.Check(
            "ANSWER: sqlite gives identical verdicts and costs little end to end, because MinHash dominates",
            (same, mem["kept"], dups, wide) == (True, 1400, 600, (True, True, True)),
            f"kept {mem['kept']}, dropped {dups}; index s memory {mem['secs']:.3f} / sqlite {batch['secs']:.3f} "
            f"/ commit-per-doc {commit['secs']:.3f}; MinHash {sig:.2f} s; end to end vs memory {e2e}; "
            f"heap {per_doc / 1000:.1f} KB/doc vs file {batch['db_bytes'] / mem['kept'] / 1000:.2f} KB/doc",
        ),
        practice.Check(
            "FINDING: the doc's 'one extra disk read per document' is 32 for every new document",
            (claim, batch["lookups"], commit["lookups"], 45552 - 32 * 1400) == (True, 45552, 45552, 752),
            f"{batch['lookups']} lookups for 2,000 docs: 32 x 1,400 keepers + 752 for 600 duplicates",
        ),
        practice.Check(
            "FINDING: about half the in-memory index is a signature store the pipeline never reads",
            8000 < per_doc < 15000 and 0.35 < share < 0.6,
            f"{per_doc / 1000:.1f} KB heap per kept doc, {share:.0%} of it the unused `_signatures` store",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
