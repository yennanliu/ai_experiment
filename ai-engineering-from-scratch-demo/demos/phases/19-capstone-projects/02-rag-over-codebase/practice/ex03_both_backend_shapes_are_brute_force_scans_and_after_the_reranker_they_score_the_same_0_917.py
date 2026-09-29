"""Exercise 3 -- both backend shapes are full scans with sub-millisecond p99 on 6 chunks, and after the reranker they score the same.

    Benchmark Qdrant hybrid search vs pgvector + pgvectorscale at your corpus size. Report p99 at batch size 1.

Reading of the exercise: no Qdrant or Postgres server can run inside this
repo's offline gate. So the benchmark compares the two backends' query
*shapes* on the lesson's own engine, one query at a time (batch size 1). The
Qdrant shape is one hybrid request: dense and sparse prefetches fused
server-side with RRF at Qdrant's documented default. That default is k = 2
over zero-based ranks, which is `rrf(k_rrf=1)` in the lesson's one-based
formula. The pgvector + pgvectorscale shape is dense only. pgvectorscale adds
StreamingDiskANN over pgvector and ships no BM25 or sparse index, so there is
no second list to fuse. Both shapes end in the lesson's `rerank`, keep 5, and
run 100 timed queries. "Your corpus size" is the lesson's 6 chunks. A
3,000-chunk corpus of seeded synthetic chunks, built from the same
vocabulary, shows how p99 scales. (Qdrant hybrid-queries docs and the
pgvectorscale README, read 2026-09-29.)

**ANSWER: at batch size 1 on the 6-chunk corpus, p99 is under a millisecond
for both shapes, and at 3,000 chunks it is a few milliseconds.** Wall-clock
varies by machine, so the check asserts properties and prints the numbers
(about 0.04 ms and 8 ms on the authoring laptop). The deterministic part is the
work per query. Both shapes are brute force: `DenseIndex.search` computes N
cosines per query (6 and 3,000), and `BM25Index.search` walks all N term
tables for every query term in the vocabulary. Neither engine has an ANN
index, which is the thing that separates Qdrant (HNSW) from pgvectorscale
(DiskANN).

**FINDING: the backend choice does not move MRR@10.** MRR@10 after
`rerank` is 0.917 with the lesson's k=60, with Qdrant's k=2, and with no BM25
at all (the pgvector shape). Before `rerank`, dropping BM25 costs 0.104
(0.875 to 0.771). The reranker erases that difference.

**FINDING: the index build is quadratic.** `BM25Index.add` recomputes
`avgdl` by summing every stored length, so building N chunks sums N(N+1)/2
lengths: 4,501,500 for 3,000 chunks. The lesson's six chunks average 25.7
lines, so a 2M-LOC fleet is 77,922 chunks and 3.04e9 sums. The same happens
again on every re-index.

Structure: `shape` runs one query in either backend shape; `timed` gives
p50/p99 over 100 queries; `Counted` counts the `avgdl` sums.
"""

from __future__ import annotations

import hashlib
import random
import time

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "02-rag-over-codebase"
QUESTIONS = [
    ("how is S3 multipart abort wired into retry budget", 0), ("where is authorization centralized", 3),
    ("how does rank fusion work", 5), ("AbortMultipartOnFail", 0), ("check_permission", 3), ("abortUpload", 2),
    ("what is the backoff schedule for a bucket", 1), ("where are s3 abort metrics emitted", 2),
    ("OPA policy engine query", 4), ("merge dense and sparse results", 5),
    ("who decides if a user may act on a resource", 3), ("per bucket retry budget config", 1),
]
SHAPES = {"qdrant_hybrid": (True, 1), "pgvector_dense": (False, 60), "lesson_hybrid": (True, 60)}
FLEET_LOC = 2_000_000


class Counted(list):
    """A list that counts the elements `sum()` pulls out of it."""

    pulled = 0

    def __iter__(self):
        Counted.pulled += len(self)
        return super().__iter__()


def shape(ref, dense, bm25, q, name, stage="rerank"):
    use_sparse, k_rrf = SHAPES[name]
    fused = ref.rrf(dense.search(q, k=10), bm25.search(q, k=10) if use_sparse else [], k_rrf=k_rrf)
    return ref.rerank(q, fused, top_k=5) if stage == "rerank" else fused


def mrr10(ref, dense, bm25, name, stage):
    ranks = [([c.anchor() for c, _ in shape(ref, dense, bm25, q, name, stage)][:10], ref.SAMPLE_CORPUS[g].anchor())
             for q, g in QUESTIONS]
    return round(sum(1 / (a.index(w) + 1) if w in a else 0.0 for a, w in ranks) / len(QUESTIONS), 3)


def timed(ref, dense, bm25, name):
    lat = []
    for i in range(100):
        t = time.perf_counter()
        shape(ref, dense, bm25, QUESTIONS[i % len(QUESTIONS)][0], name)
        lat.append((time.perf_counter() - t) * 1e3)
    lat.sort()
    return {"p50_ms": round(lat[49], 3), "p99_ms": round(lat[98], 3)}


def synthetic(ref, n):
    rng = random.Random(0)
    vocab = sorted({w for c in ref.SAMPLE_CORPUS for w in ref.tokenize(f"{c.symbol} {c.summary} {c.body}")})
    return list(ref.SAMPLE_CORPUS) + [
        ref.Chunk("synth", f"f{i}.py", 1, 26, "_".join(rng.sample(vocab, 2)), " ".join(rng.choices(vocab, k=12)),
                  " ".join(rng.choices(vocab, k=10))) for i in range(n - 6)]


def measure(ref, chunks):
    Counted.pulled, dense, bm25 = 0, ref.DenseIndex(), ref.BM25Index(doc_lens=Counted())
    for c in chunks:
        dense.add(c)
        bm25.add(c)
    terms = [t for q, _ in QUESTIONS for t in set(ref.tokenize(q)) if t in bm25.df]
    out = {"sums": Counted.pulled, "cosines": len(dense.vectors), "bm25_scans": len(terms) * len(bm25.tf),
           "terms": len(terms), **{s: timed(ref, dense, bm25, s) for s in SHAPES}}
    out["mrr"] = {f"{s}/{st}": mrr10(ref, dense, bm25, s, st) for s in SHAPES for st in ("fused", "rerank")}
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.hash = lambda s: int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")
    spans = [c.end_line - c.start_line + 1 for c in ref.SAMPLE_CORPUS]
    small = measure(ref, ref.SAMPLE_CORPUS)
    return {6: small, 3000: measure(ref, synthetic(ref, 3000)), "mrr": small["mrr"],
            "fleet_chunks": round(FLEET_LOC / (sum(spans) / len(spans)))}


def latency_ok(small, big):
    fast = all(small[s]["p99_ms"] < 1 and small[s]["p50_ms"] <= small[s]["p99_ms"] for s in SHAPES)
    return fast and all(big[s]["p99_ms"] > 5 * small[s]["p99_ms"] for s in SHAPES)


def verify(result):
    r, small, big = result, result[6], result[3000]
    m, f = r["mrr"], r["fleet_chunks"]
    rows = {n: ", ".join(f"{s} {d[s]}" for s in SHAPES) for n, d in (("6", small), ("3,000", big))}
    return [
        practice.Check(
            "ANSWER: p99 at batch size 1 is under 1 ms at 6 chunks for both shapes, and both are full scans",
            latency_ok(small, big) and (small["cosines"], big["cosines"]) == (6, 3000)
            and big["bm25_scans"] == 3000 * big["terms"],
            f"6 chunks: {rows['6']}; 3,000 chunks: {rows['3,000']}; cosines per query {small['cosines']} and "
            f"{big['cosines']}; BM25 scans {big['bm25_scans']} for {big['terms']} matched terms",
        ),
        practice.Check(
            "FINDING: the backend choice does not move MRR@10 after the reranker",
            (m["qdrant_hybrid/rerank"], m["pgvector_dense/rerank"], m["lesson_hybrid/rerank"]) == (0.917,) * 3
            and (m["lesson_hybrid/fused"], m["qdrant_hybrid/fused"], m["pgvector_dense/fused"]) == (0.875, 0.875, 0.771),
            f"MRR@10 {m}",
        ),
        practice.Check(
            "FINDING: the index build is quadratic",
            (small["sums"], big["sums"], f, f"{f * (f + 1) / 2:.3g}") == (21, 4501500, 77922, "3.04e+09"),
            f"lengths summed while building: {small['sums']} (6 chunks), {big['sums']} (3,000); "
            f"a 2M-LOC fleet is {f} chunks and {f * (f + 1) / 2:.3g} sums",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
