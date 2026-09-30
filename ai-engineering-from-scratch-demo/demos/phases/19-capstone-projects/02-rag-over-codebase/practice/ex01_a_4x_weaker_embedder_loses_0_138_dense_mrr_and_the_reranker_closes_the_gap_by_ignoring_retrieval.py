"""Exercise 1 -- a 4x weaker embedder loses 0.138 dense MRR@10, and the reranker closes the gap by ignoring retrieval.

    Swap Voyage-code-3 for nomic-embed-code self-hosted. Measure the MRR@10 delta. Report whether the gap closes with re-ranking enabled.

Reading of the exercise: neither model can run offline here (one is a hosted
API, the other a 7B checkpoint), and the lesson's dense slot is `fake_embed`,
a hashed bag of words. So the swap is made where the lesson makes it: the
embedder behind `DenseIndex` is replaced by a weaker one and everything else
is the lesson's own pipeline (`BM25Index`, `rrf`, `rerank`). The "hosted"
embedder is `fake_embed` at its 64 dimensions, the "self-hosted" one at 16
dimensions (4x more hash collisions). Each is averaged over 20 hash keys, so
one lucky key cannot decide the delta. The eval is 12 labelled questions on
the lesson's 6-chunk `SAMPLE_CORPUS`, one gold chunk each. It includes the 3
questions `main.py` asks, 3 exact-symbol questions, and 6 paraphrases.
MRR@10 is measured at three stages: dense only, dense+BM25 fused by RRF,
and fused then reranked.

**ANSWER: the delta is 0.138 dense-only, 0.017 after fusion, and -0.009 with
reranking, so the gap closes.** Mean MRR@10: dense 0.785 vs 0.647, fused
0.902 vs 0.885, reranked 0.931 vs 0.940. With reranking on, the weaker
embedder comes out slightly ahead.

**FINDING: the gap closes because the reranker overrides retrieval.** Every
fused score lies between 1/66 and 2/61, a spread of 0.018. The rerank bonus is
0.1 per shared summary word and 0.9 per shared symbol word, so the fused prior
only breaks ties. A 4-dimension embedder (dense MRR 0.558) reranks to 0.960.
Reranking the whole corpus with no retrieval at all scores 0.958, and all
three embedders land within 0.03 of it.

**FINDING: retrieval never filters anything on this corpus.** `answer` asks
dense search for k=10 and the corpus has 6 chunks. So all 6 chunks reach the
reranker on 12/12 questions, whatever the embedder. The lesson's "rerank
top-50 and keep top-10" is, in the code, rerank everything and keep 5.

Structure: `use_embedder` rebinds the reference module's `hash` (a keyed
blake2b) and `fake_embed` (the dimension); `stages` runs the lesson's
pipeline stage by stage; `mrr10` scores the ranked anchors.
"""

from __future__ import annotations

import functools
import hashlib
import statistics

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "02-rag-over-codebase"
# (question, index of the gold chunk in SAMPLE_CORPUS)
QUESTIONS = [
    ("how is S3 multipart abort wired into retry budget", 0), ("where is authorization centralized", 3),
    ("how does rank fusion work", 5), ("AbortMultipartOnFail", 0), ("check_permission", 3), ("abortUpload", 2),
    ("what is the backoff schedule for a bucket", 1), ("where are s3 abort metrics emitted", 2),
    ("OPA policy engine query", 4), ("merge dense and sparse results", 5),
    ("who decides if a user may act on a resource", 3), ("per bucket retry budget config", 1),
]
KEYS = [f"embedder-{i}" for i in range(20)]
DIMS = {"hosted": 64, "self_hosted": 16, "tiny": 4}


def use_embedder(ref, orig, key, dim):
    """Swap the dense model: a keyed hash in place of the process-salted builtin."""
    ref.hash = lambda s: int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8, key=key.encode()).digest(), "big")
    ref.fake_embed = functools.partial(orig, dim=dim)


def mrr10(ref, ranked_lists):
    total = 0.0
    for (_, gold), ranked in zip(QUESTIONS, ranked_lists):
        anchors = [c.anchor() for c, _ in ranked][:10]
        gold_anchor = ref.SAMPLE_CORPUS[gold].anchor()
        total += 1 / (anchors.index(gold_anchor) + 1) if gold_anchor in anchors else 0.0
    return total / len(QUESTIONS)


def stages(ref):
    dense, bm25 = ref.DenseIndex(), ref.BM25Index()
    for chunk in ref.SAMPLE_CORPUS:
        dense.add(chunk)
        bm25.add(chunk)
    d = [dense.search(q, k=10) for q, _ in QUESTIONS]
    fused = [ref.rrf(x, bm25.search(q, k=10)) for x, (q, _) in zip(d, QUESTIONS)]
    reranked = [ref.rerank(q, f, top_k=5) for f, (q, _) in zip(fused, QUESTIONS)]
    stage_mrr = {"dense": mrr10(ref, d), "fused": mrr10(ref, fused), "rerank": mrr10(ref, reranked)}
    return stage_mrr, sum(len(f) == len(ref.SAMPLE_CORPUS) for f in fused)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    orig = ref.fake_embed
    out = {}
    for name, dim in DIMS.items():
        runs = []
        for key in KEYS:
            use_embedder(ref, orig, key, dim)
            runs.append(stages(ref))
        out[name] = {s: round(statistics.mean(r[0][s] for r in runs), 3) for s in ("dense", "fused", "rerank")}
        out[name]["whole_corpus_reranked"] = min(r[1] for r in runs)
    flat = [(c, 0.0) for c in ref.SAMPLE_CORPUS]
    out["no_retrieval_rerank"] = round(mrr10(ref, [ref.rerank(q, flat, top_k=5) for q, _ in QUESTIONS]), 3)
    out["prior_span"] = round(2 / 61 - 1 / 66, 3)
    return out


def verify(result):
    r = result
    h, s, t = r["hosted"], r["self_hosted"], r["tiny"]
    gap = {k: round(h[k] - s[k], 3) for k in ("dense", "fused", "rerank")}
    return [
        practice.Check(
            "ANSWER: the MRR@10 gap is 0.138 dense-only, 0.017 fused, and -0.009 reranked, so reranking closes it",
            gap == {"dense": 0.138, "fused": 0.017, "rerank": -0.009}
            and (h["dense"], s["dense"], h["rerank"], s["rerank"]) == (0.785, 0.647, 0.931, 0.94),
            f"hosted {h}, self-hosted {s}, gap {gap}",
        ),
        practice.Check(
            "FINDING: the gap closes because the reranker overrides retrieval",
            (t["dense"], t["rerank"], r["no_retrieval_rerank"], r["prior_span"]) == (0.558, 0.96, 0.958, 0.018)
            and all(abs(x["rerank"] - r["no_retrieval_rerank"]) < 0.03 for x in (h, s, t)),
            f"4-dim embedder: dense {t['dense']} -> reranked {t['rerank']}; rerank with no retrieval "
            f"{r['no_retrieval_rerank']}; fused-score spread {r['prior_span']} vs 0.1 per summary word",
        ),
        practice.Check(
            "FINDING: retrieval never filters anything on this corpus",
            all(r[n]["whole_corpus_reranked"] == len(QUESTIONS) for n in DIMS),
            f"questions where all 6 chunks reach the reranker: "
            f"{[r[n]['whole_corpus_reranked'] for n in DIMS]} of {len(QUESTIONS)}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
