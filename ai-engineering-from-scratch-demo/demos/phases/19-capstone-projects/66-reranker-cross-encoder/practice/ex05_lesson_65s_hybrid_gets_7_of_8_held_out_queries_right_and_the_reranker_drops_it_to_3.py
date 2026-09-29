"""Exercise 5 — lesson 65's hybrid alone gets 7 of 8 held-out queries right; chaining the reranker after it drops that to 3.

    Replace the deterministic mock bi-encoder with the one from lesson 65 and chain the two stages. Measure the change in top-K versus bi-encoder alone.

Reading of the exercise: "the one from lesson 65" is its `HybridRetriever`
(BM25 + dense + RRF), loaded from that lesson's own `code/main.py`. It indexes
this lesson's 8-doc `CORPUS` as `Doc(doc_id, "", text)`, because lesson 65's
own corpus has 7 docs and no d8. Its fused top-N becomes `Candidate`s for
this lesson's `rerank`, using the cross-encoder `main()` trains (60 epochs).
K = 3 as in `main()`. N is 5 and 8, and 8 is the whole corpus. The change in
top-K is recall@1, recall@3, how many of the 3 slots survive reranking, and on
how many queries top-1 moves. The queries are the 5 training queries and 8
held-out paraphrases, one per doc. Lesson 66's mock `BiEncoder` gets the same
treatment for comparison.

**ANSWER: chaining makes lesson 65's retriever worse.** Held-out, 8 queries:

| first stage | N | r@1 alone | r@1 chained | r@3 alone | r@3 chained | top-1 moved | slots kept |
|---|---:|---:|---:|---:|---:|---:|---:|
| hybrid (65) | 5 | 7 | 3 | 7 | 6 | 6 | 16/24 |
| hybrid (65) | 8 | 7 | 3 | 7 | 3 | 6 | 7/24 |
| mock (66) | 5 | 4 | 2 | 6 | 4 | 6 | 14/24 |
| mock (66) | 8 | 4 | 3 | 6 | 3 | 7 | 8/24 |

On the 5 training queries, hybrid at N = 5 holds 4/5 at top-1 before and after
reranking, and the mock goes from 4/5 to 5/5. At N = 8 both chains fall to 3/5.

**FINDING: the hybrid retriever is the better first stage, and the reranker
throws that away.** Alone it gets 7/8 held-out top-1 against the mock's
4/8. After reranking the two land within one query of each other (3 vs 2 at
N = 5, 3 and 3 at N = 8).

**FINDING: at N = 8 the first stage no longer matters.** Every pool is
the whole corpus, so the reranked top-3 is identical, query by query, for
both retrievers. Swapping retrievers only changes the result when N is
smaller than the corpus.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch  # noqa: F401 - the reference module needs it
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON, RETRIEVER = "19-capstone-projects", "66-reranker-cross-encoder", "65-hybrid-retrieval-bm25-dense"
K = 3
HELD_OUT = {  # one paraphrase per corpus doc, sharing few words with it
    "cancel a broken s3 upload": "d1",
    "big file gets split into pieces": "d2",
    "limit on retries for a bucket": "d3",
    "who is allowed to do what": "d4",
    "opa runtime wrapper": "d5",
    "fusing keyword and vector results": "d6",
    "memory per embedding in float32": "d7",
    "terminate a running job": "d8",
}

HELD = {  # (alone r@1, chained r@1, alone r@3, chained r@3, top-1 moved, slots kept)
    "hybrid_held_5": (7, 3, 7, 6, 6, 16), "hybrid_held_8": (7, 3, 7, 3, 6, 7),
    "mock_held_5": (4, 2, 6, 4, 6, 14), "mock_held_8": (4, 3, 6, 3, 7, 8),
}
KEYS = ("alone_r1", "chain_r1", "alone_r3", "chain_r3", "top1_moved", "overlap")


def stages(ref, hyb):
    """Two first stages over the same corpus: this lesson's mock bi-encoder and lesson 65's hybrid."""
    bi, hybrid = ref.BiEncoder(), hyb.HybridRetriever()
    for c in ref.CORPUS:
        bi.add(c)
        hybrid.add(hyb.Doc(c.doc_id, "", c.text))

    def fused(query, n):
        hits = hybrid.search(query, k_each=n, k_out=n)["fused"]
        return [ref.Candidate(d.doc_id, d.body, s) for d, s in hits]

    return {"mock": bi.search, "hybrid": fused}


def score(ref, model, first, queries, n):
    """Top-K before and after reranking, tallied over `queries` in the order of KEYS."""
    rows, chained, pool = [], {}, n
    for q, g in queries.items():
        hits = first(q, n)
        alone, chained[q], pool = [c.doc_id for c in hits][:K], [c.doc_id for c, _ in ref.rerank(model, q, hits, K)], min(pool, len(hits))
        rows.append((alone[0] == g, chained[q][0] == g, g in alone, g in chained[q], alone[0] != chained[q][0], len(set(alone) & set(chained[q]))))
    return {**dict(zip(KEYS, map(sum, zip(*rows)))), "pool": pool, "chained": chained}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    hyb = parity.load_reference(PHASE, RETRIEVER, "main")
    model = ref.CrossEncoder()
    ref.train_tiny(model, ref.TRAIN_TRIPLES, epochs=60)
    by_text = {c.text: c.doc_id for c in ref.CORPUS}
    train = {t.query: by_text[t.document] for t in ref.TRAIN_TRIPLES if t.label == 1.0}
    firsts = stages(ref, hyb)
    return {f"{name}_{split}_{n}": score(ref, model, first, qs, n) for name, first in firsts.items()
            for split, qs in (("train", train), ("held", HELD_OUT)) for n in (5, 8)}


def verify(result):
    r = result
    got = {k: tuple(r[k][f] for f in KEYS) for k in HELD}
    train = {k: (v["alone_r1"], v["chain_r1"]) for k, v in r.items() if "train" in k}
    best = max(r[k]["chain_r1"] for k in HELD)
    return [
        practice.Check(
            "ANSWER: chaining makes lesson 65's retriever worse",
            (got, train) == (HELD, {"mock_train_5": (4, 5), "mock_train_8": (4, 3), "hybrid_train_5": (4, 4), "hybrid_train_8": (4, 3)}),
            f"held-out {KEYS}: {got}; training r@1 alone/chained {train}",
        ),
        practice.Check(
            "FINDING: the hybrid retriever is the better first stage, and the reranker throws that away",
            (r["hybrid_held_5"]["alone_r1"], r["mock_held_5"]["alone_r1"], best) == (7, 4, 3),
            f"alone r@1 hybrid {r['hybrid_held_5']['alone_r1']}/8, mock {r['mock_held_5']['alone_r1']}/8; "
            f"best chained r@1 {best}/8",
        ),
        practice.Check(
            "FINDING: at N = 8 the first stage no longer matters",
            (r["hybrid_held_8"]["pool"], r["hybrid_held_8"]["chained"] == r["mock_held_8"]["chained"],
             r["hybrid_held_5"]["chained"] == r["mock_held_5"]["chained"]) == (8, True, False),
            f"N=8 pools {r['hybrid_held_8']['pool']} docs; chained top-3 identical for both retrievers: "
            f"{r['hybrid_held_8']['chained'] == r['mock_held_8']['chained']} (N=5: "
            f"{r['hybrid_held_5']['chained'] == r['mock_held_5']['chained']})",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
