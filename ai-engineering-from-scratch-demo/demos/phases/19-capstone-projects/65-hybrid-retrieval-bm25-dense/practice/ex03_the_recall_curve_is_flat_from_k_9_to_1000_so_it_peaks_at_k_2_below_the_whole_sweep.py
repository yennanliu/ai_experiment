"""Exercise 3 -- the recall curve is flat from k = 9 to 1000, so it peaks at k <= 2, below the whole sweep.

    Sweep RRF k across 10, 30, 60, 100, 200. Plot the recall@k curve from lesson 68. Report the value of k where the curve peaks on your corpus.

Reading of the exercise: lesson 65's `HybridRetriever` (depth 10 per list)
indexes lesson 68's 12-doc `CORPUS` and is graded by lesson 68's own
`evaluate_pipeline` on its 4 `QRELS`, at recall@1, @3 and @5, plus MRR. The
"curve" is recall@1/3/5 against RRF k, printed as a table with a bar per
point. Beyond the five asked values the sweep also runs k = 1..9, and every k
from 1 to 1000 is searched for a change in any query's fused order.

**ANSWER: over 10, 30, 60, 100 and 200 the curve is flat and has no peak.**
Every one of the five gives recall@1 0.375, recall@3 0.75, recall@5 1.0 and
MRR 0.675. The fused order of all 12 docs is identical for every k from 9 to
1000; it changes only at k = 1, 2, 3, 6 and 9. The curve peaks at k = 1 and
2 (recall@1 0.625, recall@3 1.0, MRR 0.8333), below the whole requested
range. The reason is one query: for "how do production search engines fuse
two retrievers", gold d6 sits at (BM25 1, dense 4) and d12 at (2, 2). RRF
puts d6 first while 1/(k+1) + 1/(k+4) > 2/(k+2), which holds for k < 2. At
k = 2 they tie exactly and d6 wins on insertion order; beyond, the lower rank
sum wins. With two lists, a large k makes RRF rank by rank sum, so every
k above the last crossing gives the same ranking.

**FINDING: on lesson 68's corpus, fusion loses to BM25 alone at every k.**
BM25 alone scores recall@1 0.625, recall@3 1.0, MRR 0.875; dense alone
0.125, 0.5, 0.4375. At k = 60 the fused list sits between the two
(0.375, 0.75, 0.675): the mock dense list mostly drags BM25's hits down.

**FINDING: lesson 68 grades a stand-in, not this retriever.** Its
`hybrid_pipeline` is a synonym-expansion bag of words labelled "stand-in
for the lesson 65 retriever", and it scores recall@1 0.875 and MRR 1.0 on
the same qrels. The lesson 65 retriever actually scores 0.375 and 0.675 at
k = 60.

Structure: `retriever()` indexes lesson 68's corpus; `pipeline()` adapts it
to lesson 68's pipeline shape; `curve()` calls `evaluate_pipeline`;
`breakpoints()` scans k; `plot()` draws the table.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "65-hybrid-retrieval-bm25-dense"
EVAL = "68-rag-eval-precision-recall"
SWEEP = (10, 30, 60, 100, 200)
METRICS = ("recall@1", "recall@3", "recall@5", "mrr")


def retriever(ref, l68):
    """Lesson 65's HybridRetriever over lesson 68's 12-doc corpus."""
    r = ref.HybridRetriever()
    for d in l68.CORPUS:
        r.add(ref.Doc(d.doc_id, d.title, d.body))
    return r


def pipeline(r, mode="fused"):
    """Lesson 68's pipeline shape: (query, k) -> (doc ids, answer)."""
    return lambda q, k: ([d.doc_id for d, _ in r.search(q, k_each=10, k_out=12)[mode][:k]], "")


def curve(l68, fn):
    out = l68.evaluate_pipeline(fn, l68.QRELS, ks=(1, 3, 5))
    return tuple(round(out[m], 4) for m in METRICS)


def breakpoints(r, l68, top=1000):
    """Every k in 1..top at which some query's full fused order changes."""
    seen, prev = [], None
    for k in range(1, top + 1):
        r.rrf_k = k
        cur = [pipeline(r)(q.query, 12)[0] for q in l68.QRELS]
        if cur != prev:
            seen.append(k)
        prev = cur
    return seen


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    l68 = parity.load_reference(PHASE, EVAL, "main")
    r = retriever(ref, l68)
    sweep = {}
    for k in SWEEP + tuple(range(1, 10)):
        r.rrf_k = k
        sweep[k] = curve(l68, pipeline(r))
    return {
        "sweep": sweep,
        "bm25": curve(l68, pipeline(r, "bm25")), "dense": curve(l68, pipeline(r, "dense")),
        "stand_in": curve(l68, l68.hybrid_pipeline),
        "breaks": breakpoints(r, l68),
    }


def plot(sweep):
    rows = [f"k={k:<4} " + "  ".join(f"{m} {v:.3f} {'#' * round(v * 10):<10}" for m, v in zip(METRICS[:3], sweep[k]))
            for k in sorted(sweep)]
    return "\n      ".join([""] + rows)


def verify(result):
    r = result
    sweep = r["sweep"]
    flat = {sweep[k] for k in SWEEP}
    best = max(v[:3] for v in sweep.values())
    return [
        practice.Check(
            "ANSWER: flat over the five asked k; the curve peaks at k = 1 and 2",
            (flat, r["breaks"], [k for k, v in sorted(sweep.items()) if v[:3] == best], sweep[1])
            == ({(0.375, 0.75, 1.0, 0.675)}, [1, 2, 3, 6, 9], [1, 2], (0.625, 1.0, 1.0, 0.8333)),
            f"fused order changes only at k = {r['breaks']} in 1..1000; recall curve:" + plot(sweep),
        ),
        practice.Check(
            "FINDING: fusion loses to BM25 alone at every k on lesson 68's corpus",
            (r["bm25"], r["dense"], max(v[3] for v in sweep.values()) < r["bm25"][3])
            == ((0.625, 1.0, 1.0, 0.875), (0.125, 0.5, 0.75, 0.4375), True),
            f"BM25 {r['bm25']}, dense {r['dense']}, fused at k=60 {sweep[60]}",
        ),
        practice.Check(
            "FINDING: lesson 68's 'hybrid' is a stand-in that outscores the real retriever",
            r["stand_in"] == (0.875, 1.0, 1.0, 1.0),
            f"lesson 68 hybrid_pipeline {r['stand_in']} vs lesson 65 at k=60 {sweep[60]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
