"""Exercise 1 — no knee: reranking lowers held-out recall@1 from 4/8 to 3/8, and N = 10..50 all rerank the same 8 docs.

    Sweep N from 5 to 50 and plot recall@1 of the reranked output. Find the knee on this fixture.

Reading of the exercise: "this fixture" is the lesson's own 8-document
`CORPUS`, its `BiEncoder`, and a `CrossEncoder` trained exactly as `main()`
trains it (`train_tiny`, 60 epochs). Recall@1 is whether the reranked top-1
is the gold document. The lesson labels only its 5 training queries (the
label-1.0 document of each), so the curve is drawn twice: on those 5, and on
8 held-out paraphrases (`HELD_OUT`, one per document) that the model never
saw. N is swept over 5, 10, 20, 30, 40, 50 as asked, and over 1..8 to show
the whole curve. The "plot" is the table the check prints.

**ANSWER: there is no knee on this fixture; the curve falls, then goes flat
at N = 8.** Reranked recall@1 over N = 5, 10, 20, 30, 40, 50:

| queries | 5 | 10 | 20 | 30 | 40 | 50 | bi-encoder alone |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 training | 5 | 3 | 3 | 3 | 3 | 3 | 4 |
| 8 held-out | 2 | 3 | 3 | 3 | 3 | 3 | 4 |

Over N = 1..8 the held-out curve is 4, 2, 2, 2, 2, 2, 3, 3. The best N is
1, which means not reranking at all. The doc's table (0.62 at N = 5 rising to
a plateau of 0.86) is labelled illustrative, and this fixture shows the
opposite shape.

**FINDING: the corpus has 8 documents, so N = 10 to 50 are the same run.**
For every held-out query, `search(q, n)` for n >= 8 returns the full corpus.
Five of the six sweep points score one identical pool.

**FINDING: the cross-encoder cannot even hold its own training queries.**
Growing the pool from 5 to all 8 documents drops training-query recall@1
from 5/5 to 3/5. The model puts d6 above d3 and d8 above d4. d6 and d8 are
the only two documents the training data never labels below 1.0.

**FINDING: `main()`'s 3 demo queries are all training queries.** That is
the leak the doc warns about under "Training data leaks into the eval".
Even so, one of the three (the authorization query) reranks d8, the
job-cancellation doc, to top-1 instead of d4.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch  # noqa: F401 - the reference module needs it
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "66-reranker-cross-encoder"
SWEEP = [5, 10, 20, 30, 40, 50]
DEMO = ["how do we abort a multipart upload", "centralized authorization check function", "how do we cancel a job"]
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


def gold_train(ref):
    """The lesson's only labelled queries: each training query's label-1.0 document."""
    by_text = {c.text: c.doc_id for c in ref.CORPUS}
    return {t.query: by_text[t.document] for t in ref.TRAIN_TRIPLES if t.label == 1.0}


def positive_only(ref):
    """Documents that no training triple labels below 1.0."""
    below_one = {t.document for t in ref.TRAIN_TRIPLES if t.label < 1}
    return sorted(c.doc_id for c in ref.CORPUS if c.text not in below_one)


def same_pool(bi):
    """Does every n >= 10 hand the reranker the whole corpus?"""
    full = sorted(c.doc_id for c in bi.search("", 8))
    return all(sorted(c.doc_id for c in bi.search(q, n)) == full for q in HELD_OUT for n in SWEEP[1:])


def demo_top1(ref, bi, model):
    return [ref.pipeline(q, bi, model, top_n=8, top_k=3)["reranked_top_k"][0][0].doc_id for q in DEMO]


def curve(ref, bi, model, queries, sizes):
    rerank = [sum(ref.rerank(model, q, bi.search(q, n), 1)[0][0].doc_id == g for q, g in queries.items()) for n in sizes]
    ceiling = [sum(g in [c.doc_id for c in bi.search(q, n)] for q, g in queries.items()) for n in sizes]
    return rerank, ceiling


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    bi = ref.BiEncoder()
    for c in ref.CORPUS:
        bi.add(c)
    model = ref.CrossEncoder()
    ref.train_tiny(model, ref.TRAIN_TRIPLES, epochs=60)  # main()'s own call
    train = gold_train(ref)
    out = {"corpus": len(ref.CORPUS), "train_queries": len(train)}
    for name, qs in (("train", train), ("held", HELD_OUT)):
        out[name] = curve(ref, bi, model, qs, SWEEP)
        out[name + "_full"] = curve(ref, bi, model, qs, range(1, 9))
    out["train_misses"] = {g: ref.rerank(model, q, bi.search(q, 8), 1)[0][0].doc_id for q, g in train.items()}
    out["positive_only"], out["same_pool"] = positive_only(ref), same_pool(bi)
    out["demo_in_train"], out["demo_top1"] = len(set(DEMO) & set(train)), demo_top1(ref, bi, model)
    return out


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: no knee; reranked recall@1 falls with N and is flat from N = 8",
            (r["train"], r["held"], r["held_full"], r["train_full"][0][0])
            == (([5, 3, 3, 3, 3, 3], [5] * 6), ([2, 3, 3, 3, 3, 3], [7] + [8] * 5), ([4, 2, 2, 2, 2, 2, 3, 3], [4, 5, 6, 6, 7, 7, 8, 8]), 4),
            f"N={SWEEP}: train {r['train'][0]}/5, held-out {r['held'][0]}/8 (bi-encoder recall@N "
            f"{r['held'][1]}); held-out N=1..8 {r['held_full'][0]}; bi-encoder alone 4/5 and 4/8",
        ),
        practice.Check(
            "FINDING: the corpus has 8 documents, so N = 10 to 50 are the same run",
            (r["corpus"], r["same_pool"]) == (8, True),
            f"{r['corpus']} docs; every n >= 10 returns the full corpus: {r['same_pool']}",
        ),
        practice.Check(
            "FINDING: the cross-encoder cannot even hold its own training queries",
            (r["train_queries"], r["train_full"][0], r["positive_only"], r["train_misses"])
            == (5, [4, 4, 5, 5, 5, 4, 3, 3], ["d6", "d8"], {"d1": "d1", "d3": "d6", "d4": "d8", "d6": "d6", "d8": "d8"}),
            f"training-query recall@1 over N=1..8: {r['train_full'][0]} of {r['train_queries']}; gold -> top-1 "
            f"at N=8 {r['train_misses']}; docs never labelled below 1.0: {r['positive_only']}",
        ),
        practice.Check(
            "FINDING: main()'s 3 demo queries are all training queries",
            (r["demo_in_train"], r["demo_top1"]) == (3, ["d1", "d8", "d8"]),
            f"{r['demo_in_train']}/3 demo queries are in TRAIN_TRIPLES; reranked top-1 {r['demo_top1']} (gold d1, d4, d8)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
