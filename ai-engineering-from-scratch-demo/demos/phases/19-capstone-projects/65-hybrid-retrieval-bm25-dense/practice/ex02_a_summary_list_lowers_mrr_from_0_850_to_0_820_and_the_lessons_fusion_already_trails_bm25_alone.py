"""Exercise 2 -- a summary list lowers MRR from 0.850 to 0.820, and the lesson's fusion already trails BM25 alone.

    Add a third modality: chunk summaries indexed separately and fused as a third ranked list. Measure the gain.

Reading of the exercise: there is no LLM here, so a chunk's summary is
extractive: the first sentence of its body, without the title. The summaries
go into their own `DenseIndex` (the lesson's mock embedding), and, as a
control, into their own `BM25Index`; either is fused by the lesson's `rrf` as
a third list beside its BM25 and dense lists (depth 10, k = 60, equal
weights). The gain is measured on 14 labelled queries over the lesson's
7-doc `CORPUS` (`QUERIES` below): each doc once by its own name and once in
other words. The paraphrases avoid the gold doc's content words where
possible; six of seven still share a token with it, mostly function words
(`a`, `how`, `for`, `to`) plus `resource` and `search`, and BM25 ranks 5 of
the 7 first.

**ANSWER: the gain is negative.** MRR over the 14 queries:

| system | MRR | hits@1 | literal MRR | paraphrase MRR |
|---|---:|---:|---:|---:|
| BM25 alone | 0.8673 | 12 | 1.0 | 0.7347 |
| dense alone | 0.8060 | 10 | 1.0 | 0.6119 |
| summaries alone (dense) | 0.7031 | 8 | 0.8571 | 0.5490 |
| BM25 + dense (the lesson) | 0.8495 | 11 | 1.0 | 0.6990 |
| + summaries, dense | 0.8197 | 10 | 1.0 | 0.6395 |
| + summaries, BM25 | 0.8245 | 11 | 1.0 | 0.6490 |

Adding the summary list costs 0.0298 MRR and one hit@1, and all of the loss
is on paraphrases; literal queries stay at 1.0 in every fused system. A first
sentence is a strict subset of the text the other two lists already index,
so its list is a weaker, correlated third vote, and RRF counts it at full
weight. Indexing the summaries with BM25 instead gives the same verdict.

**FINDING: the lesson's own two-list fusion already loses to BM25 alone.**
0.8495 against 0.8673, again entirely on paraphrases (0.6990 against 0.7347),
the class the doc says dense retrieval rescues. The doc's opening claim is
that the vote "wins on every query class"; here it ties on literal queries
and loses on paraphrased ones, because the mock embedding is itself a hash of
the same tokens.

Structure: `summary()` makes the summaries; `indexes()` builds the four
indexes; `score()` fuses a named subset of them and scores MRR per class.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "65-hybrid-retrieval-bm25-dense"
# (query, gold doc, class) over the lesson's 7-doc CORPUS: each doc once by its own
# name and once in words its text does not use
QUERIES = [
    ("AbortMultipartOnFail", "d1", "literal"), ("tear down a chunked transfer that broke", "d1", "paraphrase"),
    ("uploading large files", "d2", "literal"), ("sending a huge attachment in pieces", "d2", "paraphrase"),
    ("per-bucket budgets", "d3", "literal"), ("limit on how many times a failing call gets another try", "d3", "paraphrase"),
    ("check_permission", "d4", "literal"), ("who is allowed to touch which resource", "d4", "paraphrase"),
    ("policy engine", "d5", "literal"), ("OPA wrapper with a time-to-live", "d5", "paraphrase"),
    ("search ranking", "d6", "literal"), ("merging keyword and meaning-based search results", "d6", "paraphrase"),
    ("index sizing", "d7", "literal"), ("RAM needed for embeddings", "d7", "paraphrase"),
]


def summary(doc):
    """Extractive: the body's first sentence, with no title."""
    return re.split(r"(?<=[.!?])\s+", doc.body)[0]


def indexes(ref):
    bm25, dense, sum_dense, sum_bm25 = ref.BM25Index(), ref.DenseIndex(), ref.DenseIndex(), ref.BM25Index()
    for d in ref.CORPUS:
        bm25.add(d)
        dense.add(d)
        s = ref.Doc(d.doc_id, "", summary(d))
        sum_dense.add(s)
        sum_bm25.add(s)
    return {"bm25": bm25, "dense": dense, "summary_dense": sum_dense, "summary_bm25": sum_bm25}


SYSTEMS = {
    "bm25 alone": ["bm25"], "dense alone": ["dense"], "summaries alone": ["summary_dense"],
    "bm25 + dense (lesson)": ["bm25", "dense"],
    "+ summaries, dense": ["bm25", "dense", "summary_dense"],
    "+ summaries, bm25": ["bm25", "dense", "summary_bm25"],
}


def score(ref, idx, names):
    """MRR and hits@1 overall and per query class, all lists at depth 10, RRF k = 60."""
    out = {"all": [], "literal": [], "paraphrase": []}
    for q, gold, cls in QUERIES:
        ids = [d.doc_id for d, _ in ref.rrf([idx[n].search(q, k=10) for n in names])]
        rr = 1 / (ids.index(gold) + 1) if gold in ids else 0.0
        out["all"].append(rr)
        out[cls].append(rr)
    return {c: (round(sum(v) / len(v), 4), sum(x == 1 for x in v)) for c, v in out.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    idx = indexes(ref)
    return {name: score(ref, idx, names) for name, names in SYSTEMS.items()}


def verify(result):
    r = result
    two, dense3, bm3 = r["bm25 + dense (lesson)"], r["+ summaries, dense"], r["+ summaries, bm25"]
    table = {name: v["all"] for name, v in r.items()}
    return [
        practice.Check(
            "ANSWER: the summary list lowers MRR, and only on paraphrases",
            (two["all"], dense3["all"], bm3["all"]) == ((0.8495, 11), (0.8197, 10), (0.8245, 11))
            and two["literal"] == dense3["literal"] == bm3["literal"] == (1.0, 7)
            and (two["paraphrase"], dense3["paraphrase"]) == ((0.699, 4), (0.6395, 3))
            and r["summaries alone"]["all"] == (0.7031, 8),
            f"(MRR, hits@1) of 14: {table}",
        ),
        practice.Check(
            "FINDING: the lesson's two-list fusion trails BM25 alone",
            r["bm25 alone"]["all"] == (0.8673, 12) and r["bm25 alone"]["paraphrase"] == (0.7347, 5)
            and two["all"][0] < r["bm25 alone"]["all"][0] and r["bm25 alone"]["literal"] == two["literal"],
            f"BM25 {r['bm25 alone']} vs fused {two}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
