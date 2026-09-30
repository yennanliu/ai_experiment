"""Exercise 4 — recall@5 is 1.000 under every chunking strategy on both corpora, and on the fixture 5 of 6 strategies build the same index.

    Add a `--strategy` flag to the chunker. Measure each strategy's contribution to end-to-end recall.

Reading of the exercise: `--strategy` picks one of six chunkers. `lesson69`
is this lesson's own recursive `Chunker` (target 400). The other five come
from Phase 19 lesson 64, which this lesson's chunker says it draws on:
`fixed_window`, `sentence_chunks`, `recursive_split`, `semantic_chunks` and
`structural_markdown`, each at its defaults. They are adapted to this
lesson's `Chunk` records. Each strategy builds a full pipeline: the chunker,
the lesson's hybrid index, the reranker trained on `TRAIN_TRIPLES`, and the
generator. It is scored with the lesson's `run_eval` on two corpora. The
first is the fixture (12 documents, 4 qrels). The second is the English
pages of lessons 64 to 69, with 8 hand-written qrels, one or two per page.
End-to-end recall is the lesson's doc-level recall@5 on the reranked
top-k. Because recall@5 saturates, doc-level recall@1 is reported too.

**ANSWER: recall@5 is 1.000 under all 6 strategies on both corpora, so no
strategy contributes to it.** The differences show only at the top:

| strategy | fixture chunks | recall@1 | real chunks | recall@1 | real p@1 |
|---|---:|---:|---:|---:|---:|
| lesson69 | 12 | 0.375 | 208 | 0.875 | 0.875 |
| fixed | 12 | 0.375 | 195 | 0.875 | 0.875 |
| sentence | 12 | 0.375 | 124 | 1.000 | 1.000 |
| recursive (lesson 64) | 12 | 0.375 | 172 | 0.500 | 0.500 |
| semantic | 18 | 0.875 | 371 | 0.875 | 0.875 |
| structural | 12 | 0.375 | 97 | 1.000 | 1.000 |

On the real pages only `structural` passes all five thresholds. The other
five fail answer_relevance (0.125 to 0.625 against 0.75).

**FINDING: on the fixture the chunker is a no-op, so the demo cannot see a
chunker regression.** The longest fixture document is 137 characters, under
every strategy's target. So 5 of 6 strategies build the same 12
one-chunk-per-document index, with identical metrics. The one that differs,
`semantic`, splits 6 two-sentence documents. That moves the distractors'
"this is unrelated" sentences out of their chunks, and precision@1 rises
from 0.500 to 1.000.

**FINDING: the lesson's recursive chunker and lesson 64's give different
answers.** On the real pages, this lesson's `Chunker` gets precision@1
0.875. Lesson 64's `recursive_split`, the strategy it names as its default,
gets 0.500.
"""

from __future__ import annotations

import argparse
import types

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "69-end-to-end-rag-system"
STRATEGIES = ["lesson69", "fixed", "sentence", "recursive", "semantic", "structural"]
PAGES = ["64-chunking-strategies-advanced", "65-hybrid-retrieval-bm25-dense", "66-reranker-cross-encoder",
         "67-query-rewriting-hyde", "68-rag-eval-precision-recall", "69-end-to-end-rag-system"]
QRELS = [
    ("how does reciprocal rank fusion merge the bm25 and dense rankings", "l65"),
    ("what does the cross-encoder reranker score and how is it trained", "l66"),
    ("how does hyde write a hypothetical document for the query", "l67"),
    ("which chunking strategy splits a document on markdown headers", "l64"),
    ("how is faithfulness of a generated answer measured", "l68"),
    ("what does the self-terminating demo do when a metric misses its threshold", "l69"),
    ("why does fixed window chunking cut sentences in half", "l64"),
    ("how do you decompose a multi-topic question into sub-questions", "l67"),
]


def adapted(ref, fn):
    """A lesson 64 strategy behind this lesson's `chunk(doc_id, text)` interface."""

    def chunk(doc_id, text):
        pieces = [c.text.strip() for c in fn(doc_id, text)]
        return [ref.Chunk(doc_id, i, t) for i, t in enumerate(p for p in pieces if p)]

    return types.SimpleNamespace(chunk=chunk)


def make_chunker(ref, c64, argv):
    parser = argparse.ArgumentParser(prog="rag-demo")
    parser.add_argument("--strategy", choices=STRATEGIES, default="lesson69")
    name = parser.parse_args(argv).strategy
    fns = {"fixed": c64.fixed_window, "sentence": c64.sentence_chunks, "recursive": c64.recursive_split,
           "semantic": c64.semantic_chunks, "structural": c64.structural_markdown}
    return ref.Chunker() if name == "lesson69" else adapted(ref, fns[name])


def run(ref, c64, strategy, docs, qrels):
    p = ref.Pipeline(chunker=make_chunker(ref, c64, ["--strategy", strategy]))
    p.ingest(docs)
    p.train_reranker_on(ref.TRAIN_TRIPLES)
    ref.EVAL_QUERIES = qrels
    m = {k: round(v, 3) for k, v in ref.run_eval(p).items()}
    m["recall@1"] = round(sum(ref.doc_level_recall(p.query(e.query).top_k, e.gold_doc_ids, 1) for e in qrels) / len(qrels), 3)
    m["chunks"] = len(p.index.bm25.chunks)
    m["passes"] = all(m[k] >= t for k, t in ref.THRESHOLDS.items())
    m["index"] = [c.text for c in p.index.bm25.chunks]
    return m


def col(rows, key):
    return [rows[s][key] for s in STRATEGIES]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    c64 = parity.load_reference(PHASE, "64-chunking-strategies-advanced", "main")
    fixture_q = ref.EVAL_QUERIES
    real = [(f"l{page[:2]}", parity.doc_text(PHASE, page)) for page in PAGES]
    real_q = [ref.EvalQuery(f"q{i}", q, {g}, "") for i, (q, g) in enumerate(QRELS)]
    fix = {s: run(ref, c64, s, ref.CORPUS, fixture_q) for s in STRATEGIES}
    rea = {s: run(ref, c64, s, real, real_q) for s in STRATEGIES}
    return {
        "fix_r5": col(fix, "recall@5"), "real_r5": col(rea, "recall@5"),
        "fix_chunks": col(fix, "chunks"), "fix_r1": col(fix, "recall@1"), "fix_p1": col(fix, "precision@1"),
        "real_chunks": col(rea, "chunks"), "real_r1": col(rea, "recall@1"), "real_p1": col(rea, "precision@1"),
        "real_rel": col(rea, "answer_relevance"), "real_pass": [s for s in STRATEGIES if rea[s]["passes"]],
        "longest": max(len(t) for _, t in ref.CORPUS), "same_index": col(fix, "index").count(fix["lesson69"]["index"]),
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: recall@5 is 1.000 under all 6 strategies on both corpora; only recall@1 moves",
            r["fix_r5"] == r["real_r5"] == [1.0] * 6 and r["fix_r1"] == [0.375] * 4 + [0.875, 0.375]
            and (r["real_chunks"], r["real_r1"]) == ([208, 195, 124, 172, 371, 97], [0.875, 0.875, 1.0, 0.5, 0.875, 1.0])
            and r["real_pass"] == ["structural"] and max(r["real_rel"][:5]) < 0.75,
            f"{STRATEGIES}: recall@5 fixture {r['fix_r5']} real {r['real_r5']}; recall@1 fixture {r['fix_r1']} "
            f"real {r['real_r1']}; real chunks {r['real_chunks']}; answer_relevance {r['real_rel']}; "
            f"passing all thresholds: {r['real_pass']}",
        ),
        practice.Check(
            "FINDING: on the fixture 5 of 6 strategies build the identical 12-chunk index",
            (r["longest"], r["same_index"], r["fix_chunks"]) == (137, 5, [12, 12, 12, 12, 18, 12])
            and r["fix_p1"] == [0.5] * 4 + [1.0, 0.5],
            f"longest doc {r['longest']} chars; same index as lesson69: {r['same_index']}/6; chunks {r['fix_chunks']}; "
            f"precision@1 {r['fix_p1']}",
        ),
        practice.Check(
            "FINDING: this lesson's recursive Chunker and lesson 64's recursive_split disagree on the real pages",
            (r["real_p1"][0], r["real_p1"][3]) == (0.875, 0.5),
            f"precision@1 lesson69 {r['real_p1'][0]} vs lesson 64 recursive {r['real_p1'][3]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
