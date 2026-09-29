"""Exercise 3 — on the six real lesson pages the demo fails answer relevance at 0.625, and `--corpus` alone silently keeps the fixture's qrels.

    Extend the demo to take a `--corpus path` flag that loads a real corpus. Re-run the eval and the threshold check.

Reading of the exercise: `main(argv)` is the lesson's demo with an argparse
front end. `--corpus path` loads every `.md` and `.txt` file under the path
as one document, with the file stem as the doc id. It reads the qrels from
`qrels.json` in the same folder if there is one. It ingests the files,
trains the reranker on the lesson's `TRAIN_TRIPLES`, runs the lesson's
`run_eval`, applies `THRESHOLDS`, and returns the exit code: 0 if every
metric passes, 1 otherwise, plus the failing names. With no flag it runs
the lesson's fixture. The real corpus is the English pages of the six
lessons this capstone composes (lessons 64 to 69), 62 KB of markdown. They
are written to a temp folder at run time with 8 hand-written qrels, one or
two per lesson. Each qrel's gold is the lesson page that answers it.

**ANSWER: the real corpus fails the threshold check on one metric.** 6
pages become 208 chunks. recall@5 is 1.000, precision@1 0.875, mrr 0.917 and
faithfulness 1.000, all passing. answer_relevance is 0.625 against 0.75: 3
of 8 answers open with a sentence that repeats under 30% of the question's
words, such as "Read the demo output side by side." Two passing answers
open with the markdown line "## Learning Objectives", because the generator
takes everything up to the first full stop. The demo exits 1 and names
`answer_relevance`. The fixture run still exits 0.

**FINDING: `--corpus` with no qrels of its own is scored against the
fixture's d1-d12.** `run_eval` reads the module-level `EVAL_QUERIES`, so a
new corpus without new qrels scores recall@5, precision@1 and mrr of
0.000, and answer_relevance 0.250. The failure names four metrics, never
the missing qrels.

**FINDING: the qrels' answer strings are never checked.** `EvalQuery`
carries a `gold_answer_substring` ("three failed parts",
"check_permission", ...), but no metric reads it. Replacing all 4 with
"zzz" leaves every metric unchanged, and the name appears once in
`main.py`, at its declaration.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "69-end-to-end-rag-system"
PAGES = ["64-chunking-strategies-advanced", "65-hybrid-retrieval-bm25-dense", "66-reranker-cross-encoder",
         "67-query-rewriting-hyde", "68-rag-eval-precision-recall", "69-end-to-end-rag-system"]
QRELS = [
    ("how does reciprocal rank fusion merge the bm25 and dense rankings", ["l65"]),
    ("what does the cross-encoder reranker score and how is it trained", ["l66"]),
    ("how does hyde write a hypothetical document for the query", ["l67"]),
    ("which chunking strategy splits a document on markdown headers", ["l64"]),
    ("how is faithfulness of a generated answer measured", ["l68"]),
    ("what does the self-terminating demo do when a metric misses its threshold", ["l69"]),
    ("why does fixed window chunking cut sentences in half", ["l64"]),
    ("how do you decompose a multi-topic question into sub-questions", ["l67"]),
]


def make_fixture(folder, with_qrels=True):
    for page in PAGES:
        (folder / f"l{page[:2]}.md").write_text(parity.doc_text(PHASE, page), encoding="utf-8")
    if with_qrels:
        (folder / "qrels.json").write_text(json.dumps([{"query": q, "gold": g} for q, g in QRELS]))


def load_corpus(ref, folder):
    folder = pathlib.Path(folder)
    docs = [(f.stem, f.read_text(encoding="utf-8")) for f in sorted(folder.iterdir()) if f.suffix in (".md", ".txt")]
    qfile = folder / "qrels.json"
    if not qfile.exists():
        return docs, ref.EVAL_QUERIES
    rows = json.loads(qfile.read_text())
    return docs, [ref.EvalQuery(f"q{i}", r["query"], set(r["gold"]), "") for i, r in enumerate(rows)]


def main(ref, argv):
    parser = argparse.ArgumentParser(prog="rag-demo")
    parser.add_argument("--corpus", default=None)
    args = parser.parse_args(argv)
    docs, qrels = load_corpus(ref, args.corpus) if args.corpus else (ref.CORPUS, ref.EVAL_QUERIES)
    p = ref.Pipeline()
    p.ingest(docs)
    p.train_reranker_on(ref.TRAIN_TRIPLES)
    saved, ref.EVAL_QUERIES = ref.EVAL_QUERIES, qrels
    try:
        metrics = {k: round(v, 3) for k, v in ref.run_eval(p).items()}
    finally:
        ref.EVAL_QUERIES = saved
    failed = [k for k, t in ref.THRESHOLDS.items() if metrics[k] < t]
    return {"exit": 1 if failed else 0, "failed": failed, "metrics": metrics, "chunks": len(p.index.bm25.chunks)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as real, tempfile.TemporaryDirectory() as bare:
        make_fixture(pathlib.Path(real))
        make_fixture(pathlib.Path(bare), with_qrels=False)
        size = sum(len(parity.doc_text(PHASE, page).encode()) for page in PAGES)
        runs = {"fixture": main(ref, []), "real": main(ref, ["--corpus", real]), "bare": main(ref, ["--corpus", bare])}
    p = ref.build_pipeline()
    before = ref.run_eval(p)
    ref.EVAL_QUERIES = [ref.EvalQuery(e.qid, e.query, e.gold_doc_ids, "zzz") for e in ref.EVAL_QUERIES]
    source = (parity.lesson_dir(PHASE, LESSON) / "code" / "main.py").read_text()
    return {**runs, "kb": size // 1000, "substring_blind": ref.run_eval(p) == before,
            "substring_mentions": source.count("gold_answer_substring")}


def verify(result):
    r = result
    real, bare = r["real"], r["bare"]
    return [
        practice.Check(
            "ANSWER: the six lesson pages fail the threshold check on answer_relevance alone (0.625 < 0.75)",
            (real["exit"], real["failed"], real["chunks"], r["kb"], r["fixture"]["exit"]) == (1, ["answer_relevance"], 208, 62, 0)
            and real["metrics"] == {"recall@5": 1.0, "precision@1": 0.875, "mrr": 0.917,
                                    "faithfulness": 1.0, "answer_relevance": 0.625},
            f"real corpus ({r['kb']} KB, {real['chunks']} chunks): {real['metrics']}, exit {real['exit']} "
            f"failing {real['failed']}; fixture exit {r['fixture']['exit']}",
        ),
        practice.Check(
            "FINDING: --corpus without qrels is scored against the fixture's d1-d12 qrels",
            bare["exit"] == 1 and [bare["metrics"][k] for k in ("recall@5", "precision@1", "mrr")] == [0.0] * 3
            and bare["failed"] == ["recall@5", "precision@1", "mrr", "answer_relevance"],
            f"no qrels.json: {bare['metrics']}, failing {bare['failed']}",
        ),
        practice.Check(
            "FINDING: gold_answer_substring is declared and never read",
            (r["substring_blind"], r["substring_mentions"]) == (True, 1),
            f"metrics unchanged with every substring set to 'zzz': {r['substring_blind']}; "
            f"mentions in main.py: {r['substring_mentions']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
