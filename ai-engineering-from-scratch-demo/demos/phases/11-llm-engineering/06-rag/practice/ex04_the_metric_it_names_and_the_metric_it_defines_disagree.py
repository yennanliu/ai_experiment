"""Exercise 4 — the metric it names and the metric it defines move in opposite directions.

    Implement a simple evaluation: given 10 question-answer pairs, run each
    question through the RAG pipeline, and measure what percentage of retrieved
    chunks contain the answer. This is retrieval recall at k.

Reading of the exercise: the 10 pairs are written against the lesson's own
`SAMPLE_DOCUMENTS`, with the gold chunk recorded alongside the answer string, so
both metrics can be computed from the same run. The pipeline is the lesson's
`RAGPipeline` at its defaults; k sweeps 1, 3 and 5 because 5 is the pipeline's
own `top_k`.

**ANSWER: the sentence defines precision@k and calls it recall@k, and the two
disagree.** Recall -- was the gold chunk retrieved -- is 10 of 10 at k = 1, 3
and 5. Precision -- what percentage of retrieved chunks contain the answer --
is 1.00, 0.367, 0.220. Retrieval is perfect and constant; the number the
exercise asks for falls by a factor of five across the same runs.

**MECHANISM: each answer lives in one chunk, so precision@k is capped at
about 1/k.** Nine of the ten answer strings appear in exactly one of the five
chunks, which puts the ceiling at 1.000, 0.367 and 0.220. The measured values
are exactly those, so "percentage of retrieved chunks that contain the answer"
is a measurement of k.

**FINDING: at k = 5 the metric is vacuous in the other direction too.** The
index holds 5 chunks and `RAGPipeline` defaults to `top_k=5`, so the pipeline as
constructed retrieves the entire corpus. Recall is 1.0 because nothing was left
out, and precision is the fraction of the corpus that happens to answer the
question.

**FINDING: string containment cannot localise an answer.** "30 days" occurs in 2
of the 5 chunks -- the refund window in `doc_0` and the service-credit deadline
in `doc_4` -- so a question about either is scored correct by the other. One of
the ten pairs is ambiguous under the metric and the retrieval is right anyway.

**ANSWER: the honest number is recall at 1, and it is 10 of 10.** Every query's
rank-1 chunk is its gold chunk, so this pipeline has nothing to fix at
retrieval; whatever is wrong downstream is `simple_generate`'s problem.

Structure: `QA` carries the question, the answer string and the gold chunk;
`evaluate` computes both metrics from one sweep.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "06-rag"
# (question, answer string, the chunk that answers it at the default chunk size)
QA = [("what is the standard refund window", "30 days", 0),
      ("what is the enterprise refund window", "60-day", 0),
      ("how long do refunds take to process", "5-7 business days", 0),
      ("what does the professional plan cost", "$99 per month", 1),
      ("what does enterprise pricing start at", "$500 per month", 1),
      ("what encryption is used at rest", "AES-256", 2),
      ("how long are backups retained", "30-day retention", 2),
      ("what is the starter rate limit", "100 requests per minute", 3),
      ("what happens when the rate limit is exceeded", "HTTP 429", 3),
      ("what uptime is guaranteed for professional", "99.9% uptime", 4)]
KS = (1, 3, 5)


def evaluate(pipeline, k):
    """Precision@k -- the exercise's definition -- and recall@k, from one sweep."""
    precision, recall = [], 0
    for question, answer, gold in QA:
        hits = pipeline.query(question, k)["retrieved"]
        precision.append(sum(answer.lower() in h["chunk"].lower() for h in hits) / k)
        recall += any(h["index"] == gold for h in hits)
    return {"precision": round(sum(precision) / len(QA), 3), "recall": recall}


def spread(pipeline, answer):
    return [i for i, chunk in enumerate(pipeline.chunks) if answer.lower() in chunk.lower()]


def homes_of(pipeline):
    return {answer: spread(pipeline, answer) for _, answer, _ in QA}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pipeline = ref.RAGPipeline()
    chunks = pipeline.index(ref.SAMPLE_DOCUMENTS)
    homes, rows = homes_of(pipeline), {k: evaluate(pipeline, k) for k in KS}
    return {"chunks": chunks, "top_k": pipeline.top_k, "pairs": len(QA), **report(homes, rows)}


def report(homes, rows):
    total = sum(len(homes[answer]) for _, answer, _ in QA)
    return {"precision": {k: rows[k]["precision"] for k in KS},
            "recall": {k: rows[k]["recall"] for k in KS},
            "unique": sum(len(v) == 1 for v in homes.values()),
            "ambiguous": {a: v for a, v in homes.items() if len(v) > 1},
            "present": sum(bool(v) for v in homes.values()),
            "ceiling": {k: round(min(1.0, total / (len(QA) * k)), 3) for k in KS}}


def verify(result):
    precision, recall, ceiling = result["precision"], result["recall"], result["ceiling"]
    gap = max(abs(precision[k] - ceiling[k]) for k in KS)
    return [
        practice.Check(
            "ANSWER: the sentence defines precision and calls it recall, and they disagree",
            all([set(recall.values()) == {result["pairs"]}, precision[1] == 1.0,
                 precision[5] < precision[1] / 4]),
            f"recall -- was the gold chunk retrieved -- is {recall} of {result['pairs']} at "
            f"k = 1, 3, 5. Precision -- the percentage of retrieved chunks containing the "
            f"answer -- is {precision}. Retrieval is perfect and constant while the number "
            "the exercise asks for falls by a factor of five over the same runs",
        ),
        practice.Check(
            "MECHANISM: each answer lives in one chunk, so precision is capped at about 1/k",
            all([result["unique"] == 9, gap < 0.03]),
            f"{result['unique']} of the {result['pairs']} answer strings appear in exactly "
            f"one of the {result['chunks']} chunks, so precision@k cannot exceed "
            f"{ceiling}. The measured {precision} is within {gap:.3f} of that ceiling "
            "everywhere: the metric is a measurement of k",
        ),
        practice.Check(
            "FINDING: at k = 5 the metric is vacuous in the other direction too",
            all([result["top_k"] == result["chunks"], recall[5] == result["pairs"]]),
            f"the index holds {result['chunks']} chunks and RAGPipeline defaults to "
            f"top_k={result['top_k']}, so the pipeline as constructed retrieves the whole "
            "corpus. Recall is perfect because nothing was left out, and precision is the "
            "fraction of the corpus that happens to answer the question",
        ),
        practice.Check(
            "FINDING: string containment cannot localise an answer",
            all([len(result["ambiguous"]) == 1, result["ambiguous"]["30 days"] == [0, 4]]),
            f"{list(result['ambiguous'])} occurs in chunks "
            f"{result['ambiguous']['30 days']} -- the refund window in doc_0 and the "
            "service-credit deadline in doc_4 -- so a question about either is scored "
            "correct by the other. The retrieval is right regardless; the label is not",
        ),
        practice.Check(
            "ANSWER: the honest number is recall at 1, and it is 10 of 10",
            all([recall[1] == result["pairs"], result["present"] == result["pairs"]]),
            f"every query's rank-1 chunk is its gold chunk, {recall[1]} of "
            f"{result['pairs']}, and all {result['present']} answer strings are in the "
            "corpus. This pipeline has nothing to fix at retrieval -- whatever is wrong "
            "downstream belongs to `simple_generate`",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
