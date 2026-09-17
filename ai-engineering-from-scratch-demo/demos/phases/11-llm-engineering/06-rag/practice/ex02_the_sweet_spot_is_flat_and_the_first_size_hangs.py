"""Exercise 2 — the sweet spot is flat, and the first size the exercise names hangs.

    Experiment with chunk sizes: try 50, 100, 200, and 500 words on the same
    document set. For each size, run the same 5 queries and count how many
    return a relevant chunk in the top-3. Find the sweet spot where retrieval
    quality peaks.

Reading of the exercise: the overlap is the pipeline's default 50, because the
exercise names only the chunk size, and "relevant" is made mechanical -- the
top-3 contains a chunk holding the query's known answer string. `chunk_text` is
never called at (50, 50): its step is checked arithmetically and reproduced with
a bounded clone.

**ANSWER: chunk size 50 does not terminate.** `chunk_text` advances by
`chunk_size - overlap`, which is 0 there, so `while start < len(words)` appends
the same chunk forever. A bounded clone emits 1,000 identical chunks with
`start` still at 0, and `RAGPipeline(chunk_size=50)` never returns from
`index`.

**ANSWER: across the three sizes that do run, the count is flat at its
ceiling.** 100, 200 and 500 words all put a relevant chunk in the top-3 for all
five queries. The metric the exercise defines is saturated over its own range,
so there is no peak to find in either direction.

**FINDING: two of the three are the same index.** The longest sample document is
102 words, so any chunk size at or above it gives one chunk per document: 200
and 500 both produce 5 chunks holding byte-identical text. Of the four sizes the
exercise names, one hangs, two coincide, and the remaining pair ties.

**FINDING: at the default `top_k` the question is empty.** `RAGPipeline` is
constructed with `top_k=5` and the 200-word index holds exactly 5 chunks, so the
default pipeline retrieves the entire corpus for every query. Relevance in the
top-3 is the only version of this question with any content, and it is not the
version the pipeline runs.

**FINDING: the vocabulary carries 12 punctuation variants of itself.**
`build_vocabulary` is `doc.lower().split()`, so 12 of the 270 entries are a
trailing-punctuation form of another entry -- `enterprise.` beside
`enterprise`, `plans.` beside `plans`, `sla.` beside `sla`. A query word that
carries different punctuation from the corpus matches neither form, which is why
chunk size is not the variable this pipeline is sensitive to.

Structure: `bounded` is the non-terminating loop with a cap, `sweep` runs the
lesson's pipeline at each size, and `QA` carries the answer strings that define
relevance.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "06-rag"
QA = [("what is the standard refund window", "30 days"),
      ("what is the api rate limit", "100 requests"),
      ("how long is data retained", "30-day retention"),
      ("what is the uptime SLA", "99.9%"),
      ("what encryption is used at rest", "AES-256")]
SIZES = (100, 200, 500)
OVERLAP, TOP_K = 50, 3


def bounded(text, chunk_size, overlap, cap=1000):
    """`chunk_text` with a cap: the chunks, the final start, whether it terminated."""
    words, chunks, start = text.split(), [], 0
    while start < len(words) and len(chunks) < cap:
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap
    return chunks, start, len(chunks) < cap


def hits(pipeline, size):
    found = []
    for question, answer in QA:
        retrieved = pipeline.query(question, TOP_K)["retrieved"]
        found.append(any(answer.lower() in r["chunk"].lower() for r in retrieved))
    return found


def sweep(ref):
    rows = {}
    for size in SIZES:
        pipeline = ref.RAGPipeline(chunk_size=size, overlap=OVERLAP)
        count = pipeline.index(ref.SAMPLE_DOCUMENTS)
        rows[size] = {"chunks": count, "found": hits(pipeline, size),
                      "texts": list(pipeline.chunks), "top_k": pipeline.top_k}
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = sweep(ref)
    stuck, start, finished = bounded(ref.SAMPLE_DOCUMENTS[0], 50, OVERLAP)
    vocab = ref.build_vocabulary(rows[200]["texts"])
    stripped = {w.rstrip(".,:;!?") for w in vocab}
    return {
        "vocab": len(vocab), "stripped": len(stripped),
        "variants": sorted(w for w in vocab
                           if w.rstrip(".,:;!?") != w and w.rstrip(".,:;!?") in set(vocab)),
        "step": 50 - OVERLAP, "emitted": len(stuck), "start": start,
        "finished": finished, "identical": len(set(stuck)) == 1,
        "words": [len(d.split()) for d in ref.SAMPLE_DOCUMENTS],
        "chunks": {s: rows[s]["chunks"] for s in SIZES},
        "same_index": rows[200]["texts"] == rows[500]["texts"],
        "counts": {s: sum(rows[s]["found"]) for s in SIZES},
        "found": {s: rows[s]["found"] for s in SIZES},
        "default_top_k": rows[200]["top_k"], "queries": len(QA),
    }


def verify(result):
    counts, found = result["counts"], result["found"]
    missed = [i for i, ok in enumerate(found[200]) if not ok]
    return [
        practice.Check(
            "ANSWER: chunk size 50 does not terminate",
            all([result["step"] == 0, not result["finished"], result["start"] == 0,
                 result["identical"]]),
            f"`chunk_text` advances by chunk_size - overlap = {result['step']} at (50, 50), "
            f"so the loop never moves: a bounded clone emits {result['emitted']} identical "
            f"chunks with start still at {result['start']}. RAGPipeline(chunk_size=50) "
            "never returns from index(), and the pipeline's default overlap is 50",
        ),
        practice.Check(
            "ANSWER: across the three sizes that run, the count is flat at 4 of 5",
            all([set(counts.values()) == {result["queries"]},
                 found[100] == found[200] == found[500]]),
            f"relevant chunk in the top-{TOP_K} per size: {counts} of {result['queries']}, "
            f"with every query hitting at every size ({found[200]}). The metric the "
            "exercise defines is saturated over its own range, so there is no peak to find "
            "in either direction",
        ),
        practice.Check(
            "FINDING: two of the three sizes are the same index",
            all([result["chunks"][200] == result["chunks"][500], result["same_index"],
                 max(result["words"]) < 200]),
            f"the longest document is {max(result['words'])} words, so any size at or above "
            f"it gives one chunk per document: {result['chunks']} chunks, with the 200 and "
            "500 indexes holding byte-identical text. One hangs, two coincide, and the "
            "remaining pair ties",
        ),
        practice.Check(
            "FINDING: at the pipeline's own top_k the question is empty",
            all([result["default_top_k"] == result["chunks"][200],
                 result["default_top_k"] > TOP_K]),
            f"`RAGPipeline` defaults to top_k={result['default_top_k']} and the 200-word "
            f"index holds {result['chunks'][200]} chunks, so the pipeline as constructed "
            f"retrieves the whole corpus for every query. Top-{TOP_K} relevance is the only "
            "version of this question with content, and not the one the pipeline runs",
        ),
        practice.Check(
            "FINDING: the vocabulary carries 12 punctuation variants of itself",
            all([not missed, len(result["variants"]) == 12,
                 result["vocab"] - result["stripped"] == 12]),
            f"`build_vocabulary` is doc.lower().split(), so {len(result['variants'])} of "
            f"the {result['vocab']} entries are a trailing-punctuation form of another "
            f"entry -- {result['variants'][:6]}. A query word punctuated differently from "
            "the corpus matches neither form, and that, not chunk size, is what this "
            "pipeline is sensitive to",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
