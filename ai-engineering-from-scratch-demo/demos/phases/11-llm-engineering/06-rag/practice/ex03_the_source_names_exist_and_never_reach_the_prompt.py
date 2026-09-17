"""Exercise 3 — the source names already exist, and never reach the prompt.

    Add metadata to each chunk (source document name, chunk position). Modify
    the prompt template to include source attribution so the LLM cites its
    sources.

Reading of the exercise: "add metadata" is read against what `RAGPipeline`
already stores, and the prompt change is measured two ways -- whether an
attribution reaches the model, and whether it changes what the model returns.
The generator is the lesson's own `simple_generate`.

**ANSWER: half of it already ships, and the half that ships is unused.**
`index` builds `self.sources` (`doc_0` .. `doc_4`) and `query` returns a
`source` for every hit. But `query` hands `build_rag_prompt` a list of chunk
*texts*, and the template labels them `[Source 1]`, `[Source 2]` -- positional.
The document names exist in the pipeline and appear nowhere in the prompt.

**FINDING: the labels are per-query, so a citation cannot be resolved.** The
same chunk is `[Source 1]` for one question and `[Source 3]` for another. Across
five queries, chunk 0 carries three different labels, so "as stated in Source 1"
means nothing once the query is out of scope.

**FINDING: chunk position is not stored, and is one line away.** `RAGPipeline`
keeps `chunks`, `embeddings`, `vocab`, `idf` and `sources` -- no position. At
chunk size 100 the 11 chunks come from 5 documents, and the `index` field
`query` returns is the global offset, which coincides with the in-document
position for only 2 of the 11.

**ANSWER: attribution added to the prompt is verifiable -- and inert.** A
template carrying `[doc_2, chunk 1]` resolves the answer to a document for 5 of
5 queries. It also leaves the answer byte-identical on 5 of 5, because
`simple_generate` re-derives the query by splitting the prompt on "question:"
and then scans `retrieved_chunks` directly. Nothing added to the prompt text is
ever read.

**FINDING: so the citation has to be attached by the pipeline, not requested
from the model.** `query` already returns the source per hit; resolving the
returned sentence back to its chunk is exact, because `simple_generate` returns
a verbatim sentence -- 5 of 5 resolve to exactly one chunk.

Structure: `attributed` is the new template, `labels` tracks what each chunk is
called per query, and `resolve` maps an answer sentence back to its source.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "06-rag"
QUERIES = ["what is the enterprise refund window", "what does the professional plan cost",
           "what encryption is used at rest", "what is the starter rate limit",
           "what uptime is guaranteed for professional"]
TOP_K = 3


def positions(sources):
    """Chunk position within its own document -- derivable, and not stored."""
    seen, out = {}, []
    for source in sources:
        seen[source] = seen.get(source, -1) + 1
        out.append(seen[source])
    return out


def attributed(ref, query, hits, places):
    """The exercise's template: the document name and the chunk position, per source."""
    blocks = [f"[{hit['source']}, chunk {places[hit['index']]}]\n{hit['chunk']}"
              for hit in hits]
    return ref.build_rag_prompt(query, blocks)


def labels(pipeline, chunk_index):
    """What `build_rag_prompt` calls one chunk, across the five queries."""
    out = []
    for query in QUERIES:
        hits = pipeline.query(query, TOP_K)["retrieved"]
        for rank, hit in enumerate(hits, 1):
            if hit["index"] == chunk_index:
                out.append(rank)
    return out


def resolve(pipeline, answer):
    """Which chunks contain the returned sentence verbatim."""
    return [i for i, chunk in enumerate(pipeline.chunks) if answer and answer in chunk]


def run(ref, pipeline, places):
    rows = []
    for query in QUERIES:
        result = pipeline.query(query, TOP_K)
        hits = result["retrieved"]
        texts = [hit["chunk"] for hit in hits]
        prompt = attributed(ref, query, hits, places)
        rows.append({"plain": result["prompt"], "attributed": prompt,
                     "answer": result["answer"],
                     "with_attribution": ref.simple_generate(prompt, texts),
                     "sources": [hit["source"] for hit in hits],
                     "resolved": resolve(pipeline, result["answer"])})
    return rows


def tally(rows, pipeline):
    return {"names_in_prompt": sum(any(s in r["plain"] for s in r["sources"])
                                   for r in rows),
            "positional": [ln for ln in rows[0]["plain"].splitlines()
                           if ln.startswith("[Source")],
            "names_after": sum(any(s in r["attributed"] for s in r["sources"])
                               for r in rows),
            "unchanged": sum(r["answer"] == r["with_attribution"] for r in rows),
            "resolvable": sum(len(r["resolved"]) == 1 for r in rows),
            "sources": pipeline.sources, "fields": sorted(vars(pipeline))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pipeline = ref.RAGPipeline()
    pipeline.index(ref.SAMPLE_DOCUMENTS)
    split = ref.RAGPipeline(chunk_size=100, overlap=50)
    split.index(ref.SAMPLE_DOCUMENTS)
    rows = run(ref, pipeline, positions(pipeline.sources))
    return {
        **tally(rows, pipeline), "labels": labels(pipeline, 0), "queries": len(QUERIES),
        "split_chunks": len(split.chunks), "split_sources": len(set(split.sources)),
        "position_matches_index": sum(p == i for i, p in enumerate(positions(split.sources))),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the source names exist in the pipeline and not in the prompt",
            all([result["sources"] == [f"doc_{i}" for i in range(5)],
                 result["names_in_prompt"] == 0, "sources" in result["fields"]]),
            f"`index` builds {result['sources']} and `query` returns a source per hit, but "
            f"`query` passes `build_rag_prompt` only the chunk texts, so the names appear "
            f"in {result['names_in_prompt']} of {result['queries']} prompts. The template "
            f"labels sources positionally: {result['positional']}",
        ),
        practice.Check(
            "FINDING: the labels are per-query, so a citation cannot be resolved",
            len(set(result["labels"])) > 1,
            f"chunk 0 is labelled {result['labels']} across the five queries -- "
            f"{len(set(result['labels']))} distinct positions. 'As stated in Source 1' "
            "identifies a rank, not a document, so it means nothing once the query is out "
            "of scope",
        ),
        practice.Check(
            "FINDING: chunk position is not stored, and is one line away",
            all(["positions" not in result["fields"],
                 result["position_matches_index"] == 2]),
            f"`RAGPipeline` keeps {result['fields']} -- no position. At chunk size 100 the "
            f"{result['split_chunks']} chunks come from {result['split_sources']} "
            f"documents, and the global `index` equals the in-document position for only "
            f"{result['position_matches_index']} of them. `sources` makes it derivable in "
            "one pass, which is what this solution does",
        ),
        practice.Check(
            "ANSWER: attribution reaches the prompt, and changes nothing",
            all([result["names_after"] == result["queries"],
                 result["unchanged"] == result["queries"]]),
            f"a template carrying [doc_N, chunk M] puts the document name in "
            f"{result['names_after']} of {result['queries']} prompts -- and leaves the "
            f"answer byte-identical in {result['unchanged']} of {result['queries']}. "
            "`simple_generate` re-derives the query by splitting on 'question:' and then "
            "scans retrieved_chunks directly: prompt text is never read",
        ),
        practice.Check(
            "FINDING: the citation has to be attached by the pipeline, not asked for",
            result["resolvable"] == result["queries"],
            f"`simple_generate` returns a verbatim sentence, so the answer resolves to "
            f"exactly one chunk for {result['resolvable']} of {result['queries']} queries, "
            "and `query` already knows that chunk's source. Attribution is a lookup the "
            "pipeline can do, not a behaviour the generator can be asked for",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
