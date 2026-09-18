"""Exercise 2 — the four categories do not cover the corpus, and the filter is a no-op.

    Implement a metadata filter. Add a "category" field to each document
    (security, billing, api, product). Before running vector search, filter
    chunks to only the relevant category. Test with "What encryption is used?"
    and verify it only searches security-category chunks.

Reading of the exercise: the filter is applied before `vector_search`, as
written, and "the relevant category" is chosen by keyword from the query, since
the exercise gives no routing rule. The categories are assigned by hand from
each document's own title.

**ANSWER: the filter does exactly what the exercise asks, and the test case it
names changes nothing.** "What encryption is used?" searches 1 chunk instead of
6 and returns the same chunk in the same position, because unfiltered vector
search already ranks the security document first. Four of the five queries are
unchanged; the fifth moves, and it moves because with one chunk per category the
filter has become the retrieval.

**FINDING: two of the six documents have no category among the four.** The Q3
earnings report and the uptime SLA are neither security, billing, api nor
product. A router that must pick one of four will send an earnings question to
whichever it guesses, and a filter that respects the four will never retrieve
those documents at all.

**FINDING: after filtering, the category is the result.** The four named
categories hold 1, 1, 1 and 1 chunk, so top-3 over a filtered set returns the
whole category and the ranking inside it is moot. The filter is not narrowing a
search; it is replacing it with a lookup.

**MECHANISM: where the filter is applied decides whether the scores move.**
Filtering the embedding list leaves the IDF -- computed over all six chunks --
untouched, so the surviving chunk keeps its original score. Re-indexing the
filtered set instead recomputes IDF over 1 document, which sends every term's
weight to the same value and makes the score meaningless. The exercise says
"before running vector search", which is the first of the two.

**CONTROL: the filter earns its place on a cross-category query.** "what
encryption does the enterprise plan include" is pulled to the product document
unfiltered, because `enterprise` appears in 5 of the 6 chunks and is the query's
only in-vocabulary content word. Filtered to security, it returns the security
document. That is the one query of six where the filter changes the answer.

Structure: `CATEGORY` is the hand assignment, `route` picks a category by
keyword, and `filtered` applies it before the search.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "07-advanced-rag"
CATEGORY = ["billing", "product", "security", "api", None, None]
NAMED = ("security", "billing", "api", "product")
KEYWORDS = {"security": ("encryption", "encrypted", "audit", "compliance", "soc"),
            "billing": ("refund", "refunds", "invoice", "price", "pricing", "cost"),
            "api": ("api", "rate", "limit", "token", "endpoint", "rest"),
            "product": ("plan", "plans", "tier", "tiers", "feature", "features")}
QUERIES = ["What encryption is used?", "what is the enterprise refund window",
           "what is the starter rate limit", "what plans are available",
           "what compliance audits are done"]
CROSS = "what encryption does the enterprise plan include"


def build(ref):
    chunks = [c for d in ref.SAMPLE_DOCUMENTS for c in ref.chunk_text(d)]
    vocab = ref.build_vocabulary(chunks)
    idf = ref.compute_idf(chunks, vocab)
    return chunks, vocab, idf, [ref.tfidf_embed(c, vocab, idf) for c in chunks]


def route(query):
    """Pick one of the four named categories by keyword; the exercise gives no rule."""
    words = set(query.lower().strip("?").split())
    hits = {name: len(words & set(terms)) for name, terms in KEYWORDS.items()}
    best = max(hits.values())
    return next(name for name in NAMED if hits[name] == best and best > 0) if best else None


def unfiltered(ref, query, parts, k=1):
    chunks, vocab, idf, embeddings = parts
    return [i for i, _ in ref.vector_search(ref.tfidf_embed(query, vocab, idf),
                                            embeddings, k)]


def filtered(ref, query, parts, category, k=1):
    """The exercise's filter: restrict the candidates before `vector_search`."""
    chunks, vocab, idf, embeddings = parts
    keep = [i for i, c in enumerate(CATEGORY) if c == category]
    subset = [embeddings[i] for i in keep]
    hits = ref.vector_search(ref.tfidf_embed(query, vocab, idf), subset, k)
    return [keep[i] for i, _ in hits], len(keep)


def reindexed(ref, query, chunks, category):
    """The other reading: re-index the filtered set, which recomputes the IDF."""
    keep = [i for i, c in enumerate(CATEGORY) if c == category]
    subset = [chunks[i] for i in keep]
    vocab = ref.build_vocabulary(subset)
    idf = ref.compute_idf(subset, vocab)
    return len(set(round(v, 6) for v in idf)), len(keep)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    parts = build(ref)
    chunks, _, idf, _ = parts
    routes = [route(q) for q in QUERIES]
    rows = [(unfiltered(ref, q, parts), filtered(ref, q, parts, r))
            for q, r in zip(QUERIES, routes)]
    return {
        "chunks": len(chunks), "categories": CATEGORY, "routes": routes, **sweep(rows),
        "uncategorised": sum(c is None for c in CATEGORY),
        "sizes": {name: sum(c == name for c in CATEGORY) for name in NAMED},
        "security_reindexed": reindexed(ref, QUERIES[0], chunks, "security"),
        "full_idf_values": len({round(v, 6) for v in idf}),
        "cross_plain": unfiltered(ref, CROSS, parts),
        "cross_filtered": filtered(ref, CROSS, parts, "security")[0],
        "enterprise_chunks": sum("enterprise" in c.lower() for c in chunks),
    }


def sweep(rows):
    return {"searched": [size for _, (_, size) in rows],
            "same": [plain == picked for plain, (picked, _) in rows],
            "moved": [(i, plain, picked) for i, (plain, (picked, _)) in enumerate(rows)
                      if plain != picked]}


def verify(result):
    reindexed_values, reindexed_size = result["security_reindexed"]
    return [
        practice.Check(
            "ANSWER: the filter does what is asked, and the named test case is unmoved",
            all([result["same"][0], sum(result["same"]) == 4,
                 result["searched"][0] == 1, result["routes"][0] == "security"]),
            f"'What encryption is used?' routes to {result['routes'][0]!r} and searches "
            f"{result['searched'][0]} chunk instead of {result['chunks']} -- and returns the "
            f"same chunk in the same position. Filtered rank-1 equals unfiltered rank-1 on "
            f"{sum(result['same'])} of {len(QUERIES)} queries; the one that moves is "
            f"{result['moved']}, where the filter is doing the retrieval rather than "
            "narrowing it",
        ),
        practice.Check(
            "FINDING: two of the six documents have no category among the four",
            all([result["uncategorised"] == 2, len(NAMED) == 4]),
            f"the assignment is {result['categories']}: the Q3 earnings report and the "
            f"uptime SLA are neither security, billing, api nor product. A router that must "
            "pick one of four will send an earnings question somewhere arbitrary, and a "
            "filter that respects the four can never retrieve those two documents",
        ),
        practice.Check(
            "FINDING: after filtering, the category is the result",
            all([set(result["sizes"].values()) == {1}, max(result["searched"]) == 1]),
            f"the four named categories hold {result['sizes']} chunks, so a top-3 over any "
            "filtered set returns the whole category and the ranking inside it is moot. "
            "The filter is not narrowing a search, it is replacing one with a lookup",
        ),
        practice.Check(
            "MECHANISM: where the filter is applied decides whether the scores move",
            all([reindexed_values == 1, result["full_idf_values"] > 1,
                 reindexed_size == 1]),
            f"filtering the embedding list leaves the IDF -- computed over all "
            f"{result['chunks']} chunks, {result['full_idf_values']} distinct values -- "
            f"untouched. Re-indexing the filtered set instead recomputes IDF over "
            f"{reindexed_size} document and collapses it to {reindexed_values} value, which "
            "makes the score meaningless. The exercise asks for the first of the two",
        ),
        practice.Check(
            "CONTROL: the filter earns its place on a cross-category query",
            all([result["cross_plain"] != result["cross_filtered"],
                 result["cross_filtered"] == [2], result["enterprise_chunks"] >= 5]),
            f"{CROSS!r} is pulled to chunk {result['cross_plain']} unfiltered, because "
            f"'enterprise' appears in {result['enterprise_chunks']} of the "
            f"{result['chunks']} chunks and is the query's only in-vocabulary content word. "
            f"Filtered to security it returns {result['cross_filtered']}. That is the one "
            "query of six where the filter changes the answer",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
