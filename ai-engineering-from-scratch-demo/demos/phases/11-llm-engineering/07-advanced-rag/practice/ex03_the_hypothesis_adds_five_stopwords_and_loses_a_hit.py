"""Exercise 3 — the hypothesis adds five stopwords to the query and loses a hit.

    Build a full HyDE pipeline using the simple generate function from
    Lesson 06. Compare retrieval quality (top-3 relevance) between direct query
    search and HyDE search on all 5 test queries. HyDE should improve results
    for vague queries.

Reading of the exercise: the lesson's `hyde_search` already is the HyDE
pipeline, and lesson 06's `simple_generate` is used for the generation half
because the exercise names it -- it is loaded from the reference repo rather
than reimplemented. "Top-3 relevance" is whether the answering document is in
the top 3.

**ANSWER: direct search gets 5 of 5 and HyDE gets 4 of 5.** HyDE loses the
encryption query, whose answering document falls out of the top 3 entirely, and
gains nothing anywhere. On this corpus it is strictly worse.

**MECHANISM: `hyde_generate_hypothesis` is a template, so the "document" it
writes contributes five stopwords and nothing else.** The hypothesis is 30 words
against the query's 6, and of the words it adds, the ones that exist in the
corpus vocabulary are `and`, `on`, `that`, `the` and `to`. Every content word it
invents -- documentation, policies, procedures, requirements -- is absent from
the corpus, so it cannot retrieve anything, and the five stopwords it does add
re-weight the ranking.

**FINDING: the query it breaks has no content words to start with.** "what
encryption is used at rest" contributes `at`, `is` and `rest` to the embedding:
the corpus writes "encrypted", not "encryption", and "using", not "used". Direct
search finds the security document on the strength of `rest`, and adding five
more stopwords is enough to displace it.

**FINDING: the topic extraction drops words it should keep.** Its filler list
includes `is`, `at` and `the`, so the topic for that query is "encryption used
rest" -- which then appears in the hypothesis alongside the quoted query, giving
those three terms double weight while the rest of the query is diluted.

**CONTROL: strip the boilerplate and HyDE returns to parity.** A hypothesis
that is the query plus the extracted topic and nothing else matches direct
search on 5 of 5. The idea needs a generator that knows the corpus vocabulary;
a template cannot supply one.

Structure: `top3` runs both arms, `added` measures what the hypothesis
contributes, and `lean_hyde` is the boilerplate-free control.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "07-advanced-rag"
QUERIES = ["what encryption is used at rest", "what is the enterprise refund window",
           "what is the starter rate limit", "what uptime is guaranteed",
           "how much does professional cost"]
GOLD = [2, 0, 3, 5, 1]
TOP_K = 3


def build(ref):
    chunks = [c for d in ref.SAMPLE_DOCUMENTS for c in ref.chunk_text(d)]
    vocab = ref.build_vocabulary(chunks)
    idf = ref.compute_idf(chunks, vocab)
    return chunks, vocab, idf, [ref.tfidf_embed(c, vocab, idf) for c in chunks]


def direct(ref, query, parts):
    _, vocab, idf, embeddings = parts
    return [i for i, _ in ref.vector_search(ref.tfidf_embed(query, vocab, idf),
                                            embeddings, TOP_K)]


def hyde(ref, query, parts):
    _, vocab, idf, embeddings = parts
    hits, hypothesis = ref.hyde_search(query, embeddings, vocab, idf, TOP_K)
    return [i for i, _ in hits], hypothesis


def lean_hyde(ref, query, parts):
    """The control: the query plus its extracted topic, with no template prose."""
    _, vocab, idf, embeddings = parts
    hypothesis = ref.hyde_generate_hypothesis(query)
    topic = hypothesis.split("involves")[0].split(", ")[-1] if "involves" in hypothesis \
        else query
    lean = f"{query} {topic}"
    return [i for i, _ in ref.vector_search(ref.tfidf_embed(lean, vocab, idf),
                                            embeddings, TOP_K)]


def added(vocab, query, hypothesis):
    """Which in-vocabulary words the hypothesis contributes that the query did not."""
    in_query = {w for w in query.lower().split() if w in vocab}
    in_hypothesis = {w.strip("'.,:") for w in hypothesis.lower().split()}
    return sorted({w for w in in_hypothesis if w in vocab} - in_query), sorted(in_query)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    lesson06 = parity.load_reference(PHASE, "06-rag", "main")
    parts = build(ref)
    chunks, vocab = parts[0], parts[1]
    rows = [(direct(ref, q, parts), hyde(ref, q, parts)) for q in QUERIES]
    extra, present = added(vocab, QUERIES[0], rows[0][1][1])
    hypothesis, top = rows[0][1][1], [chunks[i] for i in rows[0][1][0]]
    return {
        **scores(rows), "queries": len(QUERIES),
        "hypothesis_words": len(hypothesis.split()), "query_words": len(QUERIES[0].split()),
        "added": extra, "query_in_vocabulary": present,
        "topic": hypothesis.split("documentation, ")[-1].split(" involves")[0],
        "generated": lesson06.simple_generate(
            ref.build_rag_prompt(QUERIES[0], top), top),
        "lean": sum(g in lean for lean, g
                    in zip((lean_hyde(ref, q, parts) for q in QUERIES), GOLD)),
    }


def scores(rows):
    return {"direct": sum(g in plain for (plain, _), g in zip(rows, GOLD)),
            "hyde": sum(g in hyp for (_, (hyp, _)), g in zip(rows, GOLD)),
            "lost": [i for i, ((plain, (hyp, _)), g) in enumerate(zip(rows, GOLD))
                     if g in plain and g not in hyp],
            "gained": [i for i, ((plain, (hyp, _)), g) in enumerate(zip(rows, GOLD))
                       if g not in plain and g in hyp]}


def verify(result):
    return [
        practice.Check(
            "ANSWER: direct gets 5 of 5 and HyDE gets 4 of 5",
            all([result["direct"] == 5, result["hyde"] == 4, result["lost"] == [0],
                 result["gained"] == []]),
            f"top-{TOP_K} relevance: direct {result['direct']} of {result['queries']}, HyDE "
            f"{result['hyde']}. HyDE loses query {result['lost']} -- the answering document "
            f"falls out of the top {TOP_K} -- and gains on "
            f"{len(result['gained'])}. On this corpus it is strictly worse",
        ),
        practice.Check(
            "MECHANISM: the hypothesis contributes five stopwords and nothing else",
            all([result["added"] == ["and", "on", "that", "the", "to"],
                 result["hypothesis_words"] == 30]),
            f"the hypothesis is {result['hypothesis_words']} words against the query's "
            f"{result['query_words']}, and the in-vocabulary words it adds are "
            f"{result['added']}. Every content word it invents -- documentation, policies, "
            "procedures, requirements -- is absent from the corpus, so it retrieves nothing "
            "and the stopwords re-weight the ranking",
        ),
        practice.Check(
            "FINDING: the query it breaks has no content words to start with",
            result["query_in_vocabulary"] == ["at", "is", "rest"],
            f"{QUERIES[0]!r} contributes {result['query_in_vocabulary']} to the embedding: "
            "the corpus writes 'encrypted', not 'encryption', and 'using', not 'used'. "
            "Direct search finds the security document on the strength of 'rest', and five "
            "more stopwords are enough to displace it",
        ),
        practice.Check(
            "FINDING: the topic extraction drops words it should keep",
            result["topic"] == "encryption used rest",
            f"the filler list in `hyde_generate_hypothesis` includes 'is', 'at' and 'the', "
            f"so the topic becomes {result['topic']!r}. That phrase then appears in the "
            "hypothesis alongside the quoted query, doubling the weight of three terms "
            "while the rest of the query is diluted",
        ),
        practice.Check(
            "CONTROL: strip the boilerplate and HyDE returns to parity",
            all([result["lean"] == result["direct"], result["generated"]]),
            f"a hypothesis that is the query plus the extracted topic and nothing else "
            f"scores {result['lean']} of {result['queries']}, matching direct search. "
            f"Lesson 06's generator over HyDE's own top-{TOP_K} returns "
            f"{result['generated'][:40]!r}... The idea needs a generator that knows the "
            "corpus vocabulary, and a template cannot supply one",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
