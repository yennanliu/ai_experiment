"""Exercise 1 — TF-IDF and binary bag-of-words return the same top hit on all five queries.

    Replace the TF-IDF embeddings with a simple bag-of-words approach (binary:
    1 if word present, 0 if not). Compare retrieval quality on the sample
    documents. TF-IDF should outperform because it weights rare words higher.

Reading of the exercise: only the embedding changes -- the vocabulary, the
cosine, the search and the pipeline are the lesson's own -- and "retrieval
quality" is scored two ways, by the rank-1 chunk and by the top-3 set, since
the exercise does not say which.

**ANSWER: they agree on every top hit.** Five queries, five identical rank-1
chunks. The top-3 *sets* differ on 2 of the 5, at ranks 2 and 3, which no
downstream step reads differently. The exercise's prediction is not wrong here;
it is unmeasurable, because the two embeddings disagree only about chunks that
are already irrelevant.

**MECHANISM: five documents cannot spread the IDF.** `compute_idf` is
`log((n+1)/(doc_count+1)) + 1` over n = 5 chunks, so the whole vocabulary of
270 words lies between 1.000 and 2.099 -- a factor of 2.1 between the rarest
word in the corpus and the commonest, and both arms score 3 of 5 on rank-1
relevance. "Weights rare words higher" is true and
the weight is at most doubled.

**FINDING: TF is the other half of TF-IDF, and it cancels.** `compute_tf`
divides by the chunk's word count, and at the lesson's default chunk size every
chunk is a whole document of 91 to 102 words. So the TF denominators differ by
at most 12%, and the term that was supposed to distinguish the embeddings is a
near-constant scale factor on both.

**FINDING: one of the five queries scores 0.0 against every chunk**, under both
embeddings. "how do I reset my password" shares no vocabulary word with the
corpus, so the ranking is `search`'s insertion order and the two arms agree for
the one reason that proves nothing.

**CONTROL: make the corpus able to tell them apart.** Splitting to 11 chunks
widens the IDF band to 1.087-2.792, and the arms then disagree at rank 1 on 2 of
the 5 queries. Rank-1 relevance becomes TF-IDF 3, binary 1. The exercise's claim
is correct; five whole-document chunks cannot show it.

Structure: `binary_embed` is the replacement, `ranks` runs both arms through the
lesson's own `search`, and `band` reports the IDF spread.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "06-rag"
QUERIES = ["what is the refund window for enterprise customers",
           "how do I reset my password", "what is the api rate limit",
           "how long is data retained", "what is the uptime SLA"]
ANSWERS = ["60-day", None, "100 requests", "90 days", "99.9%"]
TOP_K = 3


def binary_embed(text, vocab):
    """The exercise's replacement: 1 if the word is present, 0 if not."""
    words = set(text.lower().split())
    return [1.0 if word in words else 0.0 for word in vocab]


def ranks(ref, pipeline, query, embed_query, embed_chunks):
    hits = ref.search(embed_query(query), embed_chunks, TOP_K)
    return [index for index, _ in hits], [round(score, 3) for _, score in hits]


def both_arms(ref, pipeline):
    """The same queries through TF-IDF and through binary bag-of-words."""
    tfidf = (lambda q: ref.tfidf_embed(q, pipeline.vocab, pipeline.idf),
             [ref.tfidf_embed(c, pipeline.vocab, pipeline.idf) for c in pipeline.chunks])
    binary = (lambda q: binary_embed(q, pipeline.vocab),
              [binary_embed(c, pipeline.vocab) for c in pipeline.chunks])
    return {name: [ranks(ref, pipeline, q, embed, chunks) for q in QUERIES]
            for name, (embed, chunks) in (("tfidf", tfidf), ("binary", binary))}


def relevant(pipeline, indices, answer):
    return answer is not None and any(answer.lower() in pipeline.chunks[i].lower()
                                      for i in indices)


def band(pipeline):
    return round(min(pipeline.idf), 3), round(max(pipeline.idf), 3)


def compare(ref, chunk_size):
    pipeline = ref.RAGPipeline(chunk_size=chunk_size, overlap=50)
    count = pipeline.index(ref.SAMPLE_DOCUMENTS)
    arms = both_arms(ref, pipeline)
    return {
        "chunks": count, "vocab": len(pipeline.vocab), "band": band(pipeline),
        "top1_same": [a[0][0] == b[0][0] for a, b in zip(arms["tfidf"], arms["binary"])],
        "top3_same": [a[0] == b[0] for a, b in zip(arms["tfidf"], arms["binary"])],
        "quality": {name: sum(relevant(pipeline, r[0][:1], a)
                              for r, a in zip(rows, ANSWERS))
                    for name, rows in arms.items()},
        "zero_scored": [i for i, r in enumerate(arms["tfidf"]) if set(r[1]) == {0.0}],
        "words": [len(d.split()) for d in ref.SAMPLE_DOCUMENTS],
        "scores": arms["tfidf"][0][1],
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {"shipped": compare(ref, 200), "split": compare(ref, 100)}


def verify(result):
    shipped, split = result["shipped"], result["split"]
    spread = max(shipped["words"]) / min(shipped["words"])
    return [
        practice.Check(
            "ANSWER: the two embeddings return the same rank-1 chunk on all five queries",
            all([shipped["top1_same"] == [True] * len(QUERIES),
                 sum(shipped["top3_same"]) == 3]),
            f"rank-1 agreement per query: {shipped['top1_same']}. The top-3 sets differ on "
            f"{len(QUERIES) - sum(shipped['top3_same'])} of {len(QUERIES)}, at ranks 2 and "
            f"3 only. Both arms score {shipped['quality']['tfidf']} of {len(QUERIES)} on "
            "rank-1 relevance -- the prediction is unmeasurable, not wrong",
        ),
        practice.Check(
            "MECHANISM: five documents cannot spread the IDF",
            all([shipped["chunks"] == 5, shipped["band"] == (1.0, 2.099)]),
            f"`compute_idf` is log((n+1)/(doc_count+1)) + 1 over n = {shipped['chunks']} "
            f"chunks, so the entire {shipped['vocab']}-word vocabulary lies in "
            f"{shipped['band']}. 'Weights rare words higher' is true and the weight is at "
            "most doubled between the rarest word in the corpus and the commonest",
        ),
        practice.Check(
            "FINDING: TF is the other half, and at this chunk size it cancels",
            all([spread < 1.15, shipped["chunks"] == len(shipped["words"])]),
            f"`compute_tf` divides by the chunk's word count, and at the default chunk size "
            f"every chunk is a whole document: {shipped['words']} words, a spread of "
            f"{spread:.2f}x. The term meant to distinguish the embeddings is a "
            "near-constant scale factor on both of them",
        ),
        practice.Check(
            "FINDING: one query scores 0.0 against every chunk, under both embeddings",
            shipped["zero_scored"] == [1],
            f"query {shipped['zero_scored']} -- {QUERIES[1]!r} -- shares no vocabulary word "
            f"with the corpus, so its scores are {shipped['scores'] and [0.0]} and the "
            "ranking is `search`'s insertion order. The two arms agree there for the one "
            "reason that proves nothing about either",
        ),
        practice.Check(
            "CONTROL: give the corpus enough chunks and the arms separate",
            all([split["chunks"] > shipped["chunks"], split["band"][1] > shipped["band"][1],
                 sum(split["top1_same"]) < len(QUERIES),
                 split["quality"]["tfidf"] > split["quality"]["binary"]]),
            f"splitting to {split['chunks']} chunks widens the IDF band to {split['band']} "
            f"and rank-1 agreement drops to {split['top1_same']}. Rank-1 relevance is then "
            f"TF-IDF {split['quality']['tfidf']} against binary "
            f"{split['quality']['binary']}: the exercise's claim is correct, and five "
            "whole-document chunks cannot show it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
