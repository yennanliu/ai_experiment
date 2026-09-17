"""Exercise 2 — nothing is duplicated, and the recoverable waste is the word "the".

    Implement semantic deduplication for retrieved context. If two retrieved
    documents are more than 80% similar (by word overlap or cosine similarity
    of their embeddings), keep only the higher-scored one. Measure how much
    token budget this recovers.

Reading of the exercise: "word overlap" is Jaccard over the lowercased word
sets, "cosine similarity of their embeddings" is cosine over word-count
vectors, and both are computed over the documents `ContextEngine.assemble`
actually retrieves -- the knowledge base filtered at its own 0.05 relevance
threshold -- for the five queries covering the five intents.

**ANSWER: it recovers 0 tokens, on every query and under both similarity
measures.** The highest Jaccard between any two of the ten knowledge-base
sentences is 0.176 and the highest cosine is 0.334. The threshold is 0.80.
Nothing in the corpus is within a factor of two of being a duplicate.

**FINDING: the retrieved set is 4 to 7 of the 10 documents, and most of them
match on a stopword.** For "how do I fix the failing test" six documents are
retrieved, five of them scoring exactly 0.143 -- one shared word out of seven
query words -- and in all five cases that word is "the".

**MECHANISM: `score_relevance` divides by `len(query_words)` and removes
nothing.** No stopword list, no document-length normalisation, and the
threshold is 0.05, so for any query of twenty words or fewer a single shared
token is enough to be retrieved.

**FINDING: the recoverable budget is 378 tokens, and deduplication is the wrong
tool for it.** Dropping matches whose overlap is only stopwords takes the
retrieved blocks from 80, 53, 94, 96 and 83 tokens to 13, 0, 15, 0 and 0 --
93% of the retrieved context, against deduplication's 0%.

**CONTROL: deduplication does work when there is something to deduplicate.**
Adding each retrieved document back one word short -- what an overlapping
chunker produces -- puts every pair above the threshold, and the same filter
recovers the added copies.

Structure: `jaccard` and `cosine_counts` are the two similarity measures the
exercise offers, `dedup` is the filter, and `STOPWORDS` is the control.
"""

from __future__ import annotations

import collections
import itertools
import math

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "05-context-engineering"
QUERIES = ["how do I fix the failing test", "schedule a meeting for tuesday",
           "what is the api rate limit", "query the database for user stats",
           "send an email to the team"]
THRESHOLD, RELEVANCE = 0.80, 0.05
STOPWORDS = set("a an the and or of to in is are was for on with do does did i you it its "
                "how what why when where that this".split())


def jaccard(a, b):
    left, right = set(a.lower().split()), set(b.lower().split())
    return len(left & right) / len(left | right) if left else 0.0


def cosine_counts(a, b):
    left, right = collections.Counter(a.lower().split()), collections.Counter(b.lower().split())
    dot = sum(left[w] * right[w] for w in left.keys() & right.keys())
    norms = math.hypot(*left.values()) * math.hypot(*right.values())
    return dot / norms if norms else 0.0


def retrieved(ref, engine, query):
    """Exactly what `assemble` keeps: the knowledge base above the 0.05 threshold."""
    scored = zip(engine.knowledge_base, ref.score_relevance(query, engine.knowledge_base))
    return [(doc, score) for doc, score in scored if score >= RELEVANCE]


def dedup(ref, docs, similar, threshold=THRESHOLD):
    """Keep the higher-scored document of any pair above the threshold."""
    kept = []
    for doc, score in sorted(docs, key=lambda p: -p[1]):
        if all(similar(doc, other) <= threshold for other, _ in kept):
            kept.append((doc, score))
    return kept


def tokens(ref, docs):
    return ref.count_tokens("\n".join(doc for doc, _ in docs))


def stopword_only(query, doc):
    """Is the whole query-document overlap made of stopwords?"""
    shared = set(query.lower().split()) & set(doc.lower().split())
    return bool(shared) and shared <= STOPWORDS


def paraphrase(doc):
    """The control: the same sentence one word short, as an overlapping chunker emits."""
    return " ".join(doc.split()[:-1])


def measure(ref, engine, query):
    docs = retrieved(ref, engine, query)
    inflated = docs + [(paraphrase(doc), score) for doc, score in docs]
    pairs = list(itertools.combinations(docs, 2))
    pruned = [(doc, score) for doc, score in docs if not stopword_only(query, doc)]
    total, swollen = tokens(ref, docs), tokens(ref, inflated)
    return {"retrieved": len(docs), "tokens": total, "stopword_tokens": tokens(ref, pruned),
        "max_jaccard": round(max((jaccard(a, b) for (a, _), (b, _) in pairs), default=0), 3),
        "max_cosine": round(max((cosine_counts(a, b) for (a, _), (b, _) in pairs), default=0), 3),
        "recovered_jaccard": total - tokens(ref, dedup(ref, docs, jaccard)),
        "recovered_cosine": total - tokens(ref, dedup(ref, docs, cosine_counts)),
        "stopword_only": sum(stopword_only(query, doc) for doc, _ in docs),
        "inflated": swollen,
        "inflated_recovered": swollen - tokens(ref, dedup(ref, inflated, jaccard))}


COLUMNS = ("retrieved", "tokens", "recovered_jaccard", "recovered_cosine", "stopword_tokens",
           "stopword_only", "inflated", "inflated_recovered")


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    engine = ref.ContextEngine()
    rows = [measure(ref, engine, q) for q in QUERIES]
    return {
        **{key: [row[key] for row in rows] for key in COLUMNS},
        "max_jaccard": max(row["max_jaccard"] for row in rows),
        "max_cosine": max(row["max_cosine"] for row in rows),
        "corpus": len(engine.knowledge_base),
        "shared_words": [sorted(set(QUERIES[0].lower().split()) & set(doc.lower().split()))
                         for doc, _ in retrieved(ref, engine, QUERIES[0])],
    }


def verify(result):
    recovered = sum(result["stopword_tokens"])
    total = sum(result["tokens"])
    return [
        practice.Check(
            "ANSWER: deduplication recovers 0 tokens under both measures",
            all([set(result["recovered_jaccard"]) == {0},
                 set(result["recovered_cosine"]) == {0},
                 result["max_jaccard"] < 0.2, result["max_cosine"] < THRESHOLD]),
            f"the highest Jaccard between any two retrieved documents is "
            f"{result['max_jaccard']} and the highest count-cosine is "
            f"{result['max_cosine']}, against a threshold of {THRESHOLD}. Recovered per "
            f"query: {result['recovered_jaccard']}. There is nothing to deduplicate",
        ),
        practice.Check(
            "FINDING: most of what is retrieved matches on a stopword",
            all([min(result["retrieved"]) >= 4, result["stopword_only"][0] == 5,
                 result["shared_words"][0] == ["the"]]),
            f"{result['retrieved']} of {result['corpus']} documents are retrieved per "
            f"query, {result['stopword_only']} of them overlapping only in stopwords. For "
            f"query 1 the shared words are {result['shared_words']}: five documents "
            "retrieved on the word 'the'",
        ),
        practice.Check(
            "MECHANISM: the relevance score divides by the query length and removes nothing",
            all([result["retrieved"][0] == 6, RELEVANCE == 0.05]),
            "`score_relevance` is len(query_words & doc_words) / len(query_words): no "
            f"stopword list, no length normalisation, threshold {RELEVANCE}. One shared "
            "token out of a twenty-word query scores 0.05, so a single common word is "
            "enough to be retrieved",
        ),
        practice.Check(
            "FINDING: the recoverable budget is 378 tokens, not 0",
            all([total - recovered == 378, recovered < total // 10]),
            f"retrieved tokens per query {result['tokens']} become "
            f"{result['stopword_tokens']} once stopword-only matches are dropped: "
            f"{total} -> {recovered}, recovering {total - recovered} tokens, "
            f"{(total - recovered) / total:.0%} of the retrieved context. Deduplication "
            "recovers 0 of the same budget",
        ),
        practice.Check(
            "CONTROL: dedup works when there is something to deduplicate",
            all([min(result["inflated_recovered"]) > 0,
                 all(i > t for i, t in zip(result["inflated"], result["tokens"]))]),
            f"adding each retrieved document back one word short -- what an overlapping "
            f"chunker produces -- takes the block from {result['tokens']} to "
            f"{result['inflated']} tokens, and the same dedup recovers "
            f"{result['inflated_recovered']}. The filter is correct; the corpus was clean",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
