"""Exercise 5 — the boundaries are identical, and only the tokenisation moved.

    **Sentence-based chunking**: replace fixed-size chunking with
    `chunk_by_sentences`. Run the same queries and compare retrieval scores.
    Does respecting sentence boundaries improve the results?

Reading of the exercise: both chunkers are run at the lesson's own default size,
200 tokens, over `SAMPLE_DOCUMENTS`, and the same five queries are scored
through the same `SimpleEmbedder` pipeline. "Improve" is read as the top-1
cosine score per query, which is what the lesson's `search_with_scores`
reports.

**ANSWER: it makes three of the five queries very slightly worse, improves
none, and the boundaries have nothing to do with it.** The longest document is 102 words and
the limit is 200, so *both* chunkers emit exactly 5 chunks -- one whole document
each. The boundaries are identical. Top-1 goes 0.3539 -> 0.3463, 0.4389 ->
0.4368 and 0.4366 -> 0.4354; the refund query ties at 0.4714 and the
out-of-vocabulary query is 0.0 either way.

**MECHANISM: `chunk_by_sentences` splits on "." and reassembles.** So it
shatters every decimal, version number and hostname in the corpus:

    99.9%  ->  "99."  "9%"          1.3.  ->  "1."  "3."
    0.1%   ->  "0."   "1%"          status.acme.com  ->  "status." "acme." "com"

Six tokens are destroyed and eleven junk tokens are created, taking the
vocabulary from 270 to 275 and the word count from 471 to 479. The chunker does
not preserve its input.

**FINDING: the entire measured difference is that tokenisation.** With the chunk
boundaries identical, the only thing that can move a score is the vocabulary,
and the queries that lose are exactly the ones whose target chunk contains a
mangled token -- the SLA document holds "99.9%" and "99.99%", the rate-limit
document holds "1000".

**FINDING: the sentence splitter also cannot enforce its own limit.** It
appends a sentence before testing the budget only when the chunk is already
non-empty, so a single sentence longer than `max_chunk_tokens` becomes one
oversized chunk. No sample document triggers it, which is why the bug is
invisible here.

**ANSWER: respecting sentence boundaries neither helps nor hurts, because it
was never tested.** At this corpus size the comparison the exercise sets up has
no chunking difference in it at all.

Structure: `by_sentence` and `by_fixed` build the two indexes over a shared
pipeline, and `tokens` measures what the splitter did to the vocabulary.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "04-embeddings"
QUERIES = ["refund policy for enterprise customers", "how do I reset my password",
           "what is the SLA uptime guarantee", "data retention and deletion",
           "api rate limits"]
LIMIT = 200


def build(ref, chunks):
    """The lesson's own indexing path, given a chunk list."""
    engine = ref.SemanticSearchEngine()
    engine.embedder.fit(chunks)
    for chunk in chunks:
        engine.index.add(engine.embedder.embed(chunk), chunk, {})
    return engine


def by_sentence(ref):
    return [c for d in ref.SAMPLE_DOCUMENTS for c in ref.chunk_by_sentences(d, LIMIT)]


def by_fixed(ref):
    return [c for d in ref.SAMPLE_DOCUMENTS for c in ref.chunk_text(d, LIMIT, 50)]


def top1(engine):
    return [round(engine.search(q, top_k=1)[0]["score"], 4) for q in QUERIES]


def tokens(chunks):
    return {w for c in chunks for w in c.lower().split()}


def long_sentence(ref):
    """Does the splitter enforce its own budget? One sentence over the limit."""
    text = " ".join(["word"] * 300) + "."
    return [len(c.split()) for c in ref.chunk_by_sentences(text, LIMIT)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "embeddings")
    sentence, fixed = by_sentence(ref), by_fixed(ref)
    sentence_vocab, fixed_vocab = tokens(sentence), tokens(fixed)
    return {
        "counts": (len(sentence), len(fixed)),
        "words": (sum(len(c.split()) for c in sentence),
                  sum(len(d.split()) for d in ref.SAMPLE_DOCUMENTS)),
        "vocab": (len(sentence_vocab), len(fixed_vocab)),
        "destroyed": sorted(fixed_vocab - sentence_vocab),
        "created": sorted(sentence_vocab - fixed_vocab)[:6],
        "created_count": len(sentence_vocab - fixed_vocab),
        "sentence_top1": top1(build(ref, sentence)),
        "fixed_top1": top1(build(ref, fixed)),
        "oversized": long_sentence(ref),
        "doc_words": [len(d.split()) for d in ref.SAMPLE_DOCUMENTS],
    }


def verify(result):
    sentence, fixed = result["sentence_top1"], result["fixed_top1"]
    worse = sum(a < b for a, b in zip(sentence, fixed))
    return [
        practice.Check(
            "ANSWER: both chunkers emit the same 5 whole-document chunks",
            all([result["counts"] == (5, 5), max(result["doc_words"]) < LIMIT]),
            f"the longest document is {max(result['doc_words'])} words and the limit is "
            f"{LIMIT}, so both chunkers produce {result['counts'][0]} chunks -- one whole "
            "document each. Whatever the comparison measures, it is not a difference in "
            "chunk boundaries, because there is none",
        ),
        practice.Check(
            "ANSWER: three of five queries get very slightly worse, none improves",
            all([worse == 3, sentence[0] == fixed[0], max(fixed) == fixed[0]]),
            f"top-1 per query, sentence {sentence} against fixed {fixed}: "
            f"{worse} lose, 1 ties at {sentence[0]} and the query that embeds to the zero "
            "vector is 0.0 either way. Small, consistent, and in the wrong direction for "
            "the exercise's question",
        ),
        practice.Check(
            "MECHANISM: the splitter shatters every decimal, version and hostname",
            all([len(result["destroyed"]) == 6, result["created_count"] == 11]),
            f"`text.split('.')` destroys {len(result['destroyed'])} tokens -- "
            f"{result['destroyed']} -- and creates {result['created_count']}, starting "
            f"{result['created']}. The vocabulary goes {result['vocab'][1]} -> "
            f"{result['vocab'][0]}",
        ),
        practice.Check(
            "FINDING: the chunker does not preserve its input",
            result["words"][0] > result["words"][1],
            f"the sentence chunks hold {result['words'][0]} whitespace words against the "
            f"documents' {result['words'][1]}: the split-and-reappend adds "
            f"{result['words'][0] - result['words'][1]}. With the boundaries identical, "
            "the tokenisation is the entire measured difference between the two arms",
        ),
        practice.Check(
            "FINDING: the splitter cannot enforce its own budget",
            all([result["oversized"] == [300], result["oversized"][0] > LIMIT]),
            f"the length test only fires when the current chunk is already non-empty, so a "
            f"single sentence of 300 words becomes one chunk of {result['oversized'][0]} "
            f"against a max_chunk_tokens of {LIMIT}. No sample document is long enough to "
            "trigger it, which is why the comparison above never sees it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
