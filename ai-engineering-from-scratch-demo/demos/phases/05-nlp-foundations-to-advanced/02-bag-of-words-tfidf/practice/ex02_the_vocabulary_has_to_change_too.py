"""Exercise 2 — the vocabulary has to change too.

    **Medium.** Add `n-gram` support to `bag_of_words`. Parameter `n` produces
    counts over `n`-grams. Test that `n=2` on `["the", "cat", "sat"]` produces
    bigram counts for `["the cat", "cat sat"]`.

Reading of the exercise: the named test passes on a change that breaks the
pipeline, because `bag_of_words` is not where the vocabulary is decided.
`bag_of_words(docs, vocab)` takes its columns from `build_vocab`, and its only
handling of an unknown token is `if token in vocab` -- a silent drop. Give it
bigrams while the vocabulary is still unigrams and every lookup misses, so the
matrix comes back all zeros with no error, and `tfidf`, `l2_normalize` and
`cosine_similarity` all propagate that: a document scores 0.0 against itself,
contradicting the property exercise 1 was told to verify. So "add n-gram
support to `bag_of_words`" is one function short; `build_vocab` has to see the
same n or nothing works, and nothing says so.

The second half is what n-grams cost. Counting n-grams instead of unigrams
(rather than as well as) removes every single-word match, and on the lesson's
own three documents the similarity between d1 and d2 falls from 0.3078 to
exactly 0.0 at n=2 -- the two documents share `the` and `on` and no bigram at
all. Sweeping n to 5 on the same corpus, the count of n-grams appearing in more
than one document goes 5, 4, 2, 1, 0: past n=4 nothing is shared, every
off-diagonal similarity is 0, and the representation ranks nothing.

Structure: `grams` is the sliding window, `vectors` runs the lesson's own four
functions over an already-n-grammed corpus, and `matrix` is the similarity
matrix it produces. `starved` is the failure arm -- n-grammed documents against
the unigram vocabulary the lesson's `build_vocab` would give.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "02-bag-of-words-tfidf"

RAW = ("The cat sat on the mat.", "The dog sat on the mat.", "The cat ran across the room.")
PROBE = ["the", "cat", "sat"]
SWEEP = (1, 2, 3, 4, 5)

grams = lambda doc, n: [" ".join(doc[i:i + n]) for i in range(len(doc) - n + 1)]  # noqa: E731
shared = lambda ref, dfs: sum(1 for d in dfs if d > 1)                           # noqa: E731


def vectors(ref, docs) -> tuple:
    """The lesson's own pipeline over an already-n-grammed corpus."""
    vocab = ref.build_vocab(docs)
    bow = ref.bag_of_words(docs, vocab)
    return ref.l2_normalize(ref.tfidf(bow)), vocab, ref.document_frequency(bow)


def matrix(ref, rows) -> list:
    return [[round(ref.cosine_similarity(a, b), 4) for b in rows] for a in rows]


def starved(ref, docs, n) -> dict:
    """n-grammed documents against the unigram vocabulary `build_vocab` gives."""
    bow = ref.bag_of_words([grams(doc, n) for doc in docs], ref.build_vocab(docs))
    unit = ref.l2_normalize(ref.tfidf(bow))
    return {"bow": bow, "self": ref.cosine_similarity(unit[0], unit[0]),
            "counts": sum(sum(row) for row in bow)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    docs = [ref.tokenize(text) for text in RAW]
    arms = {n: vectors(ref, [grams(doc, n) for doc in docs]) for n in SWEEP}
    return {
        "probe": grams(PROBE, 2), "identity": grams(docs[0], 1) == docs[0],
        "starved": starved(ref, docs, 2),
        "sims": {n: matrix(ref, arms[n][0]) for n in (1, 2)},
        "vocab": {n: len(arms[n][1]) for n in SWEEP},
        "shared": {n: shared(ref, arms[n][2]) for n in SWEEP},
        "lonely": grams(["hello"], 2),
        "one_token": ref.bag_of_words([grams(["hello"], 2)], arms[2][1]),
    }


def verify(result):
    starve, sims, share = result["starved"], result["sims"], result["shared"]
    return [
        practice.Check(
            "ANSWER: the named test passes -- n=2 on ['the','cat','sat'] gives the two bigrams",
            result["probe"] == ["the cat", "cat sat"] and result["identity"],
            f"a length-n sliding window gives {result['probe']}, and n=1 reproduces the lesson's own "
            f"token list unchanged, so the unigram path is a special case rather than a second "
            f"branch. On the lesson's three documents the bigram vocabulary is "
            f"{result['vocab'][2]} entries against {result['vocab'][1]} unigrams"),
        practice.Check(
            "MECHANISM: changing bag_of_words alone returns an all-zero matrix, silently",
            starve["counts"] == 0 and starve["self"] == 0,
            f"`bag_of_words(docs, vocab)` takes its columns from `build_vocab`, and its only handling "
            f"of an unknown token is `if token in vocab`. Feed it bigrams against the unigram "
            f"vocabulary and all {starve['counts']} counts land nowhere: the matrix is "
            f"{starve['bow'][0]} for every row, with no error raised"),
        practice.Check(
            "FINDING: the zeros survive the whole pipeline and break exercise 1's property",
            starve["self"] == 0,
            f"`tfidf` divides by a row sum of 0 and returns zeros, `l2_normalize` maps a zero norm "
            f"to zeros, and `cosine_similarity` returns {starve['self']}. A document scores "
            f"{starve['self']} against itself -- the exact case exercise 1 says must be 1.0 -- so a "
            f"caller checking that property is the only thing that would catch this"),
        practice.Check(
            "FINDING: counting n-grams instead of unigrams deletes every single-word match",
            sims[1][1][2] > 0.3 and sims[2][1][2] == 0.0,
            f"d1 and d2 share `the` and `on` and no bigram at all, so their similarity falls from "
            f"{sims[1][1][2]} at n=1 to {sims[2][1][2]} at n=2. The full matrices are {sims[1]} and "
            f"{sims[2]}: every off-diagonal entry drops, because the exercise's wording counts "
            f"n-grams rather than adding them to the unigrams"),
        practice.Check(
            "FINDING: past n=4 nothing is shared, and the representation ranks nothing",
            [share[n] for n in SWEEP] == [5, 4, 2, 1, 0],
            f"n-grams appearing in more than one of the three documents, for n = {list(SWEEP)}: "
            f"{[share[n] for n in SWEEP]}. Vocabulary size {[result['vocab'][n] for n in SWEEP]} "
            f"peaks at n=2 and then falls, but the useful part falls faster -- at n=5 every "
            f"off-diagonal similarity is 0 and the corpus has no structure left to measure"),
        practice.Check(
            "CONTROL: a document shorter than n produces no n-grams and no error either",
            result["lonely"] == [] and sum(result["one_token"][0]) == 0,
            f"a one-token document windowed at n=2 gives {result['lonely']}, so its row is all "
            f"zeros -- the same degenerate vector as the vocabulary mismatch above, reached a "
            f"different way. Any n-gram implementation has to decide what a document shorter than n "
            f"means, and `range(len(doc) - n + 1)` decides it is empty"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
