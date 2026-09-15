"""Exercise 2 — the one duplicate MinHash removes is the one that needed a hash set.

    **Medium:** Implement exact deduplication using SHA-256 hashes alongside the
    MinHash near-deduplication. Compare the number of duplicates caught by each
    method on a web-scraped corpus.

Reading of the exercise: "a web-scraped corpus" is the lesson's own
`generate_sample_corpus`, which plants two duplicates on purpose and names them
`near_dup_1` and `near_dup_2` -- a corpus built for this comparison is a better
test of it than one downloaded. Both methods run on the same 13 documents that
survive `clean_text` and `quality_filter`, which is where `run_pipeline` puts
deduplication.

**ANSWER: each method catches exactly one document, and it is the same
document.** `near_dup_2` is byte-identical to `base_docs[1]` -- Jaccard 1.0000 --
so SHA-256 finds it in one line and MinHash finds it with 128 hashes and 16
bands. The counts the exercise asks you to compare are 1 and 1.

**FINDING: the document planted as the near-duplicate is not caught, at any
threshold.** `near_dup_1` is a rewrite of `base_docs[0]` and scores Jaccard
**0.2703** against it. The pipeline's threshold is 0.8; dropping it to 0.3 still
removes only the exact pair. The whole apparatus removes one document, and it is
the one that needed none of it.

**MECHANISM: `get_shingles`' `k=5` is the knob, and 5 is a high setting.** The
same pair scores 0.7778 on words, 0.5543 on bigrams, 0.4356 on trigrams, and
0.2703 on 5-grams. A rewrite that changes one word destroys five shingles, so
`k=5` plus `threshold=0.8` asks for near-verbatim text. The two documents share
49 of 63 word types and the detector calls them unrelated.

**FINDING: neither method contains the other.** A document under `k` words has
no shingles at all -- `get_shingles("a b c")` is the empty set -- so four short
documents containing two identical pairs are deduplicated to four, while SHA-256
catches both pairs. In the other direction, a copy differing only in whitespace
and case has a different SHA-256 and identical shingles, so MinHash removes it
and the exact method does not. The exercise asks which catches more; on this
corpus the answer is that their intersection is all either one catches.

Structure: `exact` is the SHA-256 pass; `jaccard` and `shingle_curve` measure
what MinHash is being asked to see.
"""

from __future__ import annotations

import hashlib

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "03-data-pipelines"
THRESHOLDS = (0.8, 0.5, 0.3)
SHORT = ["a b c", "a b c", "x y", "x y"]


def exact(documents):
    """SHA-256 deduplication: the count removed, and the surviving documents."""
    seen, kept = set(), []
    for document in documents:
        digest = hashlib.sha256(document.encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            kept.append(document)
    return len(documents) - len(kept), kept


def jaccard(left, right):
    return len(left & right) / len(left | right) if left | right else 0.0


def pipeline_input(ref):
    """The documents `run_pipeline` hands to deduplicate: cleaned, quality-filtered."""
    cleaned = [ref.clean_text(doc) for doc in ref.generate_sample_corpus()]
    return [doc for doc in cleaned if ref.quality_filter(doc)]


def ranked_pairs(ref, documents):
    """Every document pair by Jaccard on the lesson's own 5-word shingles."""
    shingles = [ref.get_shingles(doc) for doc in documents]
    pairs = [(jaccard(shingles[i], shingles[j]), i, j)
             for i in range(len(documents)) for j in range(i + 1, len(documents))]
    return sorted(pairs, reverse=True)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    documents = pipeline_input(ref)
    pairs = ranked_pairs(ref, documents)
    variant = [documents[0], "  ".join(documents[0].split())]
    return {
        "documents": len(documents),
        "exact": exact(documents)[0],
        "minhash": {t: ref.deduplicate(documents, threshold=t)[1] for t in THRESHOLDS},
        "top": pairs[:2],
        "curve": {k: jaccard(ref.get_shingles(documents[pairs[1][1]], k),
                             ref.get_shingles(documents[pairs[1][2]], k))
                  for k in (1, 2, 3, 5)},
        "short_shingles": ref.get_shingles("a b c"),
        "short_minhash": ref.deduplicate(SHORT)[1],
        "short_exact": exact(SHORT)[0],
        "variant_exact": exact(variant)[0],
        "variant_minhash": ref.deduplicate(variant)[1],
        "asymmetric": (not ref.get_shingles("a b c") and ref.deduplicate(SHORT)[1] == 0
                       and exact(SHORT)[0] == 2),
    }


def verify(result):
    (best, _, _), (second, i, j) = result["top"]
    minhash, curve = result["minhash"], result["curve"]
    return [
        practice.Check(
            "ANSWER: both methods catch one document, and it is the same document",
            result["exact"] == minhash[0.8] == 1 and best == 1.0,
            f"of the {result['documents']} documents run_pipeline hands to deduplicate, SHA-256 "
            f"removes {result['exact']} and MinHash at threshold 0.8 removes {minhash[0.8]}. It "
            f"is near_dup_2, byte-identical to base_docs[1] at Jaccard {best:.4f} -- one line of "
            "hashing finds it, and so do 128 hash functions across 16 bands",
        ),
        practice.Check(
            "FINDING: the document planted as the near-duplicate is missed at every threshold",
            second < 0.3 and set(minhash.values()) == {1},
            f"near_dup_1 is a rewrite of base_docs[0] and scores {second:.4f} against it, so it "
            f"sits far below the pipeline's 0.8. Lowering the threshold does not help: "
            + ", ".join(f"{t} -> {n} removed" for t, n in minhash.items())
            + ". The near-duplicate detector removes one document from this corpus and it is "
            "the exact one; the document the corpus was built around survives",
        ),
        practice.Check(
            "MECHANISM: k=5 is the knob, and the same pair is a duplicate on words",
            curve[1] > 0.7 > curve[3] > curve[5] and curve[5] == second,
            "the same pair scored on shorter shingles: "
            + ", ".join(f"k={k} {v:.4f}" for k, v in curve.items())
            + ". A rewrite that changes one word destroys five 5-grams, so get_shingles' k=5 "
            "with threshold=0.8 is asking for near-verbatim text. On word types the two "
            f"documents agree {curve[1]:.0%}, and the configuration calls them unrelated",
        ),
        practice.Check(
            "FINDING: neither method contains the other",
            result["asymmetric"] and result["variant_minhash"] > result["variant_exact"],
            f"a document under k words has no shingles at all -- get_shingles('a b c') is "
            f"{result['short_shingles']!r} -- so four short documents holding two identical "
            f"pairs are deduplicated to four by MinHash and to two by SHA-256. In the other "
            f"direction a copy differing only in whitespace has a different digest and identical "
            f"shingles: exact removes {result['variant_exact']}, MinHash removes "
            f"{result['variant_minhash']}. The exercise asks which catches more; on this corpus "
            "their intersection is everything either one catches",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
