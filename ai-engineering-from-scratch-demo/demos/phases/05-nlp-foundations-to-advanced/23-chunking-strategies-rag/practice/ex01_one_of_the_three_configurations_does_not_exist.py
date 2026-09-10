"""Exercise 1 — one of the three configurations does not exist.

    **Easy.** Chunk one 20-page document with fixed(512, 0), recursive(512, 0),
    and recursive(512, 100). Compare chunk counts and boundary quality.

Reading of the exercise: `chunk_recursive(text, size, seps)` has no overlap
parameter. Passing 100 as the third positional argument binds it to `seps`, and
the function raises `TypeError: 'int' object is not iterable` on the first loop.
Two of the three configurations exist.

`recursive(512, 0)` does not respect 512 either. The function is not recursive:
it splits on the first separator that appears anywhere in the text and never
descends into a part that is still too long. A document whose paragraphs all fit
is chunked correctly; add one 893-character paragraph and that paragraph comes
back whole at **every** requested size -- 512, 300 and 200 all return the same
893-character chunk. `size` is an upper bound only when the input already
satisfies it.

Boundary quality separates the two that do run, and completely. Over the same
document `chunk_fixed(512, 0)` produces **0 of 4** chunks that begin at a
sentence start and end at sentence-final punctuation; `chunk_recursive(512)`
produces **3 of 3**. That is the trade the exercise is pointing at, and it is
visible in the chunk sizes: fixed returns 512, 512, 512, 75 -- three cuts at
exactly the limit, none of them at a boundary -- against 440, 274, 893.

Overlap only exists on the arm that does not need it. At 200 characters
`chunk_fixed(size, 100)` returns 17 chunks against 9 -- re-reading half of every
chunk to repair cuts it made itself. The recursive
chunker cuts at separators, so there is nothing to repair, which may be why no
overlap parameter was written for it.

Structure: `DOC` is `main()`'s own contract; `STRESS` appends one oversized
paragraph; `profile` reports count, longest chunk, over-limit chunks and clean
boundaries for one chunker; `attempt` calls the configuration the exercise names
and captures what happens.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "23-chunking-strategies-rag"

SIZES = (512, 300, 200)
DOC = """Chapter 1. Introduction. This contract is between Acme Corp and Beta Inc. The parties agree to the following terms.

Chapter 2. Payment. Acme will pay Beta thirty thousand dollars on the first of each month. Late payments incur a five percent fee.

Chapter 3. Termination. Either party may terminate this agreement with ninety days written notice. Termination for cause requires only thirty days notice. Breach of payment constitutes cause.

Chapter 4. Confidentiality. Both parties agree to keep trade secrets confidential. This obligation survives termination of the agreement.

Chapter 5. Miscellaneous. This agreement is governed by the laws of the State of California. Disputes shall be resolved by arbitration."""
OVERSIZED = "Chapter 6. Schedule. " + " ".join(
    f"Milestone {n} is due in month {n} and requires written sign off from both parties."
    for n in range(1, 12))
STRESS = DOC + "\n\n" + OVERSIZED


def clean_boundary(chunk):
    """Starts where a sentence starts and ends where one ends."""
    return bool(chunk) and chunk[0].isupper() and chunk[-1] in ".!?"


def profile(chunks, size):
    """Count, longest chunk, chunks over the requested size, and clean boundaries."""
    return {
        "chunks": len(chunks),
        "longest": max(len(c) for c in chunks),
        "over": [len(c) for c in chunks if len(c) > size],
        "clean": sum(1 for c in chunks if clean_boundary(c)),
        "sizes": [len(c) for c in chunks],
    }


def attempt(call):
    """The exception a configuration raises, or None if it runs."""
    try:
        call()
    except TypeError as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {
        "overlap_arm": attempt(lambda: ref.chunk_recursive(STRESS, 512, 100)),
        "fixed_arm": attempt(lambda: ref.chunk_fixed(STRESS, 512, 100)),
        "oversized": len(OVERSIZED),
        "recursive": {n: profile(ref.chunk_recursive(STRESS, n), n) for n in SIZES},
        "fixed": {n: profile(ref.chunk_fixed(STRESS, n), n) for n in SIZES},
        "overlapped": {n: profile(ref.chunk_fixed(STRESS, n, 100), n) for n in SIZES},
        "tidy_doc": profile(ref.chunk_recursive(DOC, 512), 512),
        "separators": ref.chunk_recursive.__defaults__[0],
    }


def verify(result):
    top, rec, fix = SIZES[0], result["recursive"], result["fixed"]
    return [
        practice.Check(
            "ANSWER: `recursive(512, 100)` is not a call this code supports",
            result["overlap_arm"] is not None and result["fixed_arm"] is None,
            f"`chunk_recursive(text, size, seps)` has no overlap parameter, so 100 binds to "
            f"`seps` and the first loop raises {result['overlap_arm']!r}. The same third argument "
            "on `chunk_fixed` is the overlap and runs fine. Two of the three configurations exist",
        ),
        practice.Check(
            "MECHANISM: and `recursive(512, 0)` does not respect 512 either",
            all(rec[n]["over"] for n in SIZES),
            f"the function splits on the first separator present in {list(result['separators'])} "
            f"and never descends into a part that is still too long, so one "
            f"{result['oversized']}-character paragraph comes back whole at every requested size: "
            f"longest chunk {[rec[n]['longest'] for n in SIZES]} for sizes {list(SIZES)}",
        ),
        practice.Check(
            "MECHANISM: `size` is a bound only when the document already satisfies it",
            not result["tidy_doc"]["over"],
            f"on `main()`'s own contract, whose paragraphs all fit, the same call returns "
            f"{result['tidy_doc']['chunks']} chunks with a longest of "
            f"{result['tidy_doc']['longest']} and nothing over the limit. The failure is invisible "
            "until a document has one long paragraph in it",
        ),
        practice.Check(
            "FINDING: boundary quality separates the two arms completely",
            fix[top]["clean"] == 0 and rec[top]["clean"] == rec[top]["chunks"],
            f"at {top} characters, fixed produces {fix[top]['clean']} of {fix[top]['chunks']} "
            f"chunks that start and end at a sentence, recursive {rec[top]['clean']} of "
            f"{rec[top]['chunks']}. The sizes say why: {fix[top]['sizes']} against "
            f"{rec[top]['sizes']} -- three cuts at exactly the limit, none at a boundary",
        ),
        practice.Check(
            "FINDING: the chunk counts trade against that, in the direction you would expect",
            fix[SIZES[-1]]["chunks"] > rec[SIZES[-1]]["chunks"],
            f"fixed returns {[fix[n]['chunks'] for n in SIZES]} chunks across {list(SIZES)} and "
            f"recursive {[rec[n]['chunks'] for n in SIZES]}: recursive makes fewer, larger, "
            "cleaner chunks and pays for it by not honouring the size at all",
        ),
        practice.Check(
            "CONTROL: overlap only exists on the arm that needs it",
            result["overlapped"][SIZES[-1]]["chunks"] > fix[SIZES[-1]]["chunks"],
            f"at {SIZES[-1]} characters `chunk_fixed(size, 100)` returns "
            f"{result['overlapped'][SIZES[-1]]['chunks']} chunks against "
            f"{fix[SIZES[-1]]['chunks']} at zero overlap -- re-reading 100 characters per chunk to "
            "repair cuts it made itself. A separator-aligned chunker has nothing to repair, which "
            "is a reason not to write the parameter the exercise asks for",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
