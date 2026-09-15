"""Exercise 5 — the target is met by memorising the text, and unreachable without it.

    Train your BPE tokenizer on a larger corpus (download a Wikipedia article).
    Tune the number of merges to achieve a compression ratio within 10% of
    tiktoken on that same text. This forces you to understand the relationship
    between corpus size, merge count, and compression quality.

Reading of the exercise: "download a Wikipedia article" is read as *a larger
piece of encyclopedic English prose*, and the one used is the lesson's own
`docs/en.md` -- 22,839 bytes and 3,210 words, 59x the demo corpus, already on
disk, and hash-pinned by the coverage gate so the numbers below stay
reproducible. Nothing in the exercise depends on the text being Wikipedia's.
"on that same text" is read literally, because it is the load-bearing phrase:
the exercise specifies that the tokenizer be scored on the text it was trained
on, and that is what makes the target reachable.

**ANSWER: 1,070 merges.** `cl100k_base` compresses the article to 0.2532 tokens
per byte; 1,070 merges reach 0.2785, exactly 110.0% of it, and 1,069 do not.

**FINDING: the target is crossed in passing, not tuned to.** Double the merges
and the ratio is 76.0% of tiktoken's; quadruple them and it is 39.0%. On the
text it trained on, compression is monotone in the merge count all the way down
to one token, so "tune to within 10%" names a point on a slide, not an optimum.
There is nothing to tune.

**FINDING: it is monotone because the measurement is self-referential.** Hold
out the last 20% of the article and the slide stops: held-out compression
improves to 0.3360 at 1,600 merges and is then *identical* at 1,700, 1,800,
1,900, 2,000 and 2,100 -- bit for bit, not merely close. Every merge after
~1,600 encodes a string that occurs only in the 80% it trained on.

**FINDING: 10% is unreachable on held-out text at any merge count.** The floor
is **145.2%** of tiktoken. So the exercise's goal is trivially over-satisfied
under the measurement it specifies and impossible under the honest one -- which
is the relationship between corpus size and merge count it says this "forces you
to understand", visible only if you measure on text you did not train on.

Structure: `fit` is the lesson's own `train` with its printing swallowed;
`SELF` are the merge counts scored on the training text, `HELD` the ones scored
on the withheld fifth.
"""

from __future__ import annotations

import contextlib
import io

import tiktoken

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "01-tokenizers"
ENCODING, TOLERANCE, SPLIT = "cl100k_base", 0.10, 0.8
ANSWER = 1070
SELF = (ANSWER - 1, ANSWER, 2 * ANSWER)
HELD = (1600, 2100)


def fit(ref, text, num_merges):
    """The lesson's own `train`, with its per-merge printing swallowed."""
    tokenizer = ref.BPETokenizer()
    with contextlib.redirect_stdout(io.StringIO()):
        tokenizer.train(text, num_merges=num_merges)
    return tokenizer


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    article = parity.doc_text(PHASE, LESSON, "en")
    encoding = tiktoken.get_encoding(ENCODING)
    cut = int(len(article) * SPLIT)
    train_text, held_text = article[:cut], article[cut:]

    def reference_ratio(text):
        return len(encoding.encode(text)) / len(text.encode("utf-8"))

    return {
        "bytes": len(article.encode("utf-8")),
        "words": len(article.split()),
        "tiktoken": reference_ratio(article),
        "tiktoken_held": reference_ratio(held_text),
        "self": {n: ref.compression_ratio(fit(ref, article, n), article) for n in SELF},
        "held": {n: ref.compression_ratio(fit(ref, train_text, n), held_text)
                 for n in HELD},
        "held_bytes": len(held_text.encode("utf-8")),
    }


def verify(result):
    target = result["tiktoken"] * (1 + TOLERANCE)
    own, held = result["self"], result["held"]
    floor = held[HELD[0]] / result["tiktoken_held"]
    return [
        practice.Check(
            f"ANSWER: {ANSWER} merges reach 110.0% of {ENCODING} on the article",
            own[ANSWER] <= target < own[ANSWER - 1],
            f"the article is {result['bytes']:,} bytes and {result['words']:,} words; "
            f"{ENCODING} compresses it to {result['tiktoken']:.4f} tokens per byte, so the "
            f"exercise's window closes at {target:.4f}. {ANSWER} merges reach "
            f"{own[ANSWER]:.4f}, {100 * own[ANSWER] / result['tiktoken']:.1f}% of tiktoken, "
            f"and {ANSWER - 1} merges reach {own[ANSWER - 1]:.4f} and miss it. That is the "
            "tuned answer the exercise asks for",
        ),
        practice.Check(
            "FINDING: the target is a point on a slide, so there is nothing to tune",
            own[2 * ANSWER] < 0.8 * result["tiktoken"],
            f"double the merge count to {2 * ANSWER} and the ratio is {own[2 * ANSWER]:.4f}, "
            f"{100 * own[2 * ANSWER] / result['tiktoken']:.1f}% of tiktoken -- better than "
            "the tokenizer it was told to match, and it keeps improving with every merge "
            "added, down to the single token Exercise 1 found at exhaustion. 'Tune the "
            "number of merges to achieve within 10%' names a crossing on a monotone curve. "
            "Every count above the answer satisfies it too",
        ),
        practice.Check(
            "FINDING: the curve is monotone because train and test are the same text",
            held[HELD[0]] == held[HELD[1]] and own[2 * ANSWER] < own[ANSWER],
            f"hold out the last {100 - int(100 * SPLIT)}% of the article "
            f"({result['held_bytes']:,} bytes) and the slide stops: held-out compression is "
            f"{held[HELD[0]]:.4f} at {HELD[0]} merges and {held[HELD[1]]:.4f} at "
            f"{HELD[1]} -- identical, bit for bit, across 500 additional merges, while on "
            f"the training text the same range keeps improving. Every merge past ~{HELD[0]} "
            "encodes a string that occurs only in the part it trained on. The exercise's "
            "'on that same text' is what makes the objective monotone",
        ),
        practice.Check(
            "FINDING: within 10% is unreachable on held-out text at any merge count",
            floor > 1 + TOLERANCE and held[HELD[1]] > result["tiktoken_held"] * 1.3,
            f"the held-out floor is {held[HELD[0]]:.4f} against tiktoken's "
            f"{result['tiktoken_held']:.4f} on the same text -- {100 * floor:.1f}%, and no "
            "merge count improves on it. So the exercise's goal is over-satisfied under the "
            "measurement it specifies and impossible under the honest one, on a corpus this "
            "size. That gap is the relationship between corpus size, merge count and "
            "compression quality the exercise says it forces you to understand, and the "
            "metric it specifies is the one that hides it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
