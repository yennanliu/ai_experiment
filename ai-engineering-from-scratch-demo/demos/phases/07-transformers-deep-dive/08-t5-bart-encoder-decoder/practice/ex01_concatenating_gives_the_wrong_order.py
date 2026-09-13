"""Exercise 1 — concatenating reproduces the length, not the sentence.

    **Easy.** Run `code/main.py`, apply span corruption to a 30-token sentence,
    verify that concatenating the non-sentinel source tokens with the decoded
    target spans reproduces the original.

Reading of the exercise: both procedures are run on the same corrupted pair --
the one the exercise describes, `[source without sentinels] + [target without
sentinels]`, and the one the lesson ships, `round_trip`, which substitutes each
span back where its sentinel stands. The sentence is the lesson's own, extended
to the 30 tokens the exercise asks for; the lesson's own is 25.

**ANSWER: the lesson's `round_trip` reproduces the original 300 times out of
300. The concatenation the exercise describes does not.** It gets the length
exactly right -- 30 tokens -- and the order wrong, with the first mismatch at
index 2, because every masked span is appended at the end instead of returned to
its own position. Concatenation cannot work: the sentinel *is* the position, and
dropping it drops the only record of where the span goes.

**FINDING: at the lesson's own default rate, 30 tokens get one span and 7.8% of
them are masked.** Two roundings compound. `n_mask = int(round(30 * 0.15))` is
`round(4.5)`, and Python rounds halves to even, so it is **4** and not 5 --
13.3% before anything is placed. Then `n_spans = int(round(4 / 3.0))` is **1**,
so "span corruption" at its documented settings produces a single span. Measured
over 300 seeds the realised mask is **2.33 tokens, 7.8%**, roughly half the 15%
asked for, because `int(rng.gauss(3, 1))` truncates toward zero and the span is
clipped again by `n - start`.

**CONTROL: the closing sentinel is load-bearing.** `round_trip`'s parser stores
a span only when it meets the *next* sentinel, so the trailing `<extra_id_k>` is
what flushes the last one. Remove it and the final span is silently dropped.

**CONTROL: no two sentinels are ever adjacent.** Over 300 seeds, zero -- which
follows from there being one span, and would matter if there were more.

Structure: `literal` is the exercise's procedure; `realised` measures what the
corruption actually masks; `flushes` probes the closing sentinel.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "08-t5-bart-encoder-decoder"
RATE, MEAN_SPAN, TRIALS, LENGTH = 0.15, 3.0, 300, 30
TEXT = ("the quick brown fox jumps over the lazy dog a stitch in time saves nine language "
        "models learn statistical patterns subword tokenization helps rare words today and "
        "again forever more")


def plain(tokens):
    """Everything that is not a sentinel, in order."""
    return [t for t in tokens if not t.startswith("<extra_id_")]


def literal(source, target):
    """The exercise's own procedure: source words, then target words."""
    return plain(source) + plain(target)


def realised(ref, sentence, trials=TRIALS):
    """(round-trips, mean spans, mean masked tokens, adjacent-sentinel runs) over seeds."""
    exact = adjacent = 0
    spans, masked = [], []
    for seed in range(trials):
        source, target = ref.corrupt_spans(sentence, RATE, MEAN_SPAN, random.Random(seed))
        exact += ref.round_trip(source, target) == sentence
        spans.append(len(source) - len(plain(source)))
        masked.append(len(plain(target)))
        adjacent += any(a.startswith("<extra_id_") and b.startswith("<extra_id_")
                        for a, b in zip(source, source[1:]))
    return exact, statistics.fmean(spans), statistics.fmean(masked), adjacent


def flushes(ref, source, target):
    """Does dropping the closing sentinel lose the last span?"""
    return ref.round_trip(source, target[:-1])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sentence = TEXT.split()[:LENGTH]
    source, target = ref.corrupt_spans(sentence, 0.20, MEAN_SPAN, random.Random(42))
    joined = literal(source, target)
    exact, spans, masked, adjacent = realised(ref, sentence)
    return {
        "length": len(sentence), "lesson_length": len(TEXT.split()[:25]),
        "joined": joined == sentence, "joined_len": len(joined),
        "mismatch": next((i for i, (a, b) in enumerate(zip(joined, sentence)) if a != b), None),
        "round_trip": ref.round_trip(source, target) == sentence,
        "exact": exact, "trials": TRIALS, "spans": spans, "masked": masked,
        "rate": masked / len(sentence), "n_mask": int(round(LENGTH * RATE)),
        "n_spans": max(1, int(round(int(round(LENGTH * RATE)) / MEAN_SPAN))),
        "truncated": flushes(ref, source, target) == sentence, "adjacent": adjacent,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: round_trip reproduces the original 300/300; the concatenation does not",
            result["round_trip"] and result["exact"] == TRIALS and not result["joined"],
            f"the lesson's round_trip is exact on {result['exact']}/{result['trials']} seeds. The "
            f"concatenation the exercise describes gets the length right -- {result['joined_len']} "
            f"tokens against {result['length']} -- and the order wrong, first mismatch at index "
            f"{result['mismatch']}. Every span is appended at the end instead of going back to "
            "where its sentinel stands",
        ),
        practice.Check(
            "ANSWER: it cannot work, because the sentinel is the position",
            result["mismatch"] is not None,
            "dropping the sentinels drops the only record of where each span belongs, so the "
            "concatenation is the original's multiset and not its sequence. round_trip substitutes "
            "in place, which is why it needs the numbering: <extra_id_k> in the source and "
            "<extra_id_k> in the target are the two halves of one pointer",
        ),
        practice.Check(
            "FINDING: at the documented rate, 30 tokens get exactly one span",
            result["n_mask"] == 4 and result["n_spans"] == 1 and result["spans"] == 1.0,
            f"n_mask = int(round({LENGTH} * {RATE})) = int(round(4.5)) = {result['n_mask']}, "
            f"because Python rounds halves to even -- 13.3% before a span is placed -- and then "
            f"n_spans = int(round({result['n_mask']} / {MEAN_SPAN})) = {result['n_spans']}. "
            f"Measured over {TRIALS} seeds the mean span count is {result['spans']:.2f}, so "
            "mean_span=3.0 selects the number of spans and never their length",
        ),
        practice.Check(
            "FINDING: the realised mask is 7.8%, about half of what was asked",
            result["rate"] < RATE * 0.65,
            f"{result['masked']:.2f} tokens of {result['length']} = {result['rate']:.1%} against "
            f"the {RATE:.0%} requested. int(rng.gauss({MEAN_SPAN}, 1)) truncates toward zero, "
            "losing half a token on average, and the span is clipped again by min(length, "
            "remaining, n - start) whenever it starts near the end",
        ),
        practice.Check(
            "CONTROL: the closing sentinel is load-bearing, and no two sentinels are adjacent",
            not result["truncated"] and result["adjacent"] == 0,
            "round_trip's parser stores a span only when it meets the *next* sentinel, so the "
            "trailing <extra_id_k> is what flushes the last one -- drop it and the reconstruction "
            f"stops matching. And over {TRIALS} seeds no two sentinels are ever adjacent in the "
            f"source ({result['adjacent']} runs), which follows from there being one span",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
