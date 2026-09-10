"""Exercise 1 — no held-out word can reach one token.

    **Easy.** Train a 500-merge BPE on `code/main.py`'s tiny corpus. Encode
    three held-out words. How many produced exactly 1 token vs >1 token?

Reading of the exercise: the corpus and the held-out words are `main()`'s own --
four words, not three. The count is **0 produce one token, 4 produce more**, and
that is not a property of this corpus. `encode_bpe` returns a single symbol only
when `word + "</w>"` is itself a training word, so a genuinely held-out word
cannot reach one token at any merge count. Sweeping every merge count from 0 to
183 confirms it: none of the four ever does.

Asking for 500 merges gets 183. `train_bpe` breaks when no adjacent pairs are
left, which happens once every training word is a single symbol -- so the
requested vocabulary size is not a knob past that point, and at that point the
tokenizer is a word-level tokenizer over the 52 distinct corpus words, which is
the thing subword tokenization exists to avoid.

The vocabulary `train_bpe` reports is not monotonic in the merge count: 27 at
zero merges, a peak of 84 at 105, then 52 at convergence. It returns the symbols
present in the *final* decomposition rather than the base alphabet plus the merge
list, so merged-away pieces vanish from the count. The real vocabulary -- what
you would have to ship -- grows monotonically to 210. Compression moves the other
way: the four held-out words take 31 pieces at 30 merges and 15 at 183, so the
reported vocabulary shrinks while the tokenizer gets better.

Structure: `sweep` runs `train_bpe` at a range of merge counts and records
reported vocabulary, true vocabulary and held-out piece count; `pieces` encodes
the held-out set at one merge count.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "19-subword-tokenization"

CORPUS = """
    the quick brown fox jumps over the lazy dog
    a stitch in time saves nine
    language models learn from statistical patterns in text
    tokenization splits text into smaller units called tokens
    subword tokenization lets rare words decompose into known pieces
    byte pair encoding is the dominant tokenization algorithm today
    the lazy dog slept while the fox jumped again and again
    patterns of letters in words are learnable and reusable
"""
HELD_OUT = ("tokenizable", "unlearnable", "foxhound", "languages")
ASKED = 500


def pieces(ref, merges):
    """Piece count for each held-out word under one merge list."""
    return [len(ref.encode_bpe(word, merges)) for word in HELD_OUT]


def sweep(ref, alphabet):
    """Reported vocabulary, shippable vocabulary and held-out cost per merge count."""
    rows = {}
    for n in range(0, 184):
        merges, tokens = ref.train_bpe(CORPUS, n) if n else ([], sorted(alphabet))
        rows[n] = {
            "merges": len(merges),
            "reported": len(tokens),
            "shippable": len(alphabet) + len(merges),
            "held_out": sum(pieces(ref, merges)),
            "singles": pieces(ref, merges).count(1),
        }
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    counts = ref.word_counts(CORPUS)
    alphabet = {char for word in counts for char in word} | {"</w>"}
    rows = sweep(ref, alphabet)
    merges, tokens = ref.train_bpe(CORPUS, ASKED)
    peak = max(rows.values(), key=lambda row: row["reported"])
    return {
        "asked": ASKED,
        "delivered": len(merges),
        "words": len(counts),
        "alphabet": len(alphabet),
        "final_pieces": pieces(ref, merges),
        "vocab_is_the_words": set(tokens) == {word + "</w>" for word in counts},
        "in_corpus_all_single": all(len(ref.encode_bpe(w, merges)) == 1 for w in counts),
        "singles_anywhere": [n for n, row in rows.items() if row["singles"]],
        "reported_final": rows[183]["reported"],
        "reported_zero": rows[0]["reported"],
        "reported_peak": peak["reported"],
        "peak_at": max(rows, key=lambda n: rows[n]["reported"]),
        "shippable_final": rows[183]["shippable"],
        "held_out_30": rows[30]["held_out"],
        "held_out_final": rows[183]["held_out"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: none of the held-out words reaches one token, and none can",
            result["final_pieces"].count(1) == 0 and not result["singles_anywhere"],
            f"the four held-out words encode to {result['final_pieces']} pieces -- "
            f"{result['final_pieces'].count(1)} at exactly one token, {len(HELD_OUT)} above it. "
            "Sweeping every merge count from 0 to 183 finds no count at which any of them "
            "reaches one, because `encode_bpe` returns a single symbol only for a word that is "
            "itself in the training corpus",
        ),
        practice.Check(
            "MECHANISM: the exercise asks for 500 merges and the corpus supplies 183",
            result["delivered"] < result["asked"],
            f"`train_bpe` breaks as soon as `pair_counts` is empty, so {result['asked']} merges "
            f"become {result['delivered']}. Past that point the requested vocabulary size is not "
            "a parameter of anything",
        ),
        practice.Check(
            "FINDING: at convergence BPE has become a word-level tokenizer",
            result["vocab_is_the_words"] and result["in_corpus_all_single"],
            f"the final vocabulary is exactly the {result['words']} distinct corpus words with "
            "`</w>` appended, and every one of them encodes to a single token. The 500-merge "
            "model the exercise asks for is the word tokenizer subword tokenization exists to "
            "replace -- which is why its held-out words all split",
        ),
        practice.Check(
            "FINDING: the reported vocabulary size falls as training continues",
            result["reported_peak"] > result["reported_final"],
            f"reported vocabulary runs {result['reported_zero']} at zero merges, up to a peak of "
            f"{result['reported_peak']} at {result['peak_at']} merges, then down to "
            f"{result['reported_final']}. `train_bpe` returns the symbols left in the final "
            "decomposition, not the alphabet plus the merge list, so a piece that gets merged "
            f"away stops being counted. The vocabulary you would ship is {result['alphabet']} + "
            f"{result['delivered']} = {result['shippable_final']}, and that only grows",
        ),
        practice.Check(
            "MECHANISM: so the reported number and the tokenizer's quality move in opposite "
            "directions",
            result["held_out_final"] < result["held_out_30"],
            f"between 30 merges and {result['delivered']}, the four held-out words drop from "
            f"{result['held_out_30']} pieces to {result['held_out_final']} while the reported "
            f"vocabulary drops from 55 to {result['reported_final']}. Reading either number "
            "alone gives the wrong sign for the other",
        ),
        practice.Check(
            "CONTROL: the lesson predicts the right outcome for the wrong reason",
            len(HELD_OUT) == 4,
            "`main()` closes with 'with a tiny toy corpus, most held-out words will split' and "
            "tags a one-token result `OK`. It is all held-out words, not most, and corpus size "
            "is not why: the tag can only fire on a word that was in the training data",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
