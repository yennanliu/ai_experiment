"""Exercise 1 — six grams of prefix, whatever follows.

    **Easy.** Run `char_ngrams("playing")` and `char_ngrams("played")`. Compute
    the Jaccard overlap of the two n-gram sets. You should see substantial
    shared pieces (`pla`, `lay`, `play`), which is why FastText transfers well
    across morphological variants.

Reading of the exercise: all three pieces it names are shared, and they are
three of the six shared n-grams in a union of 36 -- a Jaccard of 0.1667. Both
halves of that are worth stating. Against a null it is a real signal:
`playing` and `walked`, two regular past-and-present verbs, share zero, and so
do `banana` and `xylophone`. Five sixths of the evidence not being shared is
the price of `<` and `>` and a 3-gram floor, not a failure.

What the number cannot do is rank. The shared set between `playing` and any
word beginning `play` is exactly the six n-grams of the prefix `<play`, so
`played`, `player` and `playful` all share exactly 6 -- an inflection, an agent
noun and an adjective, scored identically. Jaccard separates them only through
the union, which is a function of the other word's length: 0.2222 for `play`,
0.1667 for the two six-letter words and 0.1500 for `playful`. Ranking
morphological relatives by Jaccard is ranking them by how short they are. The
whole-word gram makes it slightly worse -- `<playing>` is in one set and never
in the other, so every pair of distinct words pays one union it can never
recover.

Structure: `jaccard`, `overlap` and `shared` are the three ways to score one
pair of n-gram sets; `PREFIXED` are the words that share `<play`, `SUFFIXED`
share only an inflection, and `UNRELATED` is the null. `score` runs all three
measures against `playing` for a list of words.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "04-glove-fasttext-subword"

BASE, PARTNER = "playing", "played"
PREFIXED = ("play", "played", "player", "playful")
SUFFIXED = ("singing", "running")
UNRELATED = ("walked", "banana", "xylophone")
NAMED = ("pla", "lay", "play")

jaccard = lambda a, b: len(a & b) / len(a | b)                                    # noqa: E731
overlap = lambda a, b: len(a & b) / min(len(a), len(b))                           # noqa: E731


def score(ref, words) -> dict:
    base = ref.char_ngrams(BASE)
    return {w: {"shared": len(base & ref.char_ngrams(w)),
                "jaccard": round(jaccard(base, ref.char_ngrams(w)), 4),
                "overlap": round(overlap(base, ref.char_ngrams(w)), 4),
                "length": len(w)} for w in words}


def column(groups, name, words, key) -> list:
    return [groups[name][w][key] for w in words]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    left, right = ref.char_ngrams(BASE), ref.char_ngrams(PARTNER)
    groups = {name: score(ref, words) for name, words in
              (("prefixed", PREFIXED), ("suffixed", SUFFIXED), ("unrelated", UNRELATED))}
    return {
        "sizes": (len(left), len(right)), "shared": sorted(left & right),
        "union": len(left | right), "jaccard": jaccard(left, right),
        "named_all_shared": all(g in left & right for g in NAMED),
        "groups": groups,
        "prefix_counts": column(groups, "prefixed", PREFIXED, "shared"),
        "prefix_jaccard": column(groups, "prefixed", PREFIXED, "jaccard"),
        "prefix_lengths": column(groups, "prefixed", PREFIXED, "length"),
        "whole": (f"<{BASE}>" in left, f"<{BASE}>" in right),
        "nulls": column(groups, "unrelated", UNRELATED, "shared"),
        "stems": column(groups, "suffixed", SUFFIXED, "shared"),
        "stem_jaccard": column(groups, "suffixed", SUFFIXED, "jaccard"),
    }


def verify(result):
    counts, jac = result["prefix_counts"], result["prefix_jaccard"]
    nulls, stems = result["nulls"], result["stems"]
    return [
        practice.Check(
            "ANSWER: the Jaccard is 0.1667 -- six shared n-grams in a union of 36",
            abs(result["jaccard"] - 6 / 36) < 1e-9 and result["named_all_shared"],
            f"char_ngrams('{BASE}') has {result['sizes'][0]} grams and char_ngrams('{PARTNER}') has "
            f"{result['sizes'][1]}; they share {len(result['shared'])} -- {result['shared']} -- over "
            f"a union of {result['union']}, for {result['jaccard']:.4f}. All three pieces the "
            f"exercise names are there, and they are half of what is shared"),
        practice.Check(
            "CONTROL: against a null the signal is real -- unrelated words share nothing at all",
            max(nulls) == 0,
            f"{list(UNRELATED)} each share {nulls} n-grams with '{BASE}'. 'walked' is a "
            f"regular past tense like 'played' and overlaps it not at all, because what 'played' "
            f"shares with '{BASE}' is the stem and nothing else. So 0.1667 is not a small number "
            f"against 0 -- it is the whole of the morphological signal"),
        practice.Check(
            "MECHANISM: every word beginning 'play' shares exactly the six grams of '<play'",
            len(set(counts)) == 1 and counts[0] == len(result["shared"]),
            f"{list(PREFIXED)} share {counts} n-grams with '{BASE}' respectively -- identical. The "
            f"intersection is fixed by the common prefix, so an inflection ('played'), an agent noun "
            f"('player') and an adjective ('playful') are scored the same. Nothing after the shared "
            f"prefix can enter the intersection"),
        practice.Check(
            "FINDING: Jaccard separates them only by length, so the ranking is a length ranking",
            jac == sorted(jac, reverse=True) == [
                j for _, j in sorted(zip(result["prefix_lengths"], jac))],
            f"the same six shared grams give Jaccard {jac} for {list(PREFIXED)}, whose lengths are "
            f"{result['prefix_lengths']}. Only the union moves, and the union is the other word's "
            f"gram count. Sorting these four by Jaccard sorts them shortest first, which is not a "
            f"morphological relation"),
        practice.Check(
            "FINDING: a shared stem outscores a shared suffix, which is the property being claimed",
            min(counts) > max(stems),
            f"words sharing the stem score {counts}; 'singing' and 'running', which share only the "
            f"inflection '-ing', score {stems} at Jaccard {result['stem_jaccard']}. "
            f"The boundary markers are "
            f"what buy this: '<pl' and '<pla' are only available to words that start the same way"),
        practice.Check(
            "CONTROL: the whole-word gram is a union every distinct pair pays and never recovers",
            result["whole"] == (True, False),
            f"`char_ngrams` seeds its set with the wrapped word itself, so '<{BASE}>' is in "
            f"{BASE}'s set and in no other word's. For any two distinct words that is +1 to the "
            f"union and 0 to the intersection -- here it is 1 of the {result['union']} that "
            f"{len(result['shared'])} is divided by"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
