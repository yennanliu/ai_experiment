"""Exercise 3 — seen once is seen never.

    **Hard.** Train a 1k-merge BPE on Shakespeare's complete works. Compare
    tokenization of common words vs. rare proper nouns. Measure average tokens
    per word before and after. Write up what surprised you.

Reading of the exercise: the complete works are not in this checkout and are
not downloaded, so the corpus is the 29 English lesson documents of this phase
-- 46k word tokens over 4879 types, read through `parity.doc_text`, which
happens to contain `shakespeare` twice. Three word lists are scored rather than
two, and the third is what answers the last sentence. `COMMON` are frequent
corpus words, `RARE` are proper nouns that occur in the corpus once or twice,
and `UNSEEN` are proper nouns that occur zero times.

The surprise is that `RARE` and `UNSEEN` score the same, and `RARE` scores
slightly worse: after 1000 merges a proper noun seen once or twice in training
costs 4.400 tokens and one never seen at all costs 4.200, against 1.067 for a
common word. BPE merges by frequency, and a pair occurring once never wins a
merge against a pair occurring hundreds of times, so a name in the corpus at
count 1 buys exactly what a name outside it buys. `shakespeare`, present twice,
comes out as six tokens.

"Before and after" is the other thing to watch. The before number is not a
property of BPE at all: at zero merges every word is its characters plus
`</w>`, so tokens per word is mean word length plus one, and the three lists
score 6.933, 8.000 and 8.100 because those are their lengths. The informative
comparison is not before against after, it is common against rare after --
where common words saturate at 1.067 by 1000 merges and the tail is still at
4.4, so every merge past that point is bought for a vocabulary that no longer
needs it.

Structure: `corpus` reads the phase's lesson documents through the harness and
counts word types; the three word lists are hand-labelled below. `tokens_per`
is the mean token count for a list at a given merge budget, and `sweep` walks
the budgets so the two curves can be compared rather than just their endpoints.
"""

from __future__ import annotations

import collections
import re

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "04-glove-fasttext-subword"

WORD = re.compile(r"[A-Za-z][A-Za-z']*")
BUDGETS = (0, 100, 300, 1000)
COMMON = ("model", "word", "token", "context", "text", "return", "output", "language",
          "sentence", "vector", "training", "score", "first", "corpus", "result")
RARE = ("mikolov", "pennington", "bojanowski", "salton", "firth", "markov", "shakespeare",
        "imdb", "reuters", "penn")
UNSEEN = ("levenshtein", "kullback", "leibler", "zipf", "gutenberg", "luhn", "chomsky",
          "hungarian", "meteor", "harris")
LISTS = {"common": COMMON, "rare": RARE, "unseen": UNSEEN}

mean_length = lambda words: sum(map(len, words)) / len(words)                     # noqa: E731


def corpus() -> collections.Counter:
    """The phase's own lesson documents, located and read through the harness."""
    root = parity.find_reference_root() / "phases" / PHASE
    lessons = sorted(d.name for d in root.iterdir() if (d / "docs" / "en.md").is_file())
    texts = [parity.doc_text(PHASE, lesson, "en") for lesson in lessons]
    return collections.Counter(w.lower() for text in texts for w in WORD.findall(text))


def tokens_per(ref, words, merges) -> float:
    return sum(len(ref.apply_bpe(word, merges)) for word in words) / len(words)


def sweep(ref, counts) -> tuple:
    merges = ref.learn_bpe(counts, BUDGETS[-1])
    rows = {k: {name: round(tokens_per(ref, words, merges[:k]), 3)
                for name, words in LISTS.items()} for k in BUDGETS}
    return merges, rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    try:
        counts = corpus()
    except Exception as exc:                        # pragma: no cover - reference unreachable
        raise practice.Skip(f"needs the reference checkout: set AIEFS_REFERENCE ({exc})") from None
    merges, rows = sweep(ref, counts)
    return {
        "types": len(counts), "tokens": sum(counts.values()), "merges": len(merges),
        "rows": rows, "freq": {w: counts[w] for w in RARE},
        "unseen_freq": {w: counts[w] for w in UNSEEN},
        "lengths": {name: round(mean_length(words), 3) for name, words in LISTS.items()},
        "pieces": {w: ref.apply_bpe(w, merges) for w in ("shakespeare", "mikolov", "model")},
    }


def verify(result):
    rows, lengths, last = result["rows"], result["lengths"], BUDGETS[-1]
    first, final = rows[0], rows[last]
    return [
        practice.Check(
            "ANSWER: after 1000 merges a common word costs 1.067 tokens and a rare name 4.400",
            final["common"] < 1.2 < 4.0 < final["rare"],
            f"the complete works are not in this checkout, so the corpus is this phase's "
            f"{result['tokens']} word tokens over {result['types']} types, and "
            f"{result['merges']} merges are learned. Mean tokens per word at {last} merges: "
            f"common {final['common']}, rare proper nouns {final['rare']}, unseen proper nouns "
            f"{final['unseen']} -- a factor of {final['rare'] / final['common']:.1f}"),
        practice.Check(
            "FINDING: the surprise -- a name seen twice costs more than a name never seen",
            final["rare"] >= final["unseen"],
            f"proper nouns in the training corpus at counts {result['freq']} score "
            f"{final['rare']}; proper nouns at counts {result['unseen_freq']} score "
            f"{final['unseen']}. Being in the corpus bought nothing, because BPE merges by "
            f"frequency and a pair seen once never outranks one seen hundreds of times. "
            f"'shakespeare', present twice, comes out as {result['pieces']['shakespeare']}"),
        practice.Check(
            "MECHANISM: the 'before' number is mean word length plus one, not a fact about BPE",
            all(abs(first[name] - lengths[name] - 1) < 1e-9 for name in LISTS),
            f"at 0 merges every word is its characters plus `</w>`, so tokens per word is "
            f"{first} against mean lengths {lengths} -- exactly one more, on all three lists. "
            f"'Measure average tokens per word before and after' compares BPE to an alphabet, and "
            f"the before half is settled by how long the words are"),
        practice.Check(
            "FINDING: the common vocabulary saturates while the tail is still fragmenting",
            rows[300]["common"] < 2.0 and rows[300]["rare"] > 4.0
            and final["common"] < rows[300]["common"],
            f"common words go {[rows[k]['common'] for k in BUDGETS]} across budgets {list(BUDGETS)}, "
            f"reaching {final['common']} -- essentially one token each. Rare names go "
            f"{[rows[k]['rare'] for k in BUDGETS]} and end at {final['rare']}. The last 700 merges "
            f"move the common list by {rows[300]['common'] - final['common']:.3f} and the rare list "
            f"by {rows[300]['rare'] - final['rare']:.3f}"),
        practice.Check(
            "CONTROL: the gap is not word length -- the three lists start within 1.2 of each other",
            max(lengths.values()) - min(lengths.values()) < 1.5
            and final["rare"] - final["common"] > 3.0,
            f"mean word lengths are {lengths}, a spread of "
            f"{max(lengths.values()) - min(lengths.values()):.3f}, and at 0 merges the token counts "
            f"differ by {max(first.values()) - min(first.values()):.3f}. At {last} merges they "
            f"differ by {max(final.values()) - min(final.values()):.3f}. BPE opened that gap; the "
            f"words did not arrive with it"),
        practice.Check(
            "CONTROL: a common word becomes one token that is the word itself",
            result["pieces"]["model"] == ["model</w>"] and len(result["pieces"]["mikolov"]) > 3,
            f"'model' tokenizes to {result['pieces']['model']} and 'mikolov' to "
            f"{result['pieces']['mikolov']}. A 1000-merge vocabulary is a list of frequent words "
            f"with a character fallback behind it, and which side of that a word lands on is "
            f"decided by its corpus frequency, not by its shape"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
