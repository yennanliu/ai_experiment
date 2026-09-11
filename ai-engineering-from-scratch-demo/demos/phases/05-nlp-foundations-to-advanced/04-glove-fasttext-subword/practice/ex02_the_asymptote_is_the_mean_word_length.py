"""Exercise 2 — the asymptote is the mean word length.

    **Medium.** Extend `learn_bpe` to track vocabulary growth. Plot
    tokens-per-corpus-character as a function of number of merges. You should
    see rapid compression at first, asymptoting near ~2-3 chars per token.

Reading of the exercise: the quantity it says to plot and the quantity it says
to expect are reciprocals of each other, and they move in opposite directions.
Tokens per corpus character falls from 1.195 to 0.195; chars per token rises
from 0.837 to 5.136. Neither asymptotes near 2-3. Chars per token passes
through 2.055 at 8 merges and 2.457 at 10, which is where the curve is
steepest -- the range the exercise names is a place the curve goes through on
its way somewhere else, not where it flattens.

Where it flattens is exact rather than approximate. `learn_bpe` breaks when no
pair remains, so the toy corpus stops accepting merges at 18 however many are
asked for -- 20, 30 and 50 all return 18. At that point every word is a single
token, so chars per token equals total characters over total words, 113/22 =
5.136, the corpus's mean word length. A six-word corpus does not learn subwords;
it memorises words, and 2-3 chars per token is a property of a corpus large
enough that it cannot.

One number in the curve needs saying: at zero merges there are more tokens than
characters, 135 against 113. `learn_bpe` appends `</w>` to every word, so the
starting count is characters plus words. Tokens per corpus character therefore
starts above 1 and the "compression" its first few points show is partly the
removal of a marker the tokenizer itself added.

Structure: `CORPUS` is the lesson's own toy corpus from `main()`. `curve`
returns, for each merge budget, the merges actually learned and the token count
over the whole corpus; `chars` and `words` are the two constants the asymptote
is built from.
"""

from __future__ import annotations

import collections

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "04-glove-fasttext-subword"

CORPUS = collections.Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3,
                              "lowest": 4, "newer": 2})
BUDGETS = (0, 1, 2, 4, 6, 8, 10, 15, 20, 30, 50)
CLAIMED = (2.0, 3.0)

chars = lambda: sum(len(w) * f for w, f in CORPUS.items())                        # noqa: E731
words = lambda: sum(CORPUS.values())                                              # noqa: E731


def curve(ref) -> dict:
    """Merges actually learned and total corpus tokens, per requested budget."""
    out = {}
    for budget in BUDGETS:
        merges = ref.learn_bpe(CORPUS, budget)
        tokens = sum(len(ref.apply_bpe(word, merges)) * freq for word, freq in CORPUS.items())
        out[budget] = {"merges": len(merges), "tokens": tokens,
                       "per_char": round(tokens / chars(), 4),
                       "per_token": round(chars() / tokens, 3)}
    return out


def budgets_where(rows, predicate) -> list:
    return [b for b in BUDGETS if predicate(rows[b])]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = curve(ref)
    ceiling = rows[BUDGETS[-1]]["merges"]
    saturated = budgets_where(rows, lambda row: row["merges"] == ceiling)
    inside = budgets_where(rows, lambda row: CLAIMED[0] <= row["per_token"] <= CLAIMED[1])
    return {
        "rows": rows, "chars": chars(), "words": words(),
        "per_char": [rows[b]["per_char"] for b in BUDGETS],
        "per_token": [rows[b]["per_token"] for b in BUDGETS],
        "ceiling": ceiling, "first_saturated": min(saturated), "flat": saturated,
        "asymptote": rows[BUDGETS[-1]]["per_token"], "final_tokens": rows[BUDGETS[-1]]["tokens"],
        "inside": inside, "start": rows[0]["tokens"],
        "inside_values": [rows[b]["per_token"] for b in inside],
        "whole": sorted(w for w in CORPUS if [w + "</w>"] == ref.apply_bpe(w, ref.learn_bpe(
            CORPUS, BUDGETS[-1]))),
        "steep": [round(rows[b]["per_token"] - rows[BUDGETS[i]]["per_token"], 3)
                  for i, b in list(enumerate(BUDGETS))[1:]],
    }


def verify(result):
    rows, inside, asym = result["rows"], result["inside"], result["asymptote"]
    return [
        practice.Check(
            "ANSWER: the two quantities the exercise names are reciprocals and move opposite ways",
            result["per_char"][0] > result["per_char"][-1] and asym > rows[0]["per_token"],
            f"tokens per corpus character falls {result['per_char'][0]} -> {result['per_char'][-1]} "
            f"over {list(BUDGETS)} merges; chars per token, which is the same curve inverted, rises "
            f"{rows[0]['per_token']} -> {asym}. The exercise says to plot the first and expect the "
            f"second to settle near {CLAIMED[0]:.0f}-{CLAIMED[1]:.0f}"),
        practice.Check(
            "MECHANISM: the asymptote is exactly the corpus's mean word length",
            abs(asym - result["chars"] / result["words"]) < 5e-4
            and result["final_tokens"] == result["words"],
            f"past saturation every word is one token: {result['final_tokens']} tokens for "
            f"{result['words']} words, so chars per token is {result['chars']}/{result['words']} = "
            f"{result['chars'] / result['words']:.3f}. Nothing about that is approximate, and "
            f"nothing about it is {CLAIMED[0]:.0f}-{CLAIMED[1]:.0f}"),
        practice.Check(
            "FINDING: the curve stops at 18 merges however many are requested",
            result["ceiling"] == 18 and result["first_saturated"] < BUDGETS[-1],
            f"`learn_bpe` breaks when `pair_freq` is empty, so budgets {result['flat']} all return "
            f"{result['ceiling']} merges and identical token counts. A six-word corpus has 18 merges "
            f"in it; asking for 50 is not an error and not more compression either"),
        practice.Check(
            "FINDING: the 2-3 range is where the curve is steepest, not where it flattens",
            inside and max(inside) < result["first_saturated"] and asym > CLAIMED[1],
            f"chars per token is inside [{CLAIMED[0]:.0f}, {CLAIMED[1]:.0f}] only at budgets "
            f"{inside}, reading {result['inside_values']} -- and the per-step gains "
            f"there are the largest in the run. It passes through the exercise's range and keeps "
            f"going to {asym}"),
        practice.Check(
            "MECHANISM: at zero merges there are more tokens than characters",
            result["start"] == result["chars"] + result["words"] and result["per_char"][0] > 1.0,
            f"`learn_bpe` appends `</w>` to every word, so the starting count is "
            f"{result['chars']} characters plus {result['words']} end markers = {result['start']}, "
            f"for {result['per_char'][0]} tokens per character. The first points of the curve are "
            f"partly the tokenizer undoing a marker it added"),
        practice.Check(
            "CONTROL: a six-word corpus memorises words rather than learning subwords",
            len(result["whole"]) == len(CORPUS),
            f"at saturation all {len(result['whole'])} corpus words -- {result['whole']} -- "
            f"tokenize to a single token that is the word itself, so the vocabulary BPE produced is "
            f"the corpus. That is why the asymptote is a word length: "
            f"{CLAIMED[0]:.0f}-{CLAIMED[1]:.0f} chars per token is a statement about a corpus with "
            f"more words in it than merges available to memorise them"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
