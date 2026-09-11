"""Exercise 3 — the per-sentence table is undefined for short sentences.

    **Hard.** Fine-tune `nllb-200-distilled-600M` on a 5,000-pair domain corpus
    of your choice. Measure BLEU on a held-out set before and after fine-tuning.
    Report which kinds of sentences improved and which regressed.

Reading of the exercise: transformers and torch are absent and no 5,000-pair
corpus is here, so no fine-tune is run. The last sentence is the one that can
still be checked, because "which kinds of sentences improved and which
regressed" requires a per-sentence score, and the metric the lesson ships does
not have one for a large part of any corpus.

`simple_bleu` takes the geometric mean of four n-gram precisions with no
smoothing, and a sentence of fewer than four tokens has no 4-grams at all, so
`ngram_precision` returns 0.0 and the product collapses. Scored against itself,
a perfect translation of `Yes.` -- two tokens after the lesson's own tokenizer --
gets **0.0**; a four-token one gets 100.0. The threshold is exact and it is the
`max_n` argument. Across a 20-sentence corpus written to a length distribution
spanning 2 to 13 tokens, 4 sentences sit below it: 20% of the rows have no
before-number and no after-number, so no row in the improved-or-regressed table
either.

Above the threshold the problem changes shape rather than going away. On the
five round trips of exercise 1, three of the five score 0.0 while preserving
their meaning, so their entry in the table would read 0.0 to 0.0 whatever the
fine-tune did. chrF has no floor: it scores the three-token sentence 100.0 and
every drifted round trip between 52.1 and 80.5, and it is in the same file.

Structure: `CORPUS` is 20 (source, reference) pairs whose reference lengths span
1 to 12 tokens, each labelled with its token count; `identity` scores each
reference against itself, which is the highest score any system could earn on it.
`floor` finds the shortest length at which `simple_bleu` can return anything
above zero, by measurement rather than by reading `max_n`.
"""

from __future__ import annotations

import importlib.util
import pathlib

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "11-machine-translation"

SIBLING = "ex01_bleu_scores_a_paraphrase_like_a_wrong_language.py"
UNAVAILABLE = ("transformers", "torch", "datasets", "peft")
CORPUS = (
    "Yes.", "No.", "Understood.", "Confirmed today.", "The train left.",
    "Please sign here.", "The report is late.", "She approved the request.",
    "The flight was delayed.", "He returned the documents yesterday.",
    "The committee will meet again.", "Rain delayed the outbound flight.",
    "The proposal was accepted without discussion.",
    "She arrived at the station before the train departed.",
    "The report was published on the first of March.",
    "He explained the decision to the assembled journalists.",
    "Heavy rain delayed the connecting flight by two hours.",
    "The committee approved the revised proposal without further discussion.",
    "She sent the signed documents to the office before the deadline passed.",
    "The published report describes the delays that affected the March schedule.",
)


def lengths(ref) -> list:
    return [len(ref.tokenize(sentence)) for sentence in CORPUS]


def identity(ref) -> list:
    """The score a perfect system earns on each sentence -- the ceiling of the table."""
    return [(len(ref.tokenize(s)), round(ref.simple_bleu(s, s), 1), round(ref.chrf(s, s), 1))
            for s in CORPUS]


def floor(ref) -> int:
    """The shortest reference length that can score above zero, measured."""
    for size in range(1, 12):
        sentence = " ".join(["word"] * size)
        if ref.simple_bleu(sentence, sentence) > 0:
            return size
    return -1


def round_trips(ref, sibling) -> dict:
    """Exercise 1's five drifted round trips, rescored here."""
    scores = [sibling.score(ref, original, returned)
              for original, returned, kind in sibling.PARAGRAPH if kind == "drift"]
    return {"drift": len(scores), "drift_zero": sum(row["bleu"] == 0.0 for row in scores),
            "drift_chrf": (min(row["chrf"] for row in scores),
                           max(row["chrf"] for row in scores))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sibling = practice.load_module(pathlib.Path(__file__).with_name(SIBLING))
    rows = identity(ref)
    below = [row for row in rows if row[1] == 0.0]
    return dict(
        round_trips(ref, sibling), rows=rows, corpus=len(CORPUS), lengths=lengths(ref),
        missing=[m for m in UNAVAILABLE if importlib.util.find_spec(m) is None],
        below=len(below), below_lengths=sorted({n for n, _, _ in below}),
        floor=floor(ref), max_n=4, chrf_below=[c for _, _, c in below],
        short_pair=below[0])


def verify(result):
    rows, below = result["rows"], result["below"]
    length, bleu, chrf = result["short_pair"]
    return [
        practice.Check(
            "ANSWER: a perfect translation of a two-token sentence scores BLEU 0.0",
            bleu == 0.0 and chrf == 100.0 and result["floor"] == result["max_n"],
            f"{result['missing']} are all absent and no 5,000-pair corpus is here, so no fine-tune "
            f"is run. Scored against itself, a {length}-token reference gets `simple_bleu` {bleu} "
            f"and `chrf` {chrf}. The threshold is exact: the shortest length that can score above "
            f"zero is {result['floor']}, which is the `max_n` argument"),
        practice.Check(
            "MECHANISM: fewer than four tokens means no 4-grams, and the geometric mean collapses",
            result["below_lengths"] and max(result["below_lengths"]) < result["max_n"],
            f"`ngram_precision` returns 0.0 when the hypothesis has no n-grams of that order, and "
            f"`simple_bleu` returns 0.0 if any precision is zero. Every reference of "
            f"{result['below_lengths']} tokens is affected, and nothing a system does can move "
            f"those rows"),
        practice.Check(
            "FINDING: a fifth of a realistic corpus has no before-number and no after-number",
            below / result["corpus"] == 0.2,
            f"{below} of {result['corpus']} references, written to a length distribution spanning "
            f"{min(result['lengths'])} to {max(result['lengths'])} tokens, score 0.0 at their own "
            f"ceiling -- {below / result['corpus']:.0%} of the corpus. 'Which kinds of sentences "
            f"improved and which regressed' has no row for any of them: the before and the after "
            f"are both zero by construction"),
        practice.Check(
            "FINDING: above the threshold the table is still undefined for good paraphrases",
            result["drift_zero"] == 3 and result["drift"] == 5,
            f"of exercise 1's {result['drift']} round trips, {result['drift_zero']} score 0.0 while "
            f"preserving their meaning. Their entry would read 0.0 before and 0.0 after whatever "
            f"the fine-tune did, so the rows that survive the length threshold can still be "
            f"unreadable"),
        practice.Check(
            "MECHANISM: chrF has no floor, and it is in the same file",
            all(c == 100.0 for c in result["chrf_below"]),
            f"the {below} sentences BLEU zeroes all score chrF {sorted(set(result['chrf_below']))} at "
            f"their own ceiling, and the drifted round trips score "
            f"{result['drift_chrf'][0]} to {result['drift_chrf'][1]}. Character n-grams do not run "
            f"out on a three-token sentence, so the per-sentence table the exercise asks for is "
            f"available -- from the other function"),
        practice.Check(
            "CONTROL: the sentences BLEU can score are exactly the long ones",
            all(b > 0 for n, b, _ in rows if n >= result["floor"]),
            f"every reference of {result['floor']} tokens or more scores 100.0 at its ceiling and "
            f"every shorter one scores 0.0 -- the split is by length and by nothing else. A "
            f"before-and-after comparison on this metric is a comparison over the long half of the "
            f"corpus, reported as though it covered all of it"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
