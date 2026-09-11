"""Exercise 1 — the trap it names is the one it survives.

    **Easy.** Run `facebook/bart-large-mnli` on 20 hand-crafted (premise,
    hypothesis, label) triples covering all three classes. Measure accuracy. Add
    adversarial "subsequence heuristic" traps ("I did not eat the cake" vs "I ate
    the cake") and see if it breaks.

Reading of the exercise: `transformers` and `torch` are absent and the checkpoint
is not downloadable, so the classifier under test is the lesson's own
`predict_nli`. On 20 triples balanced across the three classes it scores
**15/20**, and the shape of the five errors is the finding.

The trap the exercise names does not break it. Negation is one of the two
features `predict_nli` reads, so "I did not eat the cake" against "I ate the
cake" is contradiction at overlap 0.50, and six of the seven negation triples are
correct. What breaks it is word order: `lexical_overlap` compares *sets* of
content words, so the prediction is invariant to any permutation of the premise
or the hypothesis -- 400 of 400 shuffles leave the label unchanged. "The doctor
saw the lawyer" entails "The lawyer saw the doctor" at overlap 1.00, and so does
every other argument swap. Three of the five errors are that.

Negation is read as parity rather than scope, which is the fourth error.
`has_negation` counts `without`, so "Nobody left the building without a badge"
and "Somebody left without a badge" both carry a negation, the parities match,
and two negations cancel into entailment.

Two structural limits sit under all of it. The label is decided by two bits --
overlap at 0.5 and negation parity -- so **neutral is unreachable whenever
overlap reaches 0.5**: no hypothesis that reuses half the premise's content words
can be called neutral, whatever it says. And 13 of the 20 labels are recoverable
from the hypothesis alone, by answering contradiction when it contains a negation
and neutral otherwise: 65% of the classifier's 75% needs no premise at all, which
is the hypothesis-only artefact NLI benchmarks are known for.

Structure: `TRIPLES` is the labelled set, six entailment, seven contradiction,
seven neutral, with the exercise's own trap among them; `score` runs the lesson's
classifier over a set of rows; `shuffled` re-runs each row with both sides
permuted; `hypothesis_only` scores the premise-free rule.
"""

from __future__ import annotations

import importlib.util
import random
from collections import Counter

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "21-nli-textual-entailment"

UNAVAILABLE = ("transformers", "torch", "sentencepiece")
SHUFFLES = 20
TRIPLES = """A cat is sleeping on the couch.|There is a cat in the room.|entailment
John walked his dog in the park.|John has a dog.|entailment
The stock market rallied today.|Stocks went up today.|entailment
She finished the marathon in three hours.|She ran a marathon.|entailment
The chef served a tasty meal.|The chef served a meal.|entailment
Three children played football in the yard.|Children played football.|entailment
A cat is sleeping on the couch.|There is no cat in the room.|contradiction
John walked his dog in the park.|John has no dog.|contradiction
The stock market rallied today.|Stocks did not move today.|contradiction
The report was accurate.|The report was not accurate.|contradiction
I ate the cake.|I did not eat the cake.|contradiction
No student passed the exam.|A student passed the exam.|contradiction
Nobody left the building without a badge.|Somebody left without a badge.|contradiction
A cat is sleeping on the couch.|The dog chased the ball.|neutral
John walked his dog in the park.|John lives in New York.|neutral
Birds were singing outside the window.|The room was silent.|neutral
The chef served a tasty meal.|The chef trained in Paris.|neutral
The doctor saw the lawyer.|The lawyer saw the doctor.|neutral
The cat chased the dog.|The dog chased the cat.|neutral
John gave Mary the book.|Mary gave John the book.|neutral"""
ROWS = tuple(tuple(line.split("|")) for line in TRIPLES.splitlines())
SWAPS = ROWS[-3:]
NEGATION = tuple(row for row in ROWS if row[2] == "contradiction")


def score(ref, rows):
    """Correct count and the misses, as (gold, predicted, overlap, hypothesis)."""
    hits, misses = 0, []
    for premise, hypothesis, gold in rows:
        predicted, overlap = ref.predict_nli(premise, hypothesis)
        hits += predicted == gold
        if predicted != gold:
            misses.append((gold, predicted, round(overlap, 2), hypothesis))
    return hits, misses


def shuffled(ref, seed=0):
    """How often permuting both sides leaves the predicted label unchanged."""
    rng, stable = random.Random(seed), 0
    for premise, hypothesis, _ in ROWS:
        for _ in range(SHUFFLES):
            sides = []
            for text in (premise, hypothesis):
                words = text.split()
                rng.shuffle(words)
                sides.append(" ".join(words))
            stable += ref.predict_nli(*sides)[0] == ref.predict_nli(premise, hypothesis)[0]
    return stable


def hypothesis_only(ref):
    """The best premise-free rule: negation in the hypothesis, then the majority label."""
    groups = Counter((ref.has_negation(ref.tokenize(h)), gold) for _, h, gold in ROWS)
    best = Counter()
    for flag in (True, False):
        rows = {label: n for (seen, label), n in groups.items() if seen == flag}
        best[flag] = max(rows.values(), default=0)
    return sum(best.values())


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    hits, misses = score(ref, ROWS)
    overlap = {row: ref.lexical_overlap(ref.tokenize(row[0]), ref.tokenize(row[1])) for row in ROWS}
    return {
        "absent": [n for n in UNAVAILABLE if importlib.util.find_spec(n) is None],
        "n": len(ROWS),
        "hits": hits,
        "misses": misses,
        "classes": dict(Counter(gold for _, _, gold in ROWS)),
        "negation_hits": score(ref, NEGATION)[0],
        "negation_n": len(NEGATION),
        "swap_hits": score(ref, SWAPS)[0],
        "swap_n": len(SWAPS),
        "stable": shuffled(ref),
        "trials": len(ROWS) * SHUFFLES,
        "neutral_above": [row[1] for row in ROWS
                          if overlap[row] >= 0.5 and ref.predict_nli(row[0], row[1])[0] == "neutral"],
        "hyp_only": hypothesis_only(ref),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 15 of 20, and the trap the exercise names is not what breaks it",
            result["negation_hits"] > result["swap_hits"],
            f"{result['absent']} are all absent, so the classifier under test is the lesson's own "
            f"`predict_nli`: {result['hits']}/{result['n']} over {result['classes']}. The "
            f"subsequence trap it names is handled {result['negation_hits']}/"
            f"{result['negation_n']} -- negation is one of the two features it reads",
        ),
        practice.Check(
            "MECHANISM: what breaks it is word order, and completely",
            result["swap_hits"] == 0,
            f"argument swaps score {result['swap_hits']}/{result['swap_n']}: 'The doctor saw the "
            "lawyer' entails 'The lawyer saw the doctor' at overlap 1.00, and so does every other "
            f"swap. Three of the {len(result['misses'])} errors are this one class",
        ),
        practice.Check(
            "MECHANISM: because the prediction cannot see order at all",
            result["stable"] == result["trials"],
            f"`lexical_overlap` compares sets of content words, so permuting both sides leaves "
            f"the label unchanged in {result['stable']} of {result['trials']} shuffles. The "
            "classifier is a bag-of-words model wearing a two-sentence interface",
        ),
        practice.Check(
            "FINDING: negation is parity, not scope, so two of them cancel",
            any(m[3].startswith("Somebody") for m in result["misses"]),
            "`has_negation` counts `without`, so 'Nobody left the building without a badge' and "
            "'Somebody left without a badge' both carry one, the parities match, and the pair "
            "comes out entailment. A rule that reads negation as a flag cannot read its scope",
        ),
        practice.Check(
            "FINDING: neutral is unreachable once overlap reaches 0.5",
            not result["neutral_above"],
            "the label is two bits -- overlap at 0.5 and negation parity -- and both branches "
            f"above the threshold return entailment or contradiction, so "
            f"{len(result['neutral_above'])} of the 20 hypotheses that reuse half the premise's "
            "content words come out neutral. None can, whatever the hypothesis says",
        ),
        practice.Check(
            "CONTROL: two thirds of the labels need no premise",
            result["hyp_only"] >= result["hits"] - 3,
            f"answering contradiction when the hypothesis contains a negation and neutral "
            f"otherwise scores {result['hyp_only']}/{result['n']} without reading the premise, "
            f"against {result['hits']}/{result['n']} for the classifier. Most of the accuracy is "
            "the hypothesis-only artefact NLI benchmarks are known for",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
