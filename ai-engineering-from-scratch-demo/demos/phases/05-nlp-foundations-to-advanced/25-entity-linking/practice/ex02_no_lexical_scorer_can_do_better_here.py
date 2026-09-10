"""Exercise 2 — no lexical scorer can do better here.

    **Medium.** Encode 50 ambiguous mentions with a sentence transformer. Embed
    each candidate's description. Compare embedding-based disambiguation to
    Jaccard context overlap.

Reading of the exercise: `sentence_transformers` is absent, so the encoder arm
cannot be built. What can be built is the rest of the lexical family -- overlap
count, Dice, and coverage of the description -- and comparing those to Jaccard
answers the question the exercise is really asking, which is whether the scoring
function is where the accuracy comes from.

It is not. On exercise 1's quoting set all four scorers return **11 of 11**; the
choice is invisible. On the paraphrase set they land within one case of each
other: 5 or 6 with the prior, 6 or 7 without.

They agree because there is nothing to disagree about. On **8 of the 11**
paraphrase contexts, no candidate description shares a single content token with
the context once the mention's own name is removed -- every lexical score is
zero, every candidate ties, and the prior decides. That is a bound on the whole
family: any function of token overlap returns the same ranking on a case where
the overlap is empty for every candidate, whatever the function.

On the quoting set only 1 of 11 is in that state, which is why every scorer looks
excellent there.

The encoder is the only arm that could differ, and the eight cases say exactly
why: linking "the United Center" to *Chicago Bulls*, or "burst its banks" to
*Seine river*, requires knowing that those phrases are about the same thing.
Token overlap cannot represent that at any weighting.

The knowledge base caps the exercise before any of this. `ALIAS_INDEX` holds 5
aliases over 14 entities, so "50 ambiguous mentions" must reuse them: there are
five distinct disambiguation problems available, and every additional mention is
another sample of one of the five.

Structure: exercise 1's two labelled sets are loaded rather than copied;
`SCORERS` holds four functions over token sets; `accuracy` runs one scorer with
and without the prior; `blank` counts contexts where no candidate shares a
content token.
"""

from __future__ import annotations

import importlib.util
import pathlib

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "25-entity-linking"

EX1 = practice.load_module(
    pathlib.Path(__file__).resolve().parent / "ex01_the_contexts_quote_the_knowledge_base.py")
SETS = EX1.SETS
UNAVAILABLE = ("sentence_transformers", "transformers", "torch")
SCORERS = {
    "jaccard": lambda a, b: len(a & b) / len(a | b) if a | b else 0.0,
    "overlap count": lambda a, b: float(len(a & b)),
    "dice": lambda a, b: 2 * len(a & b) / (len(a) + len(b)) if a or b else 0.0,
    "coverage of description": lambda a, b: len(a & b) / len(b) if b else 0.0,
}


def accuracy(ref, scorer, cases, use_prior=True):
    """Correct links under one scoring function, with the lesson's own prior weighting."""
    hits = 0
    for mention, context, gold in cases:
        tokens = ref.tokenize(context)
        candidates = ref.ALIAS_INDEX[mention.lower()]
        best = max(candidates, key=lambda q: scorer(tokens, ref.tokenize(ref.KB_DESC[q]))
                   + (0.1 * ref.PRIORS[q] if use_prior else 0.0))
        hits += best == gold
    return hits


def content_overlap(ref, cases):
    """Content tokens each candidate shares with the context, the mention name removed."""
    rows = []
    for mention, context, _ in cases:
        tokens = ref.tokenize(context) - ref.tokenize(mention)
        rows.append([len(tokens & ref.tokenize(ref.KB_DESC[q]))
                     for q in ref.ALIAS_INDEX[mention.lower()]])
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {}
    for name, cases in SETS.items():
        overlaps = content_overlap(ref, cases)
        rows[name] = {
            "n": len(cases),
            "prior": {s: accuracy(ref, fn, cases) for s, fn in SCORERS.items()},
            "flat": {s: accuracy(ref, fn, cases, use_prior=False) for s, fn in SCORERS.items()},
            "blank": sum(1 for row in overlaps if max(row) == 0),
            "overlaps": [max(row) for row in overlaps],
        }
    return {
        "absent": [n for n in UNAVAILABLE if importlib.util.find_spec(n) is None],
        "rows": rows,
        "aliases": len(ref.ALIAS_INDEX),
        "entities": len(ref.KB_DESC),
        "asked": 50,
    }


def verify(result):
    quoting, para = result["rows"]["quoting"], result["rows"]["paraphrase"]
    spread = max(para["prior"].values()) - min(para["prior"].values())
    return [
        practice.Check(
            "ANSWER: on the lesson's own cases every lexical scorer is perfect",
            len(set(quoting["prior"].values())) == 1,
            f"{result['absent']} are all absent, so the encoder arm cannot be built. The rest of "
            f"the family can: {quoting['prior']} on the quoting set -- {list(SCORERS)} all return "
            f"{max(quoting['prior'].values())}/{quoting['n']}, so the choice of scoring function "
            "is invisible there",
        ),
        practice.Check(
            "MECHANISM: and on paraphrased contexts they land within one case of each other",
            spread <= 1,
            f"with the prior {para['prior']}, without it {para['flat']}: a spread of {spread} "
            f"case over {para['n']}. Swapping Jaccard for any other token-overlap function moves "
            "less than the prior does",
        ),
        practice.Check(
            "FINDING: because on 8 of 11 no candidate shares a content token at all",
            para["blank"] > quoting["blank"] * 4,
            f"removing the mention's own name, the best content overlap across all candidates is "
            f"{para['overlaps']} on the paraphrase set -- {para['blank']} of {para['n']} cases "
            f"where every candidate scores zero -- against {quoting['blank']} of {quoting['n']} "
            "on the quoting set",
        ),
        practice.Check(
            "MECHANISM: which bounds the whole family, not just Jaccard",
            para["blank"] > para["n"] / 2,
            "any function of token overlap returns the same ranking when the overlap is empty for "
            f"every candidate, whatever the function. On {para['blank']} of {para['n']} contexts "
            "the answer is the prior's, so those cases are decided before a scorer is chosen",
        ),
        practice.Check(
            "FINDING: the encoder is the only arm that could differ, and the same 8 say why",
            para["blank"] >= 8,
            "linking 'the United Center' to *Chicago Bulls*, or 'burst its banks' to *Seine "
            "river*, means knowing those phrases are about one thing. That is the comparison the "
            "exercise wants and the one arm of it that is not installed",
        ),
        practice.Check(
            "CONTROL: the knowledge base caps the exercise before any of this",
            result["aliases"] * 10 < result["asked"] * 2,
            f"`ALIAS_INDEX` holds {result['aliases']} aliases over {result['entities']} entities, "
            f"so {result['asked']} ambiguous mentions must reuse them. There are "
            f"{result['aliases']} distinct disambiguation problems available and every extra "
            "mention is another sample of one of them",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
