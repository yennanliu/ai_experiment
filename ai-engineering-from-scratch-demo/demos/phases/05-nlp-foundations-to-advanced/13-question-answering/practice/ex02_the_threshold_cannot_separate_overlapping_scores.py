"""Exercise 2 — the threshold cannot separate overlapping scores.

    **Medium.** Add a refusal classifier. When the top retrieval score is below a
    threshold (say 0.3 cosine), return "I don't know" instead of calling the
    reader. Tune the threshold on a held-out set.

Reading of the exercise: 0.3 is a cosine and the score it is compared against is
not one. `refusal_from_score` takes whatever `toy_retrieve` returns, and
`toy_bm25_score` sums `count / (1 + length / 10)` over matching terms, which on
exercise 1's ten passages runs from 0.0 to 2.4000. A cosine cannot exceed 1; this
number routinely does. At 0.3 the rule fires only when the score is exactly zero,
so what the lesson calls a confidence threshold is a test for no shared token at
all.

Tuning it, which the exercise asks for next, is worth one question. Sweeping
every threshold the data can distinguish, the best accuracy over ten answerable
and six unanswerable questions is 0.8125 at 0.8696, against 0.7500 at the
lesson's 0.3 -- one item out of sixteen. The reason it stops there is that the
two score distributions overlap: answerable questions score 0.8696 to 2.4000 and
unanswerable ones 0.0 to 1.6667, so 2 of the 6 unanswerable score above the
lowest answerable and 4 of the 10 answerable score below the highest
unanswerable. No threshold on this signal separates them, and the tuning step
the exercise describes has a ceiling it reaches immediately.

Which unanswerable questions get through is the useful part. The three that
score highest are the three written to share vocabulary with the corpus -- a
quantum computer from Apple, a Linux division at Nokia, Guido van Rossum on the
Nintendo Switch. A lexical retriever scores them like real questions because
lexically they are real questions, and a refusal rule reading that score cannot
be told otherwise.

Structure: `UNANSWERABLE` are six questions the corpus cannot answer, three of
them sharing terms with it on purpose. `sweep` scores every candidate threshold
-- the observed scores themselves, which are the only points where the decision
can change -- and `overlap` counts the crossings between the two distributions.
"""

from __future__ import annotations

import pathlib

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "13-question-answering"

SIBLING = "ex01_two_answers_made_of_articles_match_exactly.py"
DEFAULT = 0.3
UNANSWERABLE = (
    ("What is the capital of Peru?", "unrelated"),
    ("How many moons does Neptune have?", "unrelated"),
    ("zzz qqq", "unrelated"),
    ("When did Apple release the first quantum computer?", "shares vocabulary"),
    ("Who founded Nokia's Linux handset division?", "shares vocabulary"),
    ("What did Guido van Rossum say about the Nintendo Switch?", "shares vocabulary"),
)


def kinds(ref) -> dict:
    """The unanswerable questions split by whether they were written to share vocabulary."""
    return {kind.replace(" vocabulary", ""): [(q, round(top_score(ref, q), 4))
                                              for q, k in UNANSWERABLE if k == kind]
            for kind in ("shares vocabulary", "unrelated")}


def top_score(ref, question) -> float:
    return ref.toy_retrieve(question, top_k=1)[0][0]


def accuracy(answerable, unanswerable, threshold) -> tuple:
    answered = sum(score >= threshold for score in answerable)
    refused = sum(score < threshold for score in unanswerable)
    total = len(answerable) + len(unanswerable)
    return round((answered + refused) / total, 4), answered, refused


def crossings(answerable, unanswerable) -> dict:
    """Both score distributions, the tuning sweep, and how far they overlap."""
    ranked, every = sweep(answerable, unanswerable), answerable + unanswerable
    return {"answerable": [round(s, 4) for s in answerable],
            "unanswerable": [round(s, 4) for s in unanswerable],
            "range": (round(min(every), 4), round(max(every), 4)),
            "best": ranked[0], "ranked": ranked[:3],
            "fires_at_zero": sum(s < DEFAULT for s in every) == sum(s == 0.0 for s in every),
            "above_lowest": sum(u > min(answerable) for u in unanswerable),
            "below_highest": sum(a < max(unanswerable) for a in answerable)}


def sweep(answerable, unanswerable) -> list:
    candidates = sorted({0.0, *answerable, *unanswerable})
    return sorted(((accuracy(answerable, unanswerable, t)[0], round(t, 4)) for t in candidates),
                  reverse=True)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sibling = practice.load_module(pathlib.Path(__file__).with_name(SIBLING))
    ref.CORPUS = list(sibling.PASSAGES)
    answerable = [top_score(ref, question) for question, _ in sibling.QUESTIONS]
    unanswerable = [top_score(ref, question) for question, _ in UNANSWERABLE]
    return dict(kinds(ref), **crossings(answerable, unanswerable),
                default=accuracy(answerable, unanswerable, DEFAULT),
                sizes=(len(answerable), len(unanswerable)))


def verify(result):
    best, default, span = result["best"], result["default"], result["range"]
    shared_scores = [score for _, score in result["shares"]]
    unrelated = [score for _, score in result["unrelated"]]
    return [
        practice.Check(
            "ANSWER: 0.3 is a cosine and the score is a BM25 sum that reaches 2.4",
            span[1] > 1.0 and result["fires_at_zero"],
            f"`toy_bm25_score` sums count/(1 + length/10) over matching terms, so on the ten "
            f"passages it runs {span[0]} to {span[1]} -- a cosine cannot exceed 1. At the lesson's "
            f"{DEFAULT} the rule fires only where the score is exactly zero, so the confidence "
            f"threshold is a test for no shared token at all"),
        practice.Check(
            "FINDING: tuning it is worth one question out of sixteen",
            best[0] > default[0] and best[0] - default[0] < 0.1,
            f"sweeping every threshold the data can distinguish, the best accuracy is {best[0]} at "
            f"{best[1]} against {default[0]} at {DEFAULT} -- over "
            f"{result['sizes'][0]} answerable and {result['sizes'][1]} unanswerable questions, one "
            f"item. The top three settings are {result['ranked']}"),
        practice.Check(
            "MECHANISM: the two score distributions overlap, so no threshold separates them",
            result["above_lowest"] > 0 and result["below_highest"] > 0,
            f"answerable questions score {sorted(result['answerable'])} and unanswerable ones "
            f"{sorted(result['unanswerable'])}. {result['above_lowest']} of "
            f"{result['sizes'][1]} unanswerable score above the lowest answerable and "
            f"{result['below_highest']} of {result['sizes'][0]} answerable score below the highest "
            f"unanswerable. Tuning has a ceiling and reaches it immediately"),
        practice.Check(
            "FINDING: the ones that get through are the ones written to share vocabulary",
            min(shared_scores) > min(unrelated),
            f"the three unanswerable questions built from corpus terms score "
            f"{shared_scores}; the three unrelated ones score {unrelated}. A "
            f"lexical retriever scores a fluent question about Apple and quantum computers like a "
            f"real question, because lexically it is one"),
        practice.Check(
            "MECHANISM: so the signal the rule reads is overlap, not answerability",
            max(shared_scores) > min(result["answerable"]),
            f"the highest-scoring unanswerable question reaches "
            f"{max(shared_scores)} where the lowest-scoring answerable one reaches "
            f"{min(result['answerable'])}. The rule is being asked to distinguish two things the "
            f"score does not distinguish, and no amount of held-out data changes what the score is "
            f"measuring"),
        practice.Check(
            "CONTROL: the default does refuse the questions with no overlap at all",
            default[2] == len([s for s in unrelated if s == 0.0]),
            f"at {DEFAULT} the rule refuses {default[2]} of {result['sizes'][1]} unanswerable "
            f"questions and answers all {default[1]} answerable ones. It is not broken -- it is a "
            f"zero-overlap detector doing exactly that, described in the exercise as a confidence "
            f"threshold with a tunable value"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
