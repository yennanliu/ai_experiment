"""Exercise 3 — the three numbers rank three different ways.

    **Hard.** Compare BART-large-CNN against an LLM (Claude or GPT-4) on 50
    CNN/DailyMail articles. Report ROUGE-L, factuality (by entity F1), and cost
    per summary. Document where each wins.

Reading of the exercise: neither system is reachable -- transformers and torch
are absent and no API key is set -- so the three systems compared here are ones
that can be built: the lesson's `textrank` at k=2, at k=5, and an abstractive
stand-in that rewrites each article into one sentence, sometimes getting an
entity wrong. What the exercise asks for is not a score but a verdict, "document
where each wins", and the three numbers it names produce three different
verdicts on the same nine summaries.

ROUGE-L puts the abstractive stand-in first, then k=5, then k=2. Entity F1 puts
k=5 first, and for a reason that is not about quality: extractive precision is
pinned at 1.0 (exercise 2), so the gap between the two extractive systems is
entirely recall and the longer one wins because it is longer. Cost, taken as mean
output words, puts the abstractive system first and then reverses the two
extractive ones -- k=2 is cheaper than k=5 and scores lower on ROUGE-L, so those
two axes disagree about the two systems the lesson actually builds.

Three axes, three distinct orderings, no system first on all of them and one
first on none. "Document where each wins" is answered by saying that the
exercise's own three numbers do not agree on a winner.

The fourth number the exercise does not ask for is the one that would settle it.
Every extractive summary here is a subset of its source, so none of the three
metrics can distinguish "said something false" from "said less" -- and the
abstractive system, the only one that can be false, is the only one any of them
penalise.

Structure: the articles, `rouge_l` and the entity extractor are imported from
exercises 1 and 2. `SYSTEMS` maps a name to a summariser; `evaluate` scores one
system on all five articles and returns the three quantities the exercise names.
"""

from __future__ import annotations

import importlib.util
import pathlib

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "12-text-summarization"

FIRST, SECOND = ("ex01_recall_only_rouge_is_maximised_by_the_article.py",
                 "ex02_extractive_precision_is_one_by_construction.py")
UNAVAILABLE = ("transformers", "torch", "openai", "anthropic")
# one hand-written abstractive summary per article, the first with an invented entity
ABSTRACTIVE = (
    "Researchers at MIT released a linear-time attention variant matching standard attention.",
    "The council approved three new bus routes costing 40 million dollars from March.",
    "A storm cut power to 12000 homes and cancelled ferries with no injuries.",
    "The company reported 2 billion dollars revenue up 14 percent with a 30 cent dividend.",
    "A trial of 300 patients found the treatment lowered pain scores after eight weeks.",
)


def systems(ref):
    return {"extractive k=2": lambda a, i: " ".join(ref.textrank(a, top_k=2)),
            "extractive k=5": lambda a, i: " ".join(ref.textrank(a, top_k=5)),
            "abstractive": lambda a, i: ABSTRACTIVE[i]}


def evaluate(ref, first, second, make) -> dict:
    rouge, precision, recall, words = [], [], [], []
    for index, (article, reference) in enumerate(first.ARTICLES):
        summary = make(article, index)
        rouge.append(first.rouge_l(summary, reference))
        scored = second.factuality(ref, summary, article)
        precision.append(scored["precision"])
        recall.append(scored["recall"])
        words.append(len(summary.split()))
    mean = lambda rows: round(sum(rows) / len(rows), 4)                           # noqa: E731
    p, r = mean(precision), mean(recall)
    return {"rouge_l": mean(rouge), "precision": p, "recall": r, "words": mean(words),
            "entity_f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def rank(rows, key, reverse=True) -> list:
    return sorted(rows, key=lambda name: rows[name][key], reverse=reverse)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    here = pathlib.Path(__file__)
    first = practice.load_module(here.with_name(FIRST))
    second = practice.load_module(here.with_name(SECOND))
    rows = {name: evaluate(ref, first, second, make) for name, make in systems(ref).items()}
    scored = second.factuality(ref, ABSTRACTIVE[0], first.ARTICLES[0][0])
    return {
        "unavailable": [m for m in UNAVAILABLE if importlib.util.find_spec(m) is None],
        "rows": rows, "articles": len(first.ARTICLES),
        "rankings": {"rouge_l": rank(rows, "rouge_l"), "entity_f1": rank(rows, "entity_f1"),
                     "cost": rank(rows, "words", reverse=False)},
        "invented": scored["invented"],
        "kept": sorted(set(scored["found"]) - set(scored["invented"])),
        "swap": (rows["extractive k=5"]["rouge_l"] > rows["extractive k=2"]["rouge_l"]
                 and rows["extractive k=5"]["words"] > rows["extractive k=2"]["words"]),
        "wins": {name: sum(order[0] == name for order in
                           (rank(rows, "rouge_l"), rank(rows, "entity_f1"),
                            rank(rows, "words", reverse=False))) for name in rows},
    }


def verify(result):
    rows, rankings = result["rows"], result["rankings"]
    names = sorted(rows)
    return [
        practice.Check(
            "ANSWER: the three numbers produce three different orderings of the same summaries",
            len({tuple(order) for order in rankings.values()}) == 3,
            f"{result['unavailable']} are all absent, so the three systems are the lesson's textrank "
            f"at k=2 and k=5 and a hand-written abstractive stand-in, scored on "
            f"{result['articles']} articles. ROUGE-L ranks them {rankings['rouge_l']}, entity F1 "
            f"{rankings['entity_f1']}, and cost in output words {rankings['cost']} -- no two agree"),
        practice.Check(
            "MECHANISM: entity F1 ranks the extractive systems on recall alone",
            rows["extractive k=2"]["precision"] == rows["extractive k=5"]["precision"] == 1.0,
            f"both extractive systems score precision {rows['extractive k=5']['precision']}, so "
            f"their F1 gap is entirely recall: {rows['extractive k=2']['recall']} against "
            f"{rows['extractive k=5']['recall']}. On this axis the longer summary wins because it "
            f"is longer, which is the same reason it won the last one"),
        practice.Check(
            "FINDING: the abstractive system is penalised for the entity it invented",
            rows["abstractive"]["precision"] < 1.0 and result["invented"],
            f"on the first article it invented {result['invented']} and carried over "
            f"{result['kept']} of the source's entities, which takes the mean precision over the "
            f"five to {rows['abstractive']['precision']} and its entity F1 to "
            f"{rows['abstractive']['entity_f1']}. It is the only one of the three any of these "
            f"metrics can mark down, because it is the only one that can be wrong"),
        practice.Check(
            "FINDING: the two extractive systems swap places between ROUGE-L and cost",
            result["swap"],
            f"mean output length is "
            f"{ {name: rows[name]['words'] for name in names} } words, and ROUGE-L reads "
            f"{ {name: rows[name]['rouge_l'] for name in names} }. k=5 is better on ROUGE-L and "
            f"dearer on cost than k=2, so between the two systems the lesson actually builds, those "
            f"two axes give opposite answers"),
        practice.Check(
            "MECHANISM: so 'where each wins' is settled by which of its own numbers you read",
            max(result["wins"].values()) < len(rankings) and min(result["wins"].values()) == 0,
            f"first places by axis: {result['wins']}. No system tops all three and one tops none, "
            f"over the same {result['articles']} articles and the same nine summaries. The exercise "
            f"asks which system wins where; the answer is that its three numbers do not agree on a "
            f"winner, let alone on an order"),
        practice.Check(
            "CONTROL: none of the three metrics can see a false statement that keeps its entities",
            rows["extractive k=2"]["precision"] == 1.0 == rows["extractive k=5"]["precision"],
            f"every extractive summary is a subset of its source, so it cannot state anything the "
            f"source did not; the abstractive one can, and the only part of it these metrics read "
            f"is its entity list. A summary that reversed a claim while reusing every name would "
            f"score {rows['extractive k=5']['precision']} on factuality, exactly like the copied "
            f"sentences"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
