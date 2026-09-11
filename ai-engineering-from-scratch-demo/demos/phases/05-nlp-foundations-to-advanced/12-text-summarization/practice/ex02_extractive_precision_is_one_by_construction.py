"""Exercise 2 — extractive precision is one by construction.

    **Medium.** Implement entity-level factuality: extract named entities from
    source and summary (spaCy), compute recall of source entities in summary and
    precision of summary entities against source. High precision and low recall
    mean safe but terse; low precision means hallucinated entities.

Reading of the exercise: half the metric cannot move. `textrank` returns
sentences copied from the source, so every entity in its output is an entity of
the source and precision is 1.0000 -- not usually, not on this data, but for
every k and every article, checked here as a set-inclusion over the whole sweep.
The exercise's reading of that number ("high precision and low recall mean safe
but terse") describes every extractive summary at every length, including the
one that returns the entire article.

Recall is the half that carries information, and it carries the same information
`rouge_n` did in exercise 1: it rises with k and is maximised by copying
everything. Averaged over the five articles it goes 0.1067, 0.2800, 0.4367,
0.4867, 0.6767 and 1.0000 at k = 1 to 7. So both halves of the exercise's
factuality metric reduce, for an extractive system, to "longer is better" -- the
same instruction the recall-only ROUGE gave.

The metric is not useless; it is inapplicable to the system the lesson builds. A
three-word edit to one summary -- MIT for the Canadian university, 2 billion for
1, GitLab for GitHub -- takes precision from 1.0000 to 0.0000, which is exactly
the signal the exercise describes and exactly the one an extractive summariser
can never produce. Choosing entity-level factuality as the evaluation is
choosing a metric that only reads abstractive failures.

spaCy is not installed, so entities are capitalised tokens in non-initial
position plus anything starting with a digit. That is a weaker extractor than a
model, and it does not matter here: the precision result is a statement about set
inclusion that holds for any extractor applied to both sides.

Structure: `entities` is the extractor, `factuality` scores one summary against
its source, and `sweep` runs the lesson's own `textrank` at each k over exercise
1's five articles. `HALLUCINATED` is the abstractive counter-example.
"""

from __future__ import annotations

import pathlib
import re

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "12-text-summarization"

SIBLING = "ex01_recall_only_rouge_is_maximised_by_the_article.py"
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d[\d.]*")
TOPS = (1, 2, 3, 4, 5, 7)
HALLUCINATED = ("Researchers at MIT introduced a linear attention variant trained to 2 billion "
                "parameters and released it on GitLab.")


def entities(ref, text) -> set:
    """Capitalised tokens in non-initial position, plus anything starting with a digit."""
    found = set()
    for sentence in ref.sentence_split(text):
        for index, token in enumerate(TOKEN.findall(sentence)):
            if token[0].isdigit() or (token[0].isupper() and index > 0):
                found.add(token)
    return found


def factuality(ref, summary, source) -> dict:
    got, want = entities(ref, summary), entities(ref, source)
    return {"precision": round(len(got & want) / len(got), 4) if got else 1.0,
            "recall": round(len(got & want) / len(want), 4) if want else 1.0,
            "invented": sorted(got - want), "found": sorted(got)}


def sweep(ref, articles) -> dict:
    rows = {}
    for top_k in TOPS:
        scored = [factuality(ref, " ".join(ref.textrank(article, top_k=top_k)), article)
                  for article, _ in articles]
        rows[top_k] = {
            "precision": round(sum(r["precision"] for r in scored) / len(scored), 4),
            "recall": round(sum(r["recall"] for r in scored) / len(scored), 4),
            "subset": all(not r["invented"] for r in scored)}
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    articles = practice.load_module(pathlib.Path(__file__).with_name(SIBLING)).ARTICLES
    rows = sweep(ref, articles)
    source = articles[0][0]
    return {
        "rows": rows, "articles": len(articles),
        "precision": [rows[k]["precision"] for k in TOPS],
        "recall": [rows[k]["recall"] for k in TOPS],
        "always_subset": all(rows[k]["subset"] for k in TOPS),
        "source_entities": sorted(entities(ref, source)),
        "abstractive": factuality(ref, HALLUCINATED, source),
        "terse": factuality(ref, " ".join(ref.textrank(source, top_k=1)), source),
        "whole": factuality(ref, source, source),
    }


def verify(result):
    rows, precision, recall = result["rows"], result["precision"], result["recall"]
    abstractive, terse, whole = result["abstractive"], result["terse"], result["whole"]
    return [
        practice.Check(
            "ANSWER: precision is 1.0000 at every k, on every article, by construction",
            set(precision) == {1.0} and result["always_subset"],
            f"`textrank` returns sentences copied from the source, so the summary's entities are a "
            f"subset of the source's at every k in {list(TOPS)} across "
            f"{result['articles']} articles -- checked as set inclusion, not as an average. Half "
            f"the metric the exercise defines cannot move"),
        practice.Check(
            "MECHANISM: the exercise's reading of that number describes every extractive summary",
            terse["precision"] == whole["precision"] == 1.0,
            f"'high precision and low recall mean safe but terse' -- the one-sentence summary scores "
            f"precision {terse['precision']} with recall {terse['recall']}, and the whole article "
            f"scores precision {whole['precision']} with recall {whole['recall']}. The diagnosis "
            f"reads the same for the tersest possible output and for no summarisation at all"),
        practice.Check(
            "FINDING: recall is the informative half, and it says what rouge_n said",
            recall == sorted(recall) and recall[-1] == 1.0,
            f"entity recall over the five articles is {dict(zip(TOPS, recall))} at k = {list(TOPS)}, "
            f"reaching 1.0000 only by copying the article. Both halves of the factuality metric "
            f"reduce, for an extractive system, to the same instruction exercise 1's recall-only "
            f"ROUGE gave: emit more"),
        practice.Check(
            "FINDING: the signal the exercise describes needs an abstractive system to produce it",
            abstractive["precision"] == 0.0 and abstractive["invented"],
            f"swapping three entities -- {abstractive['invented']} for "
            f"{result['source_entities']} -- takes precision from 1.0000 to "
            f"{abstractive['precision']}. That is exactly the 'low precision means hallucinated "
            f"entities' case, and exactly the case a summariser that copies sentences cannot "
            f"reach"),
        practice.Check(
            "MECHANISM: so the metric is inapplicable rather than uninformative",
            abstractive["precision"] < min(precision),
            f"entity precision separates the abstractive summary ({abstractive['precision']}) from "
            f"every extractive one (all {min(precision)}) perfectly. Choosing it as the evaluation "
            f"for the system this lesson builds is choosing a metric that only reads a failure mode "
            f"the system does not have"),
        practice.Check(
            "CONTROL: the weak extractor does not affect the precision result",
            result["source_entities"] and result["always_subset"],
            f"spaCy is absent, so entities are capitalised non-initial tokens plus digits: "
            f"{result['source_entities']} on the first article. A better extractor would find more "
            f"of them on both sides, and the subset relation -- which is what pins precision at 1 -- "
            f"holds for any extractor applied to both the summary and its own source"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
