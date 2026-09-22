"""Exercise 1 — the two riskiest assumptions tie, and the two functions disagree.

    Write five assumptions for a feature you want to build.

Reading of the exercise: five statements, one per class, each falsifiable and
each scored. The feature is one this repository could plausibly build --
generating the README answers section instead of writing it -- so the scores
have to be defended rather than invented.

**ANSWER: the five score 17, 13, 16, 17 and 14, and the top two tie.** Value
and viability both reach **17**: "a generated section saves more time than
reviewing it costs" and "the generator stays correct as the harness changes".
`prioritize` breaks the tie alphabetically and `next_experiment` breaks it by
list position, so the two functions in the same module can name different
experiments from the same input.

**FINDING: `max` and `sorted` disagree whenever two open assumptions tie.**
`prioritize` sorts by `(-risk_score, statement)` and `next_experiment` calls
`max(..., key=risk_score)`, which returns the first maximum it meets. On a
two-item fixture that ties at **17**, the ranked list puts one first and the
next experiment names the other -- a JSON document whose first row is not the
thing it says to do next.

**FINDING: the score caps irreversibility at 5 and lets the product reach
25.** `impact * uncertainty + irreversibility` means a fully irreversible bet
(irreversibility **5**) loses to any pair whose product is **5** higher.
Among these five, the only one that cannot be undone scores **14** and ranks
**fourth** of five, because its impact and uncertainty are modest.

**FINDING: `evidence` is a string, so "tested" means "somebody wrote
something".** Marking the feasibility assumption with the word "unclear"
flips its status from `open` to `tested` and removes it from
`next_experiment`'s pool -- **1** fewer candidate -- with no threshold, no
result and no comparison anywhere in the module.

Structure: `ASSUMPTIONS` is the map; `ranked()` runs the lesson's own
prioritisation; `tie()` is the two-item disagreement.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "49-map-assumptions-and-risk"

# (class, statement, impact, uncertainty, irreversibility, test)
ASSUMPTIONS = [
    ("value", "A generated answers section saves more time than reviewing it costs",
     4, 4, 1, "Generate one lesson's answers and time the review against writing them"),
    ("usability", "A reader cannot tell a generated answer from a written one",
     3, 4, 1, "Show five readers three answers of each kind and ask them to sort"),
    ("feasibility", "Every number in an answer can be sourced from a graded check detail",
     5, 3, 1, "Extract the numbers from one finished lesson and match them against its "
     "graded details"),
    ("viability", "The generator stays correct as the harness changes",
     3, 5, 2, "Re-run last month's generated answers against the current harness"),
    ("safety", "A wrong generated number is caught before the lesson ships",
     3, 3, 5, "Plant three wrong numbers and see whether the audit refuses them"),
]


def build(ref, rows=ASSUMPTIONS, evidence=None):
    return [ref.Assumption(statement, impact, uncertainty, irreversibility, test,
                           (evidence or {}).get(name, ""))
            for name, statement, impact, uncertainty, irreversibility, test in rows]


def ranked(ref, items):
    return [(row["statement"][:24], row["risk_score"], row["status"])
            for row in ref.prioritize(items)]


def tie(ref):
    """Two open assumptions with the same score, one named by each function."""
    rows = [("a", "Zebra: the harness keeps its shape", 4, 4, 1, "replay"),
            ("b", "Alpha: reviewers read what is generated", 4, 4, 1, "survey")]
    items = build(ref, rows)
    return (ref.prioritize(items)[0]["statement"].split(":")[0],
            ref.next_experiment(items).statement.split(":")[0])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    items = build(ref)
    scores = [ref.risk_score(item) for item in items]
    tested = build(ref, evidence={"feasibility": "unclear"})
    module = inspect.getsource(ref)
    irreversible = max(items, key=lambda item: item.irreversibility)
    return {
        "count": len(items), "scores": scores,
        "top": max(scores), "tied": scores.count(max(scores)),
        "ranked": ranked(ref, items),
        "next": ref.next_experiment(items).statement[:24],
        "first": ranked(ref, items)[0][0],
        "tie_first": tie(ref)[0], "tie_next": tie(ref)[1],
        "formula": "item.impact * item.uncertainty + item.irreversibility" in module,
        "max_product": 25, "max_irreversibility": 5,
        "irreversible_score": ref.risk_score(irreversible),
        "irreversible_rank": [row[1] for row in ranked(ref, items)].index(
            ref.risk_score(irreversible)) + 1,
        "open_before": len([item for item in items if not item.evidence]),
        "open_after": len([item for item in tested if not item.evidence]),
        "tested_status": ref.prioritize(tested)[
            [row["statement"] for row in ref.prioritize(tested)].index(
                ASSUMPTIONS[2][1])]["status"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the five score 17, 13, 16, 17 and 14, and the top two tie",
            all([result["count"] == 5, result["scores"] == [17, 13, 16, 17, 14],
                 result["top"] == 17, result["tied"] == 2]),
            f"the five assumptions score {result['scores']}; the top is {result['top']} and "
            f"{result['tied']} of them reach it, so the map's first row and its next "
            "experiment are decided by a tie-break rather than by risk",
        ),
        practice.Check(
            "FINDING: max and sorted disagree whenever two open assumptions tie",
            all([result["tie_first"] == "Alpha", result["tie_next"] == "Zebra"]),
            f"on a two-item fixture tying at 17, prioritize puts {result['tie_first']!r} "
            f"first and next_experiment names {result['tie_next']!r} -- a document whose "
            "first row is not the thing it says to do next",
        ),
        practice.Check(
            "FINDING: the score caps irreversibility at 5 and lets the product reach 25",
            all([result["formula"] is True, result["max_product"] == 25,
                 result["max_irreversibility"] == 5,
                 result["irreversible_score"] == 14, result["irreversible_rank"] == 4]),
            f"impact times uncertainty reaches {result['max_product']} while "
            f"irreversibility adds at most {result['max_irreversibility']}, so the only "
            f"assumption here that cannot be undone scores "
            f"{result['irreversible_score']} and ranks {result['irreversible_rank']}",
        ),
        practice.Check(
            "FINDING: evidence is a string, so tested means somebody wrote something",
            all([result["tested_status"] == "tested", result["open_before"] == 5,
                 result["open_after"] == 4]),
            f"marking one assumption with the word 'unclear' flips it to "
            f"{result['tested_status']!r} and drops the open pool from "
            f"{result['open_before']} to {result['open_after']}, with no threshold and no "
            "comparison anywhere in the module",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
