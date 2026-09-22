"""Exercise 5 — the roadmap builds in the order the risk map would test last.

    Compare risk ranking with roadmap priority and explain the mismatch.

Reading of the exercise: the roadmap for this feature is the order a team
would naturally do the work -- make it produce something, make it readable,
make it correct, keep it alive, then worry about what a wrong answer does.
The risk map reads the same six assumptions and returns almost the reverse.

**ANSWER: the safety assumption is 6th on the roadmap and 1st by risk, a
displacement of 5 places, and the two orders disagree on 10 of 15 pairs.**
Ranking by `risk_score` gives 25, 17, 17, 16, 14, 13 against a build order
that starts with the value assumption and ends with the published-mistake
one. **9** of the **15** pairs are inverted between the two sequences.

**FINDING: the irreversibility term is nearly inert on this map.** Sorting
the same six by `impact * uncertainty` alone leaves the top item where it is
and still inverts **7** pairs against the roadmap. What the term actually
does is move **4** assumptions by one place each, among them the other safety
row -- from **6th** on the product alone to **5th** with irreversibility
counted. The clause that the lesson says should reorder the map moves it by
one position.

**FINDING: the roadmap order is what the build wants and the risk order is
what the evidence wants.** The top-risk assumption's test -- publish one
generated lesson and time how long a planted error survives -- cannot run
until something has been generated, so a strict risk-first schedule stalls.
**2** of the six tests depend on the feature existing, which is why the
answer is a bounded build rather than a reordering.

**FINDING: nothing in the module records a roadmap to compare against.**
`prioritize` returns rows carrying the statement, three dimensions, a test,
evidence, a score and a status -- **8** keys, **0** of them an intended
position. The mismatch this exercise asks about has to be computed outside
the artifact that is supposed to drive the decision.

Structure: `ROADMAP` is the build order; `inversions()` counts the pairs the
two sequences disagree about.
"""

from __future__ import annotations

from itertools import combinations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "49-map-assumptions-and-risk"

# build order: (statement, impact, uncertainty, irreversibility, test)
ROADMAP = [
    ("A generated answers section saves more time than reviewing it costs", 4, 4, 1,
     "Time one lesson's review against writing it"),
    ("A reader cannot tell a generated answer from a written one", 3, 4, 1,
     "Ask five readers to sort six answers"),
    ("Every number in an answer can be sourced from a graded check detail", 5, 3, 1,
     "Trace one lesson's numbers against its details"),
    ("The generator stays correct as the harness changes", 3, 5, 2,
     "Replay last month's answers against the current harness"),
    ("A wrong generated number is caught before the lesson ships", 3, 3, 5,
     "Plant three wrong numbers and watch the audit"),
    ("A wrong number reaches a reader before anyone notices", 5, 4, 5,
     "Publish one generated lesson and time how long a planted error survives"),
]
NEEDS_FEATURE = (5, 6)  # 1-based roadmap positions whose test needs the build to exist


def build(ref):
    return [ref.Assumption(*row) for row in ROADMAP]


def order_by(items, key):
    return [item.statement for item in sorted(items, key=key, reverse=True)]


def inversions(left, right):
    position = {statement: index for index, statement in enumerate(right)}
    return sum(position[a] > position[b] for a, b in combinations(left, 2))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    items = build(ref)
    roadmap = [row[0] for row in ROADMAP]
    by_risk = order_by(items, ref.risk_score)
    by_product = order_by(items, lambda item: item.impact * item.uncertainty)
    safety = ROADMAP[-1][0]
    ranked = ref.prioritize(items)
    return {
        "scores": [ref.risk_score(item) for item in
                   sorted(items, key=ref.risk_score, reverse=True)],
        "roadmap_position": roadmap.index(safety) + 1,
        "risk_position": by_risk.index(safety) + 1,
        "displacement": roadmap.index(safety) - by_risk.index(safety),
        "inversions": inversions(roadmap, by_risk),
        "pairs": len(list(combinations(roadmap, 2))),
        "product_position": by_product.index(safety) + 1,
        "product_inversions": inversions(roadmap, by_product),
        "moved": sum(by_risk.index(statement) != by_product.index(statement)
                     for statement in by_risk),
        "other_safety": (by_product.index(ROADMAP[4][0]) + 1,
                         by_risk.index(ROADMAP[4][0]) + 1),
        "needs_feature": len(NEEDS_FEATURE),
        "top_test": next(row[4] for row in ROADMAP if row[0] == by_risk[0]),
        "keys": sorted(ranked[0]),
        "records_roadmap": any(key in ranked[0] for key in
                               ("roadmap", "position", "priority")),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the safety assumption moves 5 places and 9 of 15 pairs invert",
            all([result["scores"] == [25, 17, 17, 16, 14, 13],
                 result["roadmap_position"] == 6, result["risk_position"] == 1,
                 result["displacement"] == 5, result["inversions"] == 9,
                 result["pairs"] == 15]),
            f"by risk the six score {result['scores']}; the safety assumption sits at "
            f"{result['roadmap_position']} on the roadmap and "
            f"{result['risk_position']} by risk, and {result['inversions']} of "
            f"{result['pairs']} pairs are inverted between the two sequences",
        ),
        practice.Check(
            "FINDING: the irreversibility term is nearly inert on this map",
            all([result["product_position"] == 1, result["product_inversions"] == 7,
                 result["moved"] == 4, result["other_safety"] == (6, 5)]),
            f"sorting by impact times uncertainty alone leaves the top item at "
            f"{result['product_position']} and still inverts "
            f"{result['product_inversions']} pairs; the irreversibility term moves "
            f"{result['moved']} assumptions by one place each, including the other safety "
            f"row from {result['other_safety'][0]} to {result['other_safety'][1]}",
        ),
        practice.Check(
            "FINDING: the roadmap order is what the build wants",
            all([result["needs_feature"] == 2,
                 result["top_test"].startswith("Publish one generated lesson")]),
            f"the top-risk test is {result['top_test']!r}, which cannot run until something "
            f"has been generated; {result['needs_feature']} of the six tests depend on the "
            "feature existing, so the answer is a bounded build rather than a reordering",
        ),
        practice.Check(
            "FINDING: nothing in the module records a roadmap to compare against",
            all([len(result["keys"]) == 8, result["records_roadmap"] is False]),
            f"prioritize returns {result['keys']} -- {len(result['keys'])} keys and no "
            "intended position -- so the mismatch has to be computed outside the artifact "
            "that is supposed to drive the decision",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
