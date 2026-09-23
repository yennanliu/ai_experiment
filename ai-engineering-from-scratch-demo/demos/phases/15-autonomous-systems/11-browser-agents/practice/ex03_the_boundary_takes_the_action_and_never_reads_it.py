"""Exercise 3 — the boundary takes the action and never reads it.

    Pick one real browser-agent workflow you know (e.g., "book a flight").
    List every read and every write. Mark which writes need HITL and why.

Reading of the exercise: "mark which writes need HITL and why" needs a rule,
not a list of opinions, so each write is scored on two properties a browser
agent can actually determine -- whether it is reversible without a third
party, and whether it moves money or identity. The workflow is "book a
flight", enumerated step by step.

**ANSWER: 9 reads, 5 writes, and 3 of the writes need a human.** The reads
are the search form, results, fare rules, seat map, baggage rules, the
traveller profile, saved cards, the itinerary preview and the confirmation
page. The writes are: set search parameters, select a fare, enter passenger
details, **pay**, and **email the itinerary**. Holding a seat is reversible
by waiting; paying, writing passenger identity to a third party, and sending
mail are not -- **3** of **5**.

**FINDING: the shipped boundary cannot express any of this.**
`rw_boundary_allows(content_origin, action)` takes the action and its body
never mentions it -- the function is `content_origin == "user"`. So a $2 seat
selection and a $2000 charge are the same decision, and the **5** writes above
collapse to **1** bit that is about *provenance*, not consequence.

**FINDING: provenance and consequence are independent, and only one is
checked.** Of the 5 writes, **5** have `content_origin == "user"` in the
happy path -- the user asked for a flight -- so the boundary approves every
one of them, including the payment. It is a defense against injected writes
and not a defense against expensive ones, and the lesson's grid never puts
those two cases in the same row.

**FINDING: the reads are where the injection enters and nothing reads
them.** **9** reads, of which **4** render third-party content -- results,
fare rules, seat map, baggage rules -- and the sanitizer sees HTML from
**1** page at a time with no notion of which read it came from. The
read/write boundary's one bit is set by `run_agent` itself, from a substring
test, which is the attribution the lesson's own headline calls "itself
attackable".

Structure: `WORKFLOW` is the enumerated task; `needs_hitl()` is the
two-property rule; the boundary is asked about each write.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "11-browser-agents"

# "Book a flight", step by step. (kind, step, reversible-without-a-third-party,
# moves money or identity, renders third-party content)
WORKFLOW = (
    ("read", "search form", True, False, False),
    ("write", "set origin, destination, dates", True, False, False),
    ("read", "results list", True, False, True),
    ("read", "fare rules", True, False, True),
    ("write", "select fare, hold seat", True, False, False),
    ("read", "seat map", True, False, True),
    ("read", "baggage rules", True, False, True),
    ("read", "traveller profile", True, False, False),
    ("write", "enter passenger name, DOB, passport", False, True, False),
    ("read", "saved cards", True, False, False),
    ("write", "pay", False, True, False),
    ("read", "itinerary preview", True, False, False),
    ("read", "confirmation page", True, False, False),
    ("write", "email itinerary to the traveller", False, False, False),
)


def steps(kind):
    return [row for row in WORKFLOW if row[0] == kind]


def needs_hitl(row):
    """A write needs a human when it cannot be undone without a third party."""
    _kind, _step, reversible, _money, _third_party = row
    return not reversible


def boundary_verdicts(ref, origin="user"):
    """What the shipped boundary says about each write in the happy path."""
    return [ref.rw_boundary_allows(origin, {"endpoint": step, "body": ""})
            for _k, step, *_rest in steps("write")]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    writes, reads = steps("write"), steps("read")
    body = inspect.getsource(ref.rw_boundary_allows).split("return")[-1]
    return {
        "reads": len(reads), "writes": len(writes),
        "hitl": [row[1] for row in writes if needs_hitl(row)],
        "no_hitl": [row[1] for row in writes if not needs_hitl(row)],
        "money_or_identity": sum(row[3] for row in writes),
        "boundary_params": list(inspect.signature(ref.rw_boundary_allows).parameters),
        "boundary_reads_action": "action" in body,
        "boundary_approves": sum(boundary_verdicts(ref)),
        "boundary_blocks": sum(1 for ok in boundary_verdicts(ref) if not ok),
        "third_party_reads": sum(row[4] for row in reads),
        "sanitizer_params": list(inspect.signature(ref.sanitizer).parameters),
        "origin_set_by_agent": "content_origin = \"page\"" in inspect.getsource(ref.run_agent),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 9 reads, 5 writes, 3 of them needing a human",
            all([result["reads"] == 9, result["writes"] == 5,
                 len(result["hitl"]) == 3,
                 result["hitl"] == ["enter passenger name, DOB, passport", "pay",
                                    "email itinerary to the traveller"]]),
            f"{result['reads']} reads and {result['writes']} writes, of which "
            f"{len(result['hitl'])} cannot be undone without a third party -- "
            f"{result['hitl']} -- while {result['no_hitl']} are reversible by waiting",
        ),
        practice.Check(
            "FINDING: the shipped boundary cannot express any of this",
            all([result["boundary_params"] == ["content_origin", "action"],
                 not result["boundary_reads_action"]]),
            f"rw_boundary_allows takes {result['boundary_params']} and its body never "
            "mentions the action, so a seat hold and a payment are the same decision",
        ),
        practice.Check(
            "FINDING: provenance and consequence are independent, and one is checked",
            all([result["boundary_approves"] == 5, result["boundary_blocks"] == 0,
                 result["money_or_identity"] == 2]),
            f"in the happy path the boundary approves {result['boundary_approves']} of "
            f"{result['writes']} writes and blocks {result['boundary_blocks']}, "
            f"including the {result['money_or_identity']} that move money or identity",
        ),
        practice.Check(
            "FINDING: the reads are where the injection enters",
            all([result["third_party_reads"] == 4,
                 result["sanitizer_params"] == ["html"],
                 result["origin_set_by_agent"]]),
            f"{result['third_party_reads']} of the {result['reads']} reads render "
            "third-party content, the sanitizer sees one page of HTML at a time, and "
            "the origin bit the boundary trusts is set by the agent itself",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
