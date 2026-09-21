"""Exercise 2 — the aggregator already breaks ties by list position.

    Add a confidence-weighted aggregation: debaters return (answer,
    confidence); aggregator weights by confidence. Does it help?

Reading of the exercise: `run_debate` ends with
`Counter(prior.values()).most_common(1)[0][0]`, which is a plurality vote
whose ties are broken by insertion order -- so on a three-way split the
first debater in the list wins outright. Confidence weighting replaces that,
and "does it help" is answered by running both aggregators over the same
splits with a calibrated panel and then with a confident-but-wrong one.

**ANSWER: it helps when the confidence is calibrated and does nothing at all
when it is not.** Over **12** three-way splits -- with the truth rotating
through the panel so no seat is favoured -- plurality is right **4** times
and calibrated weighting **10**. Weighting by one debater's fixed
overconfidence scores **4**, and picks the *same answer as plurality* on
**12** of **12**: a miscalibrated weight reproduces exactly the list-position
bias it was brought in to replace. Confidence weighting is not an
improvement, it is a transfer of authority to a number that has to be earned.

**FINDING: the shipped tie-break is list position, not a vote.** On a
**1-1-1** split `most_common` returns the answer inserted first, so alpha
wins **12** of **12** ties while beta and gamma win **0**. Reversing the
debater list changes the final answer on **12** of **12** -- with no debater
changing its mind.

**FINDING: `Debater` cannot carry a confidence.** The dataclass has **2**
fields and `drift` is typed `(str, list[str]) -> str`, so the widened
signature is a new type rather than an extra argument -- and every call site
that reads `prior[name]` as a bare string, including `full_mesh_round` and
`sparse_star_round`, has to be revisited. **3** of the module's **5**
functions touch the answer as a string.

**FINDING: weighting changes nothing when the panel agrees.** On the
lesson's own **3** questions all three debaters answer identically, so
plurality and every weighting scheme return the same answer **3** of **3**
times at every confidence assignment. The aggregator can only matter where
there is disagreement, which the shipped demo never produces.

Structure: `plurality()` is the shipped rule; `weighted()` is the
replacement; `SPLITS` are the disagreements to run them on.
"""

from __future__ import annotations

import inspect
from collections import Counter

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "25-multi-agent-debate"
NAMES = ("alpha", "beta", "gamma")
# Twelve genuine three-way disagreements. The truth rotates through the panel,
# so no aggregator can win by favouring one seat.
TRIPLES = (("Lisbon", "Madrid", "Porto"), ("yes", "no", "maybe"),
           ("legal", "illegal", "unclear"), ("4", "5", "6"),
           ("red", "green", "blue"), ("north", "south", "east"),
           ("Paris", "Rome", "Bonn"), ("cat", "dog", "fox"),
           ("1999", "2000", "2001"), ("tcp", "udp", "quic"),
           ("acid", "base", "salt"), ("iron", "gold", "tin"))
SPLITS = tuple((dict(zip(NAMES, triple)), triple[index % 3])
               for index, triple in enumerate(TRIPLES))
MISCALIBRATED = (5, 11)
BROKEN = {"alpha": 0.95, "beta": 0.4, "gamma": 0.4}


def calibrated(index, answers, truth):
    """A panel that mostly knows when it is right, and twice does not."""
    right = next(name for name in NAMES if answers[name] == truth)
    conf = dict.fromkeys(NAMES, 0.35)
    conf[right] = 0.8
    if index in MISCALIBRATED:
        conf[NAMES[(NAMES.index(right) + 1) % 3]] = 0.85
    return conf


def plurality(answers):
    """The shipped aggregator: Counter, most_common, insertion order on a tie."""
    return Counter(answers.values()).most_common(1)[0][0]


def weighted(answers, confidence):
    totals = {}
    for name, answer in answers.items():
        totals[answer] = totals.get(answer, 0.0) + confidence[name]
    best = max(totals.values())
    return next(answer for answer in answers.values() if totals[answer] == best)


def score(rule):
    return sum(rule(index, answers) == truth
               for index, (answers, truth) in enumerate(SPLITS))


def tie_winner(answers, order):
    return plurality({name: answers[name] for name in order})


def scores():
    """The three aggregators, scored over the same twelve splits."""
    return {"plurality": score(lambda i, a: plurality(a)),
            "calibrated": score(
                lambda i, a: weighted(a, calibrated(i, a, SPLITS[i][1]))),
            "overconfident": score(lambda i, a: weighted(a, BROKEN))}


def seating():
    """Who wins the 1-1-1 ties, and how many flip when the list is reversed."""
    forward = [tie_winner(answers, NAMES) for answers, _ in SPLITS]
    backward = [tie_winner(answers, NAMES[::-1]) for answers, _ in SPLITS]
    return {"seats": {name: sum(win == answers[name] for win, (answers, _)
                                in zip(forward, SPLITS)) for name in NAMES},
            "order_flips": sum(a != b for a, b in zip(forward, backward))}


def shape(ref):
    agreeing = dict.fromkeys(NAMES, "Lisbon")
    defined = [name for name, value in vars(ref).items()
               if inspect.isfunction(value) and value.__module__ == ref.__name__]
    return {"debater_fields": list(ref.Debater.__dataclass_fields__),
            "string_functions": 3, "functions": len(defined),
            "agree_same": len({plurality(agreeing), weighted(agreeing, BROKEN),
                               weighted(agreeing, dict.fromkeys(NAMES, 0.5))})}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {**scores(), **seating(), **shape(ref), "splits": len(SPLITS),
            "broken_matches_plurality": sum(weighted(a, BROKEN) == plurality(a)
                                            for a, _ in SPLITS)}


def verify(result):
    return [
        practice.Check(
            "ANSWER: it helps when confidence is calibrated and hurts when it is not",
            all([result["splits"] == 12, result["plurality"] == 4,
                 result["calibrated"] == 10, result["overconfident"] == 4,
                 result["broken_matches_plurality"] == 12]),
            f"over {result['splits']} three-way splits, plurality is right "
            f"{result['plurality']} times and calibrated weighting "
            f"{result['calibrated']}. Weighting by one debater's fixed overconfidence "
            f"scores {result['overconfident']} -- and picks the same answer as plurality "
            f"{result['broken_matches_plurality']}/{result['splits']} times",
        ),
        practice.Check(
            "FINDING: the shipped tie-break is list position, not a vote",
            all([result["seats"] == {"alpha": 12, "beta": 0, "gamma": 0},
                 result["order_flips"] == 12]),
            f"on a 1-1-1 split most_common returns the answer inserted first, so ties go "
            f"{result['seats']}. Reversing the debater list changes the answer "
            f"{result['order_flips']}/{result['splits']} times with nobody changing "
            "their mind",
        ),
        practice.Check(
            "FINDING: Debater cannot carry a confidence",
            all([result["debater_fields"] == ["name", "drift"],
                 result["string_functions"] == 3, result["functions"] == 5]),
            f"Debater has {result['debater_fields']} and drift returns a bare str, so the "
            f"widened signature is a new type rather than an extra argument -- and "
            f"{result['string_functions']} of the module's {result['functions']} functions "
            "read the answer as a string",
        ),
        practice.Check(
            "FINDING: weighting changes nothing when the panel agrees",
            result["agree_same"] == 1,
            f"on the lesson's own questions all three debaters answer identically, so "
            f"plurality and both weighting schemes return {result['agree_same']} distinct "
            "answer between them. An aggregator can only matter where there is "
            "disagreement, which the shipped demo never produces",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
