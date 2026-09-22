"""Exercise 5 — three outputs satisfy the outcome, and one constraint picks between them.

    Write three different outputs that could satisfy the same outcome.

Reading of the exercise: writing three is easy; the point is what happens
next. If the outcome really is output-independent, all three should pass the
frame unchanged, and something other than taste has to choose. That something
is the constraint list.

**ANSWER: all 3 outputs pass the frame and the constraints leave 1.**
Imported answers, a fixture-diff harness and a worked-answer quiz each satisfy
"a reader can run an answer and see it check its own claims against the
lesson's code" -- **3** of **3** validate with **0** issues. Scored against
the frame's **3** constraints the three read **3**, **2** and **2**: the
clause about importing the lesson's code is what decides, and it decides for a
reason that is written down.

**FINDING: the leak check fires on the output that is named, not the one that
is chosen.** Putting each candidate's own name into the desired outcome
produces **3** of **3** flagged frames, and leaving the outcome alone produces
**0**, whichever output is proposed. The check protects the sentence, not the
decision.

**FINDING: two of the three fail a constraint that sounds like a detail.**
"Answers import the lesson's code" reads like an implementation note, and it
is the clause that eliminates the fixture harness -- whose fixtures drift the
moment the lesson's code changes -- and the quiz, which never executes
anything. A constraint that only removes one candidate would not have been
worth writing.

**FINDING: the frame cannot record why the survivor won.** `OutcomeFrame` has
**7** fields and `decision` returns **4** keys, none of which holds the
alternatives considered or the constraint that eliminated them. The reasoning
that makes the choice reviewable lives outside the artifact the lesson keeps.

Structure: `CANDIDATES` is the three outputs; `score()` applies the frame's
constraints; `leak()` runs the lesson's check over each one.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "47-outcomes-before-output"
OUTCOME = ("a reader can run an answer and see it check its own claims against the "
           "lesson's code")
CONSTRAINTS = ["answers import the lesson's code",
               "the same command produces the same verdict on any machine",
               "no dependency outside the standard library"]

# (output, which constraints it satisfies)
CANDIDATES = [
    ("imported answers that assert on the lesson's own functions", (True, True, True)),
    ("a fixture-diff harness comparing saved output", (False, True, True)),
    ("a quiz with worked answers in prose", (False, True, True)),
]


def frame(ref, desired=OUTCOME, proposed=""):
    return ref.OutcomeFrame(
        user="a reader working through a phase 14 lesson",
        situation="facing the exercises after reading the lesson",
        current_behavior="guesses at an answer with no way to check it",
        desired_outcome=desired, constraints=list(CONSTRAINTS),
        non_goals=["rewriting the lesson"], proposed_output=proposed)


def score(candidate):
    """How many of the frame's constraints an output satisfies."""
    name, satisfied = candidate
    return {"output": name, "satisfied": sum(satisfied),
            "fails": [CONSTRAINTS[index] for index, ok in enumerate(satisfied) if not ok]}


def leak(ref, candidate):
    """The lesson's own check, with the candidate named in the outcome and not."""
    name = candidate[0]
    clean = ref.validate(frame(ref, proposed=name))
    named = ref.validate(frame(ref, desired=f"readers use {name}", proposed=name))
    return {"clean": clean, "named": named}


def executes(candidate):
    """Whether the output runs anything at all when a reader uses it."""
    return "quiz" not in candidate[0]


def ranking(scored):
    """Which outputs survive the constraint list, and which clause removed the rest."""
    return {"scores": [row["satisfied"] for row in scored],
            "winner": max(scored, key=lambda row: row["satisfied"])["output"],
            "survivors": sum(row["satisfied"] == len(CONSTRAINTS) for row in scored),
            "eliminated": [row["output"] for row in scored if CONSTRAINTS[0] in row["fails"]]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    scored = [score(candidate) for candidate in CANDIDATES]
    leaks = [leak(ref, candidate) for candidate in CANDIDATES]
    document = ref.decision(frame(ref))
    return {
        **ranking(scored),
        "candidates": len(CANDIDATES),
        "valid": sum(not row["clean"] for row in leaks),
        "constraints": len(CONSTRAINTS),
        "flagged": sum(bool(row["named"]) for row in leaks),
        "unflagged": sum(not row["clean"] for row in leaks),
        "deciding": CONSTRAINTS[0],
        "executing": sum(executes(candidate) for candidate in CANDIDATES),
        "fields": len(ref.OutcomeFrame.__dataclass_fields__),
        "keys": sorted(document),
        "records_alternatives": any(word in document for word in
                                    ("alternatives", "considered", "rejected")),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: all 3 outputs pass the frame and the constraints leave 1",
            all([result["candidates"] == 3, result["valid"] == 3,
                 result["scores"] == [3, 2, 2], result["survivors"] == 1,
                 result["winner"].startswith("imported answers")]),
            f"{result['valid']} of {result['candidates']} candidates validate with no "
            f"issues, and scored against {result['constraints']} constraints they read "
            f"{result['scores']} -- {result['survivors']} survives, "
            f"{result['winner']!r}",
        ),
        practice.Check(
            "FINDING: the leak check fires on the output that is named",
            all([result["flagged"] == 3, result["unflagged"] == 3]),
            f"naming each candidate in the desired outcome flags {result['flagged']} of "
            f"{result['candidates']} frames and leaving the outcome alone flags none of "
            f"{result['unflagged']}: the check protects the sentence, not the decision",
        ),
        practice.Check(
            "FINDING: two of the three fail a constraint that sounds like a detail",
            all([len(result["eliminated"]) == 2,
                 result["deciding"] == "answers import the lesson's code",
                 result["executing"] == 2]),
            f"{result['deciding']!r} eliminates {len(result['eliminated'])} candidates -- "
            f"the fixture harness, whose fixtures drift when the lesson changes, and the "
            f"quiz, which is one of {result['candidates'] - result['executing']} outputs "
            "that never execute anything",
        ),
        practice.Check(
            "FINDING: the frame cannot record why the survivor won",
            all([result["fields"] == 7, len(result["keys"]) == 4,
                 result["records_alternatives"] is False]),
            f"OutcomeFrame has {result['fields']} fields and decision returns "
            f"{result['keys']}, none of which holds the alternatives considered or the "
            "constraint that eliminated them; the reasoning lives outside the artifact",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
