"""Exercise 3 — consequence is carried into the output and never read.

    Add consequence weighting so a severe first occurrence can be promoted
    immediately.

Reading of the exercise: before adding a weight, find the gate it is supposed
to join. `ratchet` skips a correction only when `recurrence < 1`, which no
recorded correction ever is, so today nothing is held back and "promoted
immediately" describes the existing behaviour for everything.

**ANSWER: weighting only matters once something can be refused, and then it
promotes 3 of 4 instead of 4.** Adding a one-off preference -- recurrence 1,
consequence "minor annoyance" -- to the lesson's three corrections leaves the
shipped ratchet promoting **4** of **4**. A gate of "recurrence at least 2, or
severity at least 3" promotes **3**: the scope failure on recurrence, the
setup failure on recurrence, and the first-occurrence regression on
consequence alone. The preference stays out.

**FINDING: `consequence` appears 3 times in the module and 0 of them are in a
condition.** It is a field on `Correction`, an argument to `Control`, and a
key in the JSON. The lesson's own record says to promote "when recurrence or
consequence justifies permanent complexity", and only one of those two words
reaches an `if`.

**FINDING: the recurrence gate excludes nothing.** Over the example, **0** of
**3** corrections are filtered, because a correction that has been observed
has recurrence at least 1 by construction. A gate whose threshold sits below
the minimum possible value is a comment.

**FINDING: weighting free text is the classifier problem again.** The example's
three consequences are **3** distinct strings sharing **0** words, so a weight
map is a keyword table with the same failure mode Exercise 1 measured: an
unseen phrase scores whatever the default is. The field wants a small enum --
**3** levels are enough -- and the enum belongs to the correction, not to the
scorer.

Structure: `weigh()` is the gate the exercise asks for; `cases()` adds the
one-off preference that separates the two policies.
"""

from __future__ import annotations

import inspect
import re

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "46-turn-feedback-into-system"
SEVERITY = {"user-visible failure": 3, "lost session": 2, "review churn": 1,
            "minor annoyance": 0}
ONE_OFF = ("Agent used tabs in one file", "formatting preference was implicit", 1,
           "minor annoyance")


def cases(ref):
    return list(ref.example()) + [ref.Correction(*ONE_OFF)]


def weigh(correction, floor=2, severe=3):
    """Promote on recurrence, or on a first occurrence severe enough to justify it."""
    severity = SEVERITY.get(correction.consequence, 0)
    return correction.recurrence >= floor or severity >= severe


def reads_consequence(ref):
    """How the module treats the consequence field."""
    module = inspect.getsource(ref)
    conditions = [line for line in module.splitlines()
                  if "consequence" in line and re.search(r"\b(if|while)\b", line)]
    return {"mentions": module.count("consequence"), "conditions": len(conditions),
            "gate": "recurrence < 1" in inspect.getsource(ref.ratchet)}


def vocabulary(ref):
    """Whether the consequence strings share any vocabulary to weigh."""
    consequences = [correction.consequence for correction in ref.example()]
    words = [set(text.split()) for text in consequences]
    return {"consequences": len(set(consequences)),
            "shared_words": len(set.intersection(*words)),
            "levels": len({value for value in SEVERITY.values() if value})}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    corrections = cases(ref)
    weighted = [correction for correction in corrections if weigh(correction)]
    return {
        **reads_consequence(ref), **vocabulary(ref),
        "corrections": len(corrections), "shipped": len(ref.ratchet(corrections)),
        "weighted": len(weighted),
        "dropped": sorted(correction.symptom for correction in corrections
                          if correction not in weighted),
        "severe_first": [correction.symptom for correction in weighted
                         if correction.recurrence == 1],
        "filtered": len(ref.example()) - len(ref.ratchet(ref.example())),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: weighting promotes 3 of 4 where the shipped gate promotes 4",
            all([result["corrections"] == 4, result["shipped"] == 4,
                 result["weighted"] == 3,
                 result["dropped"] == ["Agent used tabs in one file"],
                 result["severe_first"] == ["A regression escaped"]]),
            f"the shipped ratchet promotes {result['shipped']} of "
            f"{result['corrections']}; a gate of recurrence>=2 or severity>=3 promotes "
            f"{result['weighted']}, keeping {result['severe_first']} on its first "
            f"occurrence and dropping {result['dropped']}",
        ),
        practice.Check(
            "FINDING: consequence appears 3 times and never in a condition",
            all([result["mentions"] == 3, result["conditions"] == 0]),
            f"the word appears {result['mentions']} times -- a field, an argument, a JSON "
            f"key -- and {result['conditions']} of them are in an if, although the lesson "
            "says to promote when recurrence *or* consequence justifies it",
        ),
        practice.Check(
            "FINDING: the recurrence gate excludes nothing",
            all([result["gate"] is True, result["filtered"] == 0]),
            f"ratchet skips only recurrence < 1 and filters {result['filtered']} of the "
            "example's corrections, because an observed correction has recurrence at least "
            "1 by construction. A threshold below the minimum possible value is a comment",
        ),
        practice.Check(
            "FINDING: weighting free text is the classifier problem again",
            all([result["consequences"] == 3, result["shared_words"] == 0,
                 result["levels"] == 3]),
            f"the {result['consequences']} consequences share {result['shared_words']} "
            f"words, so a weight map is a keyword table with Exercise 1's failure mode; "
            f"{result['levels']} enum levels on the correction would remove the guessing",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
