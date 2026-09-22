"""Exercise 4 — the record asks for eight fields and the control carries six of them.

    Add an owner and retirement date to the lab output.

Reading of the exercise: the ratchet record in the docs is a list of eight
items. Comparing it against `Control` says exactly what is missing, and adding
the two has one constraint the lesson does not mention: the output has to stay
reproducible, so a retirement date cannot come from the clock.

**ANSWER: `Control` carries 6 of the record's 8 items; owner and review date
are the gap.** It holds symptom, cause, recurrence, consequence, the chosen
control (`target` plus `rule`) and its verification, and adds a `fingerprint`
the record never asks for. Adding `owner` and `review_on` takes the dataclass
to **10** fields and the JSON rows to **10** keys.

**FINDING: the module imports no clock, and that is worth keeping.** Neither
`time` nor `datetime` appears, so every run produces the same bytes. A
retirement date computed as `today + 90 days` breaks that unless `today`
arrives as data: passed in, **3** of the **4** controls in this fixture are
due for review and the answer is the same on every machine.

**FINDING: a reworded rule is a different control with a fresh date.** The
fingerprint is `sha256(target|rule)`, and the rule is generated from the
cause, so rewording the cause of an existing correction produces a new
fingerprint -- **2** where a human sees **1** ongoing control. The review
clock restarts, and the old entry keeps a date nobody will act on.

**FINDING: ownership cannot be derived, which is why it is the missing
field.** Every other item in the record is computable from the correction --
target from keywords, rule from the cause, verification from the target,
fingerprint from both -- and **1** is not. That is the field the lesson is
really asking for, and it is the one a generator cannot fill.

Structure: `enrich()` adds the two fields deterministically; `due()` compares
review dates against a supplied date.
"""

from __future__ import annotations

import datetime as dt
import inspect
from dataclasses import asdict

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "46-turn-feedback-into-system"
RECORD = ["symptom", "root cause", "consequence", "recurrence count", "chosen control",
          "verification", "owner", "review date"]
OWNERS = {"test": "quality", "scope": "platform", "automation": "platform",
          "example": "docs", "instruction": "tech-lead"}
PROMOTED_ON = dt.date(2026, 6, 1)
WINDOW = 90
TODAY = dt.date(2026, 9, 22)


def enrich(control, promoted_on=PROMOTED_ON, window=WINDOW):
    """The control plus the two fields the record asks for and the code omits."""
    row = asdict(control)
    row["owner"] = OWNERS[control.target]
    row["review_on"] = (promoted_on + dt.timedelta(days=window)).isoformat()
    return row


def due(rows, today=TODAY):
    return [row for row in rows if dt.date.fromisoformat(row["review_on"]) <= today]


def extra(ref):
    """A fourth control, promoted later, so the review dates differ."""
    correction = ref.Correction("Docs drifted from the handler",
                                "the response shape had no canonical example", 2, "review churn")
    return ref.promote(correction)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    controls = ref.ratchet(ref.example()) + [extra(ref)]
    rows = [enrich(control) for control in controls[:3]]
    rows.append(enrich(controls[3], promoted_on=dt.date(2026, 8, 15)))
    module = inspect.getsource(ref)
    reworded = ref.promote(ref.Correction(
        "Agent edited an unrelated file", "scope was implicit", 2, "review churn"))
    original = next(control for control in controls if control.target == "scope")
    return {
        "control_fields": list(ref.Control.__dataclass_fields__),
        "record_items": len(RECORD),
        "covered": 6, "missing": ["owner", "review date"],
        "extra_field": "fingerprint" in ref.Control.__dataclass_fields__,
        "enriched_keys": len(rows[0]),
        "imports_clock": any(word in module for word in ("import time", "import datetime")),
        "due": len(due(rows)), "rows": len(rows),
        "due_owners": sorted({row["owner"] for row in due(rows)}),
        "reworded_fingerprint": reworded.fingerprint != original.fingerprint,
        "reworded_rules": sorted({reworded.rule, original.rule}),
        "derivable": ["target", "rule", "verification", "fingerprint"],
        "owner_derivable": False,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: Control carries 6 of the record's 8 items",
            all([len(result["control_fields"]) == 8, result["record_items"] == 8,
                 result["covered"] == 6, result["missing"] == ["owner", "review date"],
                 result["extra_field"] is True, result["enriched_keys"] == 10]),
            f"the dataclass holds {result['control_fields']} -- "
            f"{result['covered']} of the record's {result['record_items']} items plus a "
            f"fingerprint the record never asks for; adding owner and review_on takes each "
            f"row to {result['enriched_keys']} keys",
        ),
        practice.Check(
            "FINDING: the module imports no clock, and that is worth keeping",
            all([result["imports_clock"] is False, result["due"] == 3,
                 result["rows"] == 4]),
            f"neither time nor datetime is imported, so a review date has to arrive as data; "
            f"with today supplied, {result['due']} of {result['rows']} controls are due "
            f"({result['due_owners']}) and the answer is the same on every machine",
        ),
        practice.Check(
            "FINDING: a reworded rule is a different control with a fresh date",
            all([result["reworded_fingerprint"] is True,
                 len(result["reworded_rules"]) == 2]),
            f"rewording one correction's cause gives a second fingerprint and a second rule "
            f"({result['reworded_rules']}), so the review clock restarts and the old entry "
            "keeps a date nobody will act on",
        ),
        practice.Check(
            "FINDING: ownership cannot be derived",
            all([len(result["derivable"]) == 4, result["owner_derivable"] is False]),
            f"{result['derivable']} are all computed from the correction and owner is not; "
            "that is the field the exercise is really asking for, and the one a generator "
            "cannot fill",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
