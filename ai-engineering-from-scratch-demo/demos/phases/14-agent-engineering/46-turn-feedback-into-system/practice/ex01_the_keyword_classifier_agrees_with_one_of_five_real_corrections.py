"""Exercise 1 — the keyword classifier agrees with one of five real corrections.

    Take five corrections from a recent coding session and classify their
    real owners.

Reading of the exercise: the five have to be real or the classification is a
vocabulary test. These five are corrections from the session that produced the
practice solutions in this repository, each labelled by hand with the layer
that actually absorbed it, before running the lesson's classifier over them.

**ANSWER: `choose_target` agrees with the hand-assigned owner on 1 of 5.**
The one it gets right is the missing canonical example. It routes a crashed
setup script to `instruction` because nothing in the wording says "command";
it routes a wrong asserted number to `example` because the cause mentions an
example; and it routes a refused `gh` call to `scope` on the word
"permission", where the control that actually stuck was a written decision.

**FINDING: the classifier reads the words, and the words describe the
symptom.** `choose_target` matches **14** keywords over
`symptom + cause`, in a fixed order, and returns on the first hit. "Regression"
beats "scope" whatever the cause says, so a scope failure that mentions a
regression is filed as a test. The lesson's own warning -- "a rule that merely
repeats the symptom will fail in the next slightly different case" -- applies
to the router as much as to the rule it writes.

**FINDING: the rule text is generated from the cause, so a vague cause makes a
vague control.** `promote` writes `f"Prevent {normalize_cause(cause)}"`, and
**3** of the five rules come back as readable English only because the causes
were written in one of the **3** shapes `normalize_cause` recognises. A cause
outside those shapes is passed through lowercased, unchanged.

**FINDING: two corrections with the same real owner produce two controls.**
The fingerprint is `sha256(target|rule)`, so the hypothesised-number and
line-ceiling corrections -- both of which the repository absorbed into
`audit_practice.py` -- get **2** distinct fingerprints and **2** rules. Dedup
happens on wording, not on destination.

Structure: `CORRECTIONS` is the labelled set; `classify()` runs the lesson's
router over it and lines the answers up.
"""

from __future__ import annotations

import inspect
import re

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "46-turn-feedback-into-system"

# (symptom, cause, recurrence, consequence, the layer that actually absorbed it)
CORRECTIONS = [
    ("The scaffold crashed on a lesson with no Chinese doc",
     "doc language assumptions were implicit", 1, "lost minutes", "automation"),
    ("A solution asserted a hypothesised number and failed",
     "the expected value had no executable example", 9, "rework", "test"),
    ("A solution passed 150 lines and the audit refused it",
     "length limit was described but not checked", 4, "rework", "test"),
    ("A citation was not found because the heading wrapped",
     "the README had no canonical example of a citation line", 2, "blocked audit", "example"),
    ("gh pr edit was rejected as non-collaborator",
     "repository permission was implicit", 2, "manual step", "instruction"),
]


def corrections(ref):
    return [ref.Correction(symptom, cause, recurrence, consequence)
            for symptom, cause, recurrence, consequence, _ in CORRECTIONS]


def classify(ref):
    """Each correction with the router's answer beside the hand-assigned owner."""
    rows = []
    for correction, row in zip(corrections(ref), CORRECTIONS):
        control = ref.promote(correction)
        rows.append({"symptom": correction.symptom, "hand": row[4],
                     "routed": control.target, "rule": control.rule,
                     "fingerprint": control.fingerprint,
                     "agrees": control.target == row[4]})
    return rows


def router_shape(ref):
    """How choose_target decides: quoted strings, destinations, keywords."""
    router = inspect.getsource(ref.choose_target)
    quoted = len(re.findall(r'"[a-z ]+"', router))
    returns = router.count("return")
    return {"quoted": quoted, "returns": returns, "keywords": quoted - returns,
            "shapes": len(re.findall(r"re\.fullmatch",
                                     inspect.getsource(ref.normalize_cause)))}


def audit_owned(rows):
    """The corrections this repository absorbed into audit_practice.py."""
    owned = [row for row in rows if row["hand"] == "test"]
    return {"audit_count": len(owned),
            "audit_fingerprints": len({row["fingerprint"] for row in owned}),
            "audit_rules": len({row["rule"] for row in owned})}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = classify(ref)
    return {
        **router_shape(ref), **audit_owned(rows),
        "rows": len(rows), "agreed": sum(row["agrees"] for row in rows),
        "routed": [row["routed"] for row in rows],
        "hand": [row["hand"] for row in rows],
        "rules": [row["rule"] for row in rows],
        "generated": sum(row["rule"].startswith("Prevent ") for row in rows),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the classifier agrees with the hand-assigned owner on 1 of 5",
            all([result["rows"] == 5, result["agreed"] == 1,
                 result["routed"] == ["instruction", "example", "instruction", "example",
                                      "scope"],
                 result["hand"] == ["automation", "test", "test", "example", "instruction"]]),
            f"the router answers {result['routed']} against hand labels "
            f"{result['hand']} -- {result['agreed']} of {result['rows']} agree, and the one "
            "it gets right is the missing canonical example",
        ),
        practice.Check(
            "FINDING: the classifier reads the words, and the words describe the symptom",
            all([result["keywords"] == 14, result["returns"] == 5]),
            f"choose_target matches {result['keywords']} keywords over symptom plus cause "
            f"in a fixed order and returns on the first hit ({result['returns']} returns), "
            "so a scope failure that mentions a regression is filed as a test",
        ),
        practice.Check(
            "FINDING: the rule text is generated from the cause",
            all([result["generated"] == 5, result["shapes"] == 3,
                 "Prevent implicit doc language assumptions" in result["rules"]]),
            f"promote writes 'Prevent <normalized cause>' for all {result['generated']} "
            f"rules, and normalize_cause recognises {result['shapes']} shapes; a cause "
            "outside them is passed through lowercased and unchanged",
        ),
        practice.Check(
            "FINDING: two corrections with the same real owner produce two controls",
            all([result["audit_count"] == 2, result["audit_fingerprints"] == 2,
                 result["audit_rules"] == 2]),
            f"the {result['audit_count']} corrections this repository absorbed into "
            f"audit_practice.py get {result['audit_fingerprints']} fingerprints and "
            f"{result['audit_rules']} rules: dedup happens on wording, not on destination",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
