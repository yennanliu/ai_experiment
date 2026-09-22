"""Exercise 1 — the stage ignores consequence whenever the data is synthetic.

    Classify three current projects by learning stage, not deployment status.

Reading of the exercise: "not deployment status" is the instruction that
matters. Three real pieces of work in this repository get classified by the
lesson's own decision function, and one of them is deployed and still a
prototype.

**ANSWER: the three classify as production, pilot and prototype, and the
prototype is the one that ships in `scripts/`.** The practice solutions --
real readers, real repository, low consequence, reversible, and the audit
already running -- come back `production`. The answers generator sketched in
Lessons 49 to 52 needs real readers before anyone trusts it and has no
operational owner, so it is a `pilot`. The scaffolder fix runs on synthetic
inputs in the sense the classifier cares about -- no real users, no real
data -- so it is a `prototype`, and it is nonetheless imported by every
lesson since.

**FINDING: `choose_stage` returns `prototype` before it ever looks at
consequence.** The first branch is `not real_users_required and not
real_data_required`, so a decision with consequence **5**, irreversible and
with no operational readiness still classifies `prototype` and draws **3**
controls. Stage drift is not a failure of discipline here; it is the
function's first line.

**FINDING: the `unknown` field is the only thing that describes the learning
question, and nothing reads it.** `BuildDecision` has **6** fields and
`choose_stage` reads **5**; the prototype control list names "learning
question" as a control while the field carrying it is never consulted.

**FINDING: the controls are strings and the stage is not observable from the
system.** `required_controls` returns **3**, **5** and **8** strings for the
three stages, and the docs say a warning banner is not enough -- the stage
should be visible in configuration, access control and telemetry. Nothing in
the returned plan is machine-checkable, so the drift the lesson warns about
cannot be detected by the artifact that names it.

Structure: `PROJECTS` is the classification; `drift()` is the
consequence-5 prototype.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "53-prototype-pilot-or-production"

# (project, unknown, real_users, real_data, consequence, reversible, readiness)
PROJECTS = [
    ("the practice solutions", "Does a reader get a runnable answer per exercise?",
     True, True, 2, True, True),
    ("the answers generator", "Would a reader trust a generated answer?",
     True, True, 3, True, False),
    ("the scaffolder fallback", "Can a manifest be built from one document?",
     False, False, 2, True, True),
]
DRIFT = ("a prototype with production consequence", "Can the pack install cleanly?",
         False, False, 5, False, False)


def build(ref, row):
    return ref.BuildDecision(row[1], row[2], row[3], row[4], row[5], row[6])


def classify(ref):
    return [(row[0], ref.choose_stage(build(ref, row))) for row in PROJECTS]


def drift(ref):
    decision = build(ref, DRIFT)
    return {"stage": ref.choose_stage(decision),
            "controls": len(ref.required_controls(ref.choose_stage(decision))),
            "consequence": decision.consequence, "reversible": decision.reversible,
            "readiness": decision.operational_readiness}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    stages = classify(ref)
    source = inspect.getsource(ref.choose_stage)
    fields = list(ref.BuildDecision.__dataclass_fields__)
    return {
        "projects": len(stages), "stages": [stage for _, stage in stages],
        "deployed_prototype": stages[2],
        "drift": drift(ref),
        "first_branch": source.index("real_users_required") < source.index("consequence >= 4"),
        "fields": fields, "read": sum(name in source for name in fields),
        "unknown_read": "decision.unknown" in source,
        "controls": {stage: len(ref.required_controls(stage))
                     for stage in ("prototype", "pilot", "production")},
        "learning_question": "learning question" in ref.required_controls("prototype"),
        "checkable": sum(any(word in control for word in ("config", "access", "telemetry"))
                         for stage in ("prototype", "pilot", "production")
                         for control in ref.required_controls(stage)),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: production, pilot and prototype -- and the prototype ships",
            all([result["projects"] == 3,
                 result["stages"] == ["production", "pilot", "prototype"],
                 result["deployed_prototype"] == ("the scaffolder fallback", "prototype")]),
            f"the three classify as {result['stages']}; the one the classifier calls "
            f"{result['deployed_prototype'][1]!r} is {result['deployed_prototype'][0]}, "
            "which ships in scripts/ and is imported by every lesson since",
        ),
        practice.Check(
            "FINDING: choose_stage returns prototype before it looks at consequence",
            all([result["drift"]["stage"] == "prototype", result["drift"]["controls"] == 3,
                 result["drift"]["consequence"] == 5,
                 result["drift"]["reversible"] is False,
                 result["first_branch"] is True]),
            f"a decision at consequence {result['drift']['consequence']}, irreversible and "
            f"with no readiness, still classifies {result['drift']['stage']!r} and draws "
            f"{result['drift']['controls']} controls, because the synthetic-inputs test is "
            "the function's first line",
        ),
        practice.Check(
            "FINDING: the unknown field is never read",
            all([len(result["fields"]) == 6, result["read"] == 5,
                 result["unknown_read"] is False, result["learning_question"] is True]),
            f"BuildDecision has {len(result['fields'])} fields and choose_stage reads "
            f"{result['read']}; the prototype control list names 'learning question' as a "
            "control while the field carrying it is never consulted",
        ),
        practice.Check(
            "FINDING: the controls are strings and the stage is not observable",
            all([result["controls"] == {"prototype": 3, "pilot": 5, "production": 8},
                 result["checkable"] == 0]),
            f"the stages draw {result['controls']} controls and "
            f"{result['checkable']} of them mention configuration, access control or "
            "telemetry, so the drift the lesson warns about cannot be detected by the "
            "artifact that names it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
