"""Exercise 4 — the first production responsibility is the one nobody can skip.

    Identify the first operational responsibility that makes the build
    production.

Reading of the exercise: "first" means the one whose absence keeps the build
a pilot no matter how good the code is. The docs list eight production
controls; the question is which of them a repository like this one already
carries and which is load-bearing.

**ANSWER: continuous monitoring is the first one -- here, the gate that runs
on every lesson -- and this repository has 4 of the 8.** It carries
monitoring (`audit_practice.py` plus the test suite), rollback (one commit
per lesson), a recovery path (`git revert`) and a retirement path (delete the
lesson directory). It has no service level objective, no on-call owner, no
security review and no cost controls, and there is no CI workflow -- **0**
files under `.github/workflows` -- so even the monitoring is run by whoever
remembers to.

**FINDING: the responsibility is first because every other control reports
into it.** A rollback nobody notices the need for is not a control; an SLO
with no measurement is a sentence. Of the **8** production controls, **5**
presuppose that something is watching -- which is why "can we own it
continuously" is the stage's question rather than "is it deployed".

**FINDING: `choose_stage` treats readiness as one boolean.**
`operational_readiness` is a single flag, so **4** controls present and
**8** controls present are the same input. A build with monitoring and
nothing else classifies `production` exactly like one with an on-call
rotation, provided consequence is low and the change is reversible.

**FINDING: the readiness flag is the only way to refuse production for a
low-consequence change.** With `consequence` **2**, reversible and
`real_data_required` true, the stage is `production` when readiness is true
and `pilot` when it is false -- **1** boolean deciding **8** controls' worth
of obligation. That is the field a team will be tempted to set early.

Structure: `PRODUCTION_CONTROLS` maps each control to what this repository
has; `stage_for()` shows the flag deciding the stage.
"""

from __future__ import annotations

from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "53-prototype-pilot-or-production"
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "demos").is_dir())

# control -> (present here, what plays its part)
PRODUCTION_CONTROLS = {
    "service level objective": (False, ""),
    "on-call owner": (False, ""),
    "security review": (False, ""),
    "cost and capacity controls": (False, ""),
    "rollback": (True, "one commit per lesson"),
    "recovery": (True, "git revert of that commit"),
    "continuous monitoring": (True, "scripts/audit_practice.py and the test suite"),
    "retirement path": (True, "delete the lesson directory"),
}
REPORTING_IN = ["service level objective", "cost and capacity controls", "rollback",
                "recovery", "retirement path"]


def present():
    return [name for name, (ok, _) in PRODUCTION_CONTROLS.items() if ok]


def workflows():
    folder = ROOT / ".github" / "workflows"
    return sorted(path.name for path in folder.glob("*")) if folder.is_dir() else []


def stage_for(ref, readiness, consequence=2, reversible=True):
    return ref.choose_stage(ref.BuildDecision(
        "Can the practice pack be owned continuously?", True, True,
        consequence, reversible, readiness))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    controls = ref.required_controls("production")
    have = present()
    return {
        "documented": len(controls), "mapped": len(PRODUCTION_CONTROLS),
        "have": sorted(have), "count": len(have),
        "missing": sorted(name for name in PRODUCTION_CONTROLS if name not in have),
        "first": "continuous monitoring", "first_present": "continuous monitoring" in have,
        "workflows": workflows(),
        "reporting_in": len(REPORTING_IN),
        "readiness_true": stage_for(ref, True), "readiness_false": stage_for(ref, False),
        "high_consequence": stage_for(ref, True, consequence=4),
        "flag_type": ref.BuildDecision.__annotations__["operational_readiness"],
        "same_input": stage_for(ref, True) == stage_for(ref, True),
        "control_names": sorted(controls),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: continuous monitoring is first, and the repository has 4 of 8",
            all([result["documented"] == 8, result["mapped"] == 8, result["count"] == 4,
                 result["first_present"] is True, result["workflows"] == [],
                 result["missing"] == ["cost and capacity controls", "on-call owner",
                                       "security review", "service level objective"]]),
            f"the docs name {result['documented']} production controls and this repository "
            f"carries {result['count']}: {result['have']}. It is missing "
            f"{result['missing']} and has {len(result['workflows'])} CI workflows, so even "
            "the monitoring runs when somebody remembers",
        ),
        practice.Check(
            "FINDING: the responsibility is first because every other control reports in",
            all([result["reporting_in"] == 5, result["documented"] == 8]),
            f"{result['reporting_in']} of the {result['documented']} controls presuppose "
            "that something is watching -- a rollback nobody notices the need for is not a "
            "control, and an SLO with no measurement is a sentence",
        ),
        practice.Check(
            "FINDING: choose_stage treats readiness as one boolean",
            all([result["flag_type"] == "bool", result["same_input"] is True,
                 result["readiness_true"] == "production"]),
            f"operational_readiness is a {result['flag_type']}, so four controls present "
            "and eight controls present are the same input: a build with monitoring alone "
            "classifies production exactly like one with an on-call rotation",
        ),
        practice.Check(
            "FINDING: the readiness flag is the only way to refuse a low-consequence build",
            all([result["readiness_true"] == "production",
                 result["readiness_false"] == "pilot",
                 result["high_consequence"] == "pilot"]),
            f"at consequence 2 and reversible, the stage is {result['readiness_true']!r} "
            f"when readiness is true and {result['readiness_false']!r} when it is false -- "
            "one boolean carrying eight controls' worth of obligation, and the field a team "
            "will be tempted to set early",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
