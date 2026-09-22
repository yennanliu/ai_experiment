"""Exercise 1 — the irreversible item schedules into wave two, beside the docs.

    Add a migration item that requires explicit human approval.

Reading of the exercise: the interesting part is not writing the item, it is
watching where the scheduler puts it. A migration is the one step in this plan
that cannot be undone by editing a file, and the lesson's own validation list
ends with "the first irreversible action occurs before the relevant
uncertainty is resolved".

**ANSWER: the migration lands in wave 2 next to the documentation, and
nothing objects.** Adding `migration` with `depends_on=("contract",)` gives
waves `[['contract'], ['docs', 'implementation', 'migration'], ['integration']]`
-- validate returns **0** issues, so the plan is `ready` with an irreversible
step running concurrently with the item that decides the public contract's
wording.

**FINDING: `WorkItem` has 5 fields and none of them says "ask first".**
Approval, reversibility and blast radius have nowhere to live, so the only way
to express "a human signs this off" is to invent an item -- and the graph is
the only enforcement the plan has. The lesson's own table of five commitments
does not include the one this exercise asks for.

**FINDING: 5 of the lesson's 6 rejection rules are implemented; the missing
one is this exercise.** `validate` covers duplicate ids, missing evidence,
missing proof, unknown dependencies and cycles. The sixth -- irreversible work
before the uncertainty it depends on -- has **0** lines, which the docs admit
("requires judgment") and which the modelling makes mechanical anyway.

**FINDING: modelling approval as a dependency moves the migration, and
exposes a second ordering bug.** An `approval` node the migration waits on
gives **4** waves --
`[['contract'], ['docs', 'implementation'], ['approval', 'integration'],
['migration']]` -- so the migration now runs last, after the approval, and the
integration gate runs in wave 3, *before* the schema change it is supposed to
verify. Nothing in the plan says the gate depends on the migration, so the
scheduler is right and the plan is wrong.

Structure: `plan()` builds the lesson's example plus the migration;
`gated()` adds the approval node and re-schedules.
"""

from __future__ import annotations

import inspect
import re

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "44-plan-from-evidence"
MIGRATION = ("migration", "Add the unique index on normalized email",
             ("migrations/0007_accounts.py:1",), ("contract",),
             "python3 manage.py migrate --check")


def plan(ref, extra=()):
    return list(ref.example()) + [ref.WorkItem(*row) for row in extra]


def gated(ref):
    """The same plan with a human approval node the migration must wait on."""
    approval = ("approval", "Human signs off on the index and its lock window",
                ("docs/runbook.md:12",), ("docs",), "recorded approval in the task board")
    migration = (MIGRATION[0], MIGRATION[1], MIGRATION[2], ("contract", "approval"),
                 MIGRATION[4])
    return plan(ref, (approval, migration))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    base = ref.example()
    with_migration = plan(ref, (MIGRATION,))
    approved = gated(ref)
    waves = ref.execution_waves(with_migration)
    gated_waves = ref.execution_waves(approved)
    validator = inspect.getsource(ref.validate)
    return {
        "base_waves": ref.execution_waves(base),
        "waves": waves, "issues": ref.validate(with_migration),
        "status": ref.plan_document(with_migration)["status"],
        "wave_two": waves[1],
        "fields": list(ref.WorkItem.__dataclass_fields__),
        "approval_words": sum(word in validator.lower()
                              for word in ("approval", "irreversible", "reversible")),
        "rules": len(re.findall(r"issues\.append", validator)),
        "gated_waves": gated_waves, "gated_count": len(gated_waves),
        "migration_wave": next(i for i, wave in enumerate(gated_waves, 1)
                               if "migration" in wave),
        "added": len(approved) - len(base),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the migration lands in wave 2 beside the documentation",
            all([result["waves"] == [["contract"], ["docs", "implementation", "migration"],
                                     ["integration"]],
                 result["issues"] == [], result["status"] == "ready",
                 result["base_waves"][1] == ["docs", "implementation"]]),
            f"adding the migration gives waves {result['waves']} with "
            f"{len(result['issues'])} issues and status {result['status']!r}: an "
            "irreversible step scheduled concurrently with the item that decides the "
            "public contract's wording",
        ),
        practice.Check(
            "FINDING: WorkItem has 5 fields and none says 'ask first'",
            all([len(result["fields"]) == 5,
                 result["fields"] == ["id", "change", "evidence", "depends_on", "proof"]]),
            f"the five commitments are {result['fields']}; approval, reversibility and "
            "blast radius have nowhere to live, so the graph is the only enforcement the "
            "plan has",
        ),
        practice.Check(
            "FINDING: 5 of the 6 rejection rules are implemented",
            all([result["rules"] == 5, result["approval_words"] == 0]),
            f"validate appends {result['rules']} kinds of issue -- duplicate ids, missing "
            f"evidence, missing proof, unknown dependencies, cycles -- and mentions "
            f"approval or reversibility {result['approval_words']} times. The sixth rule is "
            "this exercise",
        ),
        practice.Check(
            "FINDING: modelling approval as a dependency exposes a second ordering bug",
            all([result["gated_count"] == 4, result["migration_wave"] == 4,
                 result["added"] == 2,
                 result["gated_waves"][2] == ["approval", "integration"]]),
            f"the gated plan schedules {result['gated_waves']}: the migration moves to wave "
            f"{result['migration_wave']}, and the integration gate sits in wave 3 -- before "
            "the schema change it is meant to verify, because nothing declares that "
            "dependency",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
