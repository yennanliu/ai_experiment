"""Exercise 2 — peer review implies four roles and six artifacts, and the mapper needs one boolean.

    Read MetaGPT Sections 3-4 (arXiv:2308.00352). Encode one SOP from your own
    domain (not software) as role prompts. How many roles does the SOP imply?

Reading of the exercise: the SOP is journal peer review, written the way
MetaGPT §3.1 writes its roles -- each step names who acts, what structured
artifact it reads and what it publishes -- so "how many roles" is counted
from the steps rather than chosen, and it can be counted two ways.

**ANSWER: four roles by actor, six by MetaGPT's one-role-one-artifact
convention.** Seven steps: author submits a manuscript; editor makes a desk
decision; editor sends invitations; two reviewers each write a report; editor
writes the decision letter; production makes the proof. Distinct actors:
author, editor, reviewer, production -- **4**. Distinct artifact schemas:
**6**. MetaGPT's five roles map one-to-one onto five artifacts (PRD, system
design, task list, code, tests); peer review does not, because the editor
owns **3** of the 6. Keeping 4 roles means the editor's prompt carries three
output schemas; splitting it gives 6 narrow roles, which is MetaGPT's move.

**FINDING: the SOP is a DAG six stages deep, not a seven-step chain.** The two
reviewer steps depend on the same inputs and not on each other, so a
MetaGPT-style publish-subscribe pool runs them together: the longest path is
6 stages. The reviewer is the only role instantiated twice -- the one place
the SOP wants parallel workers of one role.

**FINDING: the lesson's mapper decides "role decomposition" on one boolean the
user sets.** Encoded as a reference `Design` (automation, 5 agents,
verification required, 72 hours), peer review maps to MetaGPT/ChatDev with
`roles_distinct=True` and to Anthropic Research with `roles_distinct=False`
-- a parallel research supervisor for a fixed-order SOP. No field describes
order, artifacts or handoff contracts, which are what MetaGPT's §3 is about.

Structure: `STEPS` is the SOP as (role, reads, publishes); `ROLE_PROMPTS` are
the role prompts the exercise asks for, generated from it.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "25-case-studies-2026-sota"
STEPS = [  # (role, artifacts read, artifact published)
    ("author", (), "manuscript"),
    ("editor", ("manuscript",), "desk_decision"),
    ("editor", ("desk_decision",), "invitations"),
    ("reviewer", ("manuscript", "invitations"), "report"),
    ("reviewer", ("manuscript", "invitations"), "report"),
    ("editor", ("report",), "decision_letter"),
    ("production", ("decision_letter", "manuscript"), "proof"),
]


def role_prompts(steps):
    """One prompt per role: what it subscribes to and which schemas it must publish."""
    prompts = {}
    for role, reads, publishes in steps:
        entry = prompts.setdefault(role, {"reads": set(), "publishes": set()})
        entry["reads"].update(reads)
        entry["publishes"].add(publishes)
    return {role: (f"You are the {role}. Subscribe to {sorted(e['reads']) or 'the request'}. "
                   f"Publish only {sorted(e['publishes'])}.", len(e["publishes"]))
            for role, e in prompts.items()}


def depth(steps):
    """Longest path: a step's stage is one past the latest stage producing what it reads."""
    stage_of, stages = {}, []
    for _, reads, publishes in steps:
        stage = 1 + max((stage_of[a] for a in reads), default=0)
        stages.append(stage)
        stage_of[publishes] = max(stage_of.get(publishes, 0), stage)
    return max(stages)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    prompts = role_prompts(STEPS)

    def mapped(roles_distinct):
        return ref.map_to_case(ref.Design("peer-review", "automation", 5, True, 72.0,
                                          roles_distinct, False))
    return {
        "steps": len(STEPS), "roles": sorted(prompts),
        "artifacts": len({p for _, _, p in STEPS}),
        "editor_owns": prompts["editor"][1],
        "reviewer_instances": sum(r == "reviewer" for r, _, _ in STEPS),
        "depth": depth(STEPS),
        "with_roles": mapped(True), "without_roles": mapped(False),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: four roles by actor, six by one-role-one-artifact",
            all([len(result["roles"]) == 4, result["artifacts"] == 6,
                 result["editor_owns"] == 3]),
            f"{result['steps']} steps, actors {result['roles']}, {result['artifacts']} "
            f"artifact schemas; the editor publishes {result['editor_owns']} of them, "
            "where MetaGPT's five roles map one-to-one onto five artifacts",
        ),
        practice.Check(
            "FINDING: the SOP is a DAG six stages deep, not a seven-step chain",
            result["depth"] == 6 and result["reviewer_instances"] == 2,
            f"longest path {result['depth']} of {result['steps']} steps: the "
            f"{result['reviewer_instances']} reviewer steps read the same inputs and "
            "run together",
        ),
        practice.Check(
            "FINDING: the mapper decides role decomposition on one boolean",
            result["with_roles"] == "metagpt_chatdev"
            and result["without_roles"] == "anthropic_research",
            f"the same SOP maps to {result['with_roles']} with roles_distinct=True and "
            f"to {result['without_roles']} with it False",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
