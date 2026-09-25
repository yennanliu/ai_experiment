"""Exercise 5 — this build is the Research shape, and the mapper files it under MetaGPT.

    Pick your current multi-agent project. Which of the three case studies is
    the closest reference? Which design decisions from that case study have
    you NOT yet adopted? Write down one you will adopt this quarter.

Reading of the exercise: the current multi-agent project is the one writing
this file -- a lead agent that fans Phase 16's lessons out to four forked
workers at a time, each in a fresh context, then reviews, gates and commits
each lesson itself. "Closest" is scored by which case's own `patterns` list
the project already follows, labelled by hand below, and compared with what
the lesson's mapper says.

**ANSWER: Anthropic Research -- 3 of its 4 patterns adopted, against 1 of 4
for MetaGPT/ChatDev and 0 of 4 for OpenClaw.** Adopted: fresh-context
subagents (each fork starts clean on one lesson), orchestrator synthesis (the
lead writes every commit), a verification role (the audit, test, dependency
and README gates run on every lesson). Not adopted: rainbow deployment --
N/A for a batch build -- and, from the post rather than the mapper's list,
asynchronous dispatch: the post says its "lead agents execute subagents
synchronously, waiting for each set of subagents to complete", and so does
this build, which starts a batch of four only when the last one returns.
**The one to adopt this quarter: dispatch the next lesson the moment any
worker finishes.** With 4 workers and 16 lessons of uneven length, batching
pays for the slowest fork in every batch.

**FINDING: the mapper files this project under MetaGPT/ChatDev.** Described
as an engineering task with 5 agents, verification, 2 hours and no distinct
roles -- every fork runs the same role -- `map_to_case` returns
`metagpt_chatdev`, on `task_type` alone. Relabel the task "research" and the
same structure maps to Anthropic Research. The mapper reads what the work is
*about*, not how it is organised.

**FINDING: two of the mapper's five inputs cannot change its answer.** Over
every combination of task type, the three booleans and two runtimes, flipping
`verification_required` or `runtime_duration_hours` changes 0 of 64
mappings: the branch that reads them returns `anthropic_research`, the same
as the fallthrough below it. The lesson's "Classic enterprise automation ->
CrewAI or LangGraph" has no case at all: the shipped automation design maps
to MetaGPT through `roles_distinct`, and without it to Anthropic Research.

Structure: `ADOPTED` is the hand labelling of this project against each
case's patterns; `dead_inputs()` sweeps the mapper's input grid.
"""

from __future__ import annotations

import itertools

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "25-case-studies-2026-sota"
ADOPTED = {  # this build, against each case's own `patterns` list
    "fresh-context subagents", "orchestrator synthesis", "verification role",
    "structured artifact handoffs",
}
TYPES = ("research", "engineering", "population", "automation")


def dead_inputs(ref):
    """Mappings changed by flipping verification_required or runtime_duration_hours."""
    changed = total = 0
    for task, roles, network in itertools.product(TYPES, (False, True), (False, True)):
        answers = {ref.map_to_case(ref.Design("x", task, 5, verify, hours, roles, network))
                   for verify, hours in itertools.product((False, True), (0.5, 2.0))}
        total += 4
        changed += 4 * (len(answers) > 1)
    return changed, total


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    scores = {key: sum(p in ADOPTED for p in case["patterns"])
              for key, case in ref.CASES.items()}
    missing = [p for p in ref.CASES["anthropic_research"]["patterns"] if p not in ADOPTED]

    def build(task):
        return ref.map_to_case(ref.Design("phase16-build", task, 5, True, 2.0, False, False))
    automation = ref.Design("internal-automation", "automation", 3, True, 0.5, True, False)
    changed, total = dead_inputs(ref)
    return {
        "scores": scores, "closest": max(scores, key=scores.get), "missing": missing,
        "as_engineering": build("engineering"), "as_research": build("research"),
        "changed": changed, "total": total,
        "automation": ref.map_to_case(automation),
        "automation_plain": ref.map_to_case(ref.Design("a", "automation", 3, True, 0.5,
                                                       False, False)),
        "case_keys": sorted(ref.CASES),
    }


def verify(result):
    scores = result["scores"]
    return [
        practice.Check(
            "ANSWER: Anthropic Research -- 3 of its 4 patterns adopted",
            all([result["closest"] == "anthropic_research",
                 scores == {"anthropic_research": 3, "metagpt_chatdev": 1,
                            "openclaw_moltbook": 0},
                 result["missing"] == ["rainbow deployment"]]),
            f"patterns adopted per case {scores}; the Research case's missing one is "
            f"{result['missing']}, and the post's synchronous lead is the one to replace",
        ),
        practice.Check(
            "FINDING: the mapper files this project under MetaGPT/ChatDev",
            result["as_engineering"] == "metagpt_chatdev"
            and result["as_research"] == "anthropic_research",
            f"described as engineering the build maps to {result['as_engineering']}; the "
            f"identical structure labelled research maps to {result['as_research']}",
        ),
        practice.Check(
            "FINDING: two of the mapper's five inputs cannot change its answer",
            all([result["changed"] == 0, result["total"] == 64,
                 result["automation"] == "metagpt_chatdev",
                 result["automation_plain"] == "anthropic_research",
                 "automation" not in " ".join(result["case_keys"])]),
            f"flipping verification_required or runtime_duration_hours changes "
            f"{result['changed']} of {result['total']} mappings; automation maps to "
            f"{result['automation']} with distinct roles and {result['automation_plain']} "
            f"without, and no case among {result['case_keys']} is for automation",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
