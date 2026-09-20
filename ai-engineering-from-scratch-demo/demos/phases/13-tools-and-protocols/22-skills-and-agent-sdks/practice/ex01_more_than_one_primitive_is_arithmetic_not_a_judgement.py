"""Exercise 1 — more than one primitive is arithmetic, not a judgement.

    Classify five workflows from your own team using `TaskShape`. Defend every
    case where you choose more than one primitive.

Reading of the exercise: "defend" implies the multi-primitive cases are
choices, and in this model they are not -- `select_primitives` is a
disjunction over six independent booleans, so the count of primitives is the
count of true flags. What is actually being defended is the *shape*: whether a
workflow really has two responsibilities or whether two flags were set for
one. So the five cases are classified and then each multi-primitive one is
re-examined by asking which flag could be dropped.

**ANSWER: five workflows, and the three multi-primitive ones each have two
genuine responsibilities.** A nightly changelog is `AGENTS.md` + hook + code;
a PR reviewer is skill + subagent; a deploy check is skill + MCP tool. Each
pairing survives the test of dropping one flag, because removing it removes
a responsibility rather than a preference.

**FINDING: the count is the number of true flags, with no interaction.**
`select_primitives` appends once per flag and never inspects a combination,
so a shape with **6** flags returns **6** primitives and a shape with **0**
returns `("prompt",)`. Setting a flag is choosing a primitive; the function
cannot disagree with the shape it is given.

**FINDING: the output order is the function's, not the task's.** Primitives
come back in the order the `if`s are written -- `AGENTS.md`, Agent Skill, MCP
tool, hook, ordinary code, subagent -- so a reader cannot tell which is
primary. Two workflows with different centres of gravity and the same flags
produce byte-identical answers.

**FINDING: the default is the weakest primitive, and it is reached by
silence.** An unclassified shape returns `("prompt",)` -- the fallback for a
workflow nobody has characterised is the one with no durable artifact. The
model cannot distinguish "genuinely just a prompt" from "not yet analysed".

**FINDING: nothing forbids a contradictory shape.** `deterministic_logic`
and `repeatable_method` together return ordinary code *and* an Agent Skill --
a real tension, since the first says the steps are fixed and the second says
a model should follow them. **6** independent booleans admit **64** shapes and
the function accepts all of them.

Structure: `shape` builds one `TaskShape` from flag names, and `without` drops
one flag so each pairing can be tested for redundancy.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "22-skills-and-agent-sdks"
WORKFLOWS = {
    "nightly-changelog": ("repository_default", "lifecycle_event", "deterministic_logic"),
    "pr-reviewer": ("repeatable_method", "isolated_delegation"),
    "deploy-check": ("repeatable_method", "external_capability"),
    "commit-style": ("repository_default",),
    "ad-hoc-question": (),
}


def shape(ref, flags):
    return ref.TaskShape(**{flag: True for flag in flags})


def without(ref, flags, dropped):
    return ref.select_primitives(shape(ref, [f for f in flags if f != dropped]))


def drops_lose_one(ref, multi, chosen):
    """Every flag in a multi-primitive shape is carrying exactly one primitive."""
    return all(len(without(ref, flags, flag)) == len(chosen[name]) - 1
               for name, flags in multi.items() for flag in flags)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    chosen = {name: ref.select_primitives(shape(ref, flags))
              for name, flags in WORKFLOWS.items()}
    multi = {name: flags for name, flags in WORKFLOWS.items() if len(chosen[name]) > 1}
    fields = [f.name for f in dataclasses.fields(ref.TaskShape)]
    return {
        "chosen": {k: list(v) for k, v in chosen.items()},
        "multi": sorted(multi),
        "each_drop_loses_one": drops_lose_one(ref, multi, chosen),
        "all_flags": list(ref.select_primitives(shape(ref, fields))),
        "no_flags": list(ref.select_primitives(shape(ref, []))),
        "fields": fields, "shapes": 2 ** len(fields),
        "order": list(ref.select_primitives(shape(ref, fields))),
        "contradiction": list(ref.select_primitives(
            shape(ref, ["deterministic_logic", "repeatable_method"]))),
        "same_flags_same_answer": (ref.select_primitives(shape(ref, WORKFLOWS["pr-reviewer"]))
                                   == ref.select_primitives(
                                       shape(ref, reversed(WORKFLOWS["pr-reviewer"])))),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: five workflows, and each multi-primitive one survives dropping a flag",
            all([len(result["chosen"]) == 5, result["multi"] == ["deploy-check",
                                                                 "nightly-changelog",
                                                                 "pr-reviewer"],
                 result["each_drop_loses_one"],
                 result["chosen"]["pr-reviewer"] == ["Agent Skill", "subagent"],
                 result["chosen"]["ad-hoc-question"] == ["prompt"]]),
            f"the five classify as {result['chosen']}. The {len(result['multi'])} "
            f"multi-primitive cases {result['multi']} each lose exactly one primitive when a "
            "flag is dropped, so both flags are carrying a responsibility rather than a "
            "preference",
        ),
        practice.Check(
            "FINDING: the count is the number of true flags, with no interaction",
            all([len(result["all_flags"]) == 6, result["no_flags"] == ["prompt"],
                 len(result["fields"]) == 6]),
            f"a shape with all {len(result['fields'])} flags returns "
            f"{len(result['all_flags'])} primitives and one with none returns "
            f"{result['no_flags']}. select_primitives appends once per flag and never "
            "inspects a combination -- setting a flag is choosing a primitive",
        ),
        practice.Check(
            "FINDING: the output order is the function's, not the task's",
            all([result["order"] == ["AGENTS.md", "Agent Skill", "MCP tool", "hook",
                                     "ordinary code", "subagent"],
                 result["same_flags_same_answer"]]),
            f"primitives come back as {result['order']} -- the order the ifs are written -- "
            "so a reader cannot tell which is primary, and two workflows with different "
            "centres of gravity and the same flags produce byte-identical answers",
        ),
        practice.Check(
            "FINDING: the default is the weakest primitive, and contradictions are admitted",
            all([result["no_flags"] == ["prompt"],
                 result["contradiction"] == ["Agent Skill", "ordinary code"],
                 result["shapes"] == 64]),
            f"an unclassified shape returns {result['no_flags']} -- the fallback for a "
            f"workflow nobody has characterised is the one with no durable artifact. And "
            f"deterministic_logic with repeatable_method returns {result['contradiction']}, "
            f"a real tension: {len(result['fields'])} independent booleans admit "
            f"{result['shapes']} shapes and the function accepts all of them",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
