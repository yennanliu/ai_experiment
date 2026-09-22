"""Exercise 1 — the ticket had six sentences, and four of them were one surface.

    Convert a backlog ticket into the six specification surfaces.

Reading of the exercise: the ticket converted here is a real one from this
repository -- the scaffolder raised `FileNotFoundError` on lessons that ship
English documentation only, and the fix landed in commit `98c2d51`. Writing
it as six surfaces is what shows which parts of the original sentence were
requirements and which were someone's implementation.

**ANSWER: the ticket becomes a specification that validates `executable`,
and 4 of its 6 sentences belong to a single surface.** The outcome, the two
invariants, the two examples, the two non-goals, the three decisions and the
two proofs compile with **0** issues. Four of the original sentences --
"catch the error", "default to an empty list", "keep the en text", "leave
the bilingual path alone" -- are all *decisions*, and two of them turn out to
be delegated.

**FINDING: the validator exempts the one mode the docs require an
explanation for.** `validate` demands a rationale when the mode is not
`delegated`, and the lesson's text says a delegated decision is one the agent
"owns and must explain". The shipped example has **1** delegated decision
with an empty rationale and compiles clean, so the surface that carries the
agent's reasoning is the surface nothing checks.

**FINDING: the invariant is checkable against the repository and the
instruction was not.** "A lesson with no Chinese document still scaffolds,
with the English text in both slots" is true of the manifest this repository
generated for an English-only lesson: **5** of **5** exercises carry
identical `en` and `zh` text, against **0** of **5** for a bilingual one.
"Default `zh_items` to an empty list" is an instruction that cannot be
checked from the artifact at all.

**FINDING: `compile_contract` routes decisions by mode and never counts
them.** It returns **6** keys, three of which are the mode buckets, so a
specification with **3** decisions and one with **30** look the same shape.
The question the reviewer actually has -- how much judgment was delegated --
is the arithmetic the document does not do.

Structure: `TICKET` is the original text; `spec()` is the conversion;
`manifests()` checks the invariant against the real files.
"""

from __future__ import annotations

import sys
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "51-write-specifications-that-preserve-judgment"
BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "demos").is_dir()) / "demos"))

TICKET = [
    "the scaffolder crashes on lessons that ship English documentation only",
    "catch the FileNotFoundError",
    "default the Chinese items to an empty list",
    "keep the English text in the Chinese slot",
    "leave the bilingual path alone",
    "every lesson from 43 onward needs this",
]
DECISION_SENTENCES = TICKET[1:5]


def spec(ref, production_mode="locked"):
    return ref.Specification(
        outcome="every lesson in the phase can be scaffolded, whatever languages it ships",
        invariants=["a lesson with no Chinese document still scaffolds",
                    "a bilingual lesson keeps both texts distinct"],
        examples=["an English-only lesson produces a manifest whose zh text equals its en",
                  "a bilingual lesson produces a manifest whose zh text differs"],
        non_goals=["translating anything", "changing the exercise text"],
        decisions=[
            ref.Decision("Which exception does the fallback catch?", "delegated", ""),
            ref.Decision("May the fallback invent Chinese text?", production_mode,
                         "invented text would ship as the lesson's own words"),
            ref.Decision("What goes in the Chinese slot when the file is missing?",
                         "bounded", "the English text, unchanged"),
        ],
        proof=["the English-only lesson's manifest carries identical en and zh text",
               "the bilingual lesson's manifest carries different text"])


def manifests():
    """The invariant, checked against manifests this repository generated."""
    from harness import yamlite
    rows = {}
    for lesson in ("42-agent-workbench-capstone", "47-outcomes-before-output"):
        path = BASE / lesson / "practice" / "practice.yaml"
        exercises = yamlite.loads(path.read_text(encoding="utf-8"))["exercises"]
        rows[lesson[:2]] = (len(exercises),
                            sum(row["en"].strip() == row["zh"].strip() for row in exercises))
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    contract = ref.compile_contract(spec(ref))
    shipped = ref.example()
    delegated = [item for item in shipped.decisions if item.mode == "delegated"]
    return {
        "sentences": len(TICKET), "decision_sentences": len(DECISION_SENTENCES),
        "status": contract["status"], "issues": contract["issues"],
        "surfaces": len(ref.Specification.__dataclass_fields__),
        "delegated": contract["agent_may_decide"],
        "bounded": [row["question"] for row in contract["bounded_decisions"]],
        "checkpoint": contract["human_checkpoint"],
        "shipped_delegated_without_rationale":
            sum(not item.rationale.strip() for item in delegated),
        "shipped_issues": ref.validate(shipped),
        "manifests": manifests(),
        "keys": sorted(contract),
        "counts_decisions": any("count" in key or "total" in key for key in contract),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the ticket compiles executable and 4 of 6 sentences are decisions",
            all([result["sentences"] == 6, result["decision_sentences"] == 4,
                 result["status"] == "executable", result["issues"] == [],
                 result["surfaces"] == 6, len(result["delegated"]) == 1,
                 len(result["checkpoint"]) == 1]),
            f"the conversion compiles {result['status']!r} with "
            f"{len(result['issues'])} issues across {result['surfaces']} surfaces; "
            f"{result['decision_sentences']} of the ticket's {result['sentences']} "
            f"sentences are decisions, of which {len(result['delegated'])} is delegated and "
            f"{len(result['checkpoint'])} needs a human",
        ),
        practice.Check(
            "FINDING: the validator exempts the one mode the docs require an explanation for",
            all([result["shipped_delegated_without_rationale"] == 1,
                 result["shipped_issues"] == []]),
            f"the shipped example carries "
            f"{result['shipped_delegated_without_rationale']} delegated decision with an "
            f"empty rationale and validates with {len(result['shipped_issues'])} issues, "
            "although the text says a delegated decision is one the agent owns and must "
            "explain",
        ),
        practice.Check(
            "FINDING: the invariant is checkable and the instruction was not",
            all([result["manifests"]["47"] == (5, 5),
                 result["manifests"]["42"] == (5, 0)]),
            f"the English-only lesson's manifest carries identical en and zh text in "
            f"{result['manifests']['47'][1]} of {result['manifests']['47'][0]} exercises "
            f"against {result['manifests']['42'][1]} of {result['manifests']['42'][0]} for "
            "a bilingual one; 'default the Chinese items to an empty list' cannot be "
            "checked from the artifact at all",
        ),
        practice.Check(
            "FINDING: compile_contract routes decisions by mode and never counts them",
            all([len(result["keys"]) == 6, result["counts_decisions"] is False,
                 len(result["delegated"]) + len(result["bounded"])
                 + len(result["checkpoint"]) == 3]),
            f"the document returns {result['keys']}: three mode buckets and no total, so a "
            "specification with three decisions and one with thirty look the same shape and "
            "'how much judgment was delegated' is arithmetic the reader has to do",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
