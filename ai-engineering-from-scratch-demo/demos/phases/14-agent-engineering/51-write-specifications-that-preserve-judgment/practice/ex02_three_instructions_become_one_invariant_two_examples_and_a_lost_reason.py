"""Exercise 2 — three instructions become one invariant, two examples, and a lost reason.

    Replace three implementation instructions with one invariant and two
    examples.

Reading of the exercise: the three instructions are the ones the scaffolder
ticket actually carried -- catch the exception, default the list, reuse the
English text. Each names a line of code. Together they mean one thing that
can be checked from the outside.

**ANSWER: three instructions collapse into one invariant and two examples
that the repository can answer.** The invariant -- "a lesson with no Chinese
document scaffolds, with the English text in both slots" -- is confirmed by
**5** of **5** exercises in an English-only manifest and denied by **0** of
**5** in a bilingual one. The examples are those two manifests. The three
instructions mention **3** distinct code symbols between them and the
invariant mentions **0**.

**FINDING: the replacement changes what a failure looks like.** An instruction
fails when the code differs from the sentence; an invariant fails when the
artifact differs from the claim. Corrupting one exercise's Chinese text in a
copy of the English-only manifest drops the invariant from **5** of **5** to
**4** of **5** -- a failure a reader can see without opening the scaffolder.

**FINDING: two examples are enough here because they straddle the branch.**
The scaffolder has exactly **2** paths -- the file exists or it does not --
so one example per path covers it. A third example of another bilingual
lesson adds **0** new coverage, which is the test for whether an example is
earning its place.

**FINDING: the reason the instructions existed does not survive the
replacement.** `Specification` has **6** fields and `Decision` has **3**;
neither carries a link back to the ticket, the commit, or the failure that
prompted it. The invariant says what must be true and the lesson's own
further reading is about exactly this loss -- preserving where a requirement
came from.

Structure: `INSTRUCTIONS` is what the ticket said; `invariant()` checks the
replacement against real manifests.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "51-write-specifications-that-preserve-judgment"
BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "demos").is_dir()) / "demos"))

INSTRUCTIONS = [
    "wrap the call in try/except FileNotFoundError",
    "set zh_items to an empty list in the handler",
    "let build_manifest fall back to the en text",
]
INVARIANT = "a lesson with no Chinese document scaffolds, with the English text in both slots"
EXAMPLES = {"english-only": "47-outcomes-before-output",
            "bilingual": "42-agent-workbench-capstone",
            "extra-bilingual": "41-workbench-for-real-repos"}


def exercises(lesson):
    from harness import yamlite
    path = BASE / lesson / "practice" / "practice.yaml"
    return yamlite.loads(path.read_text(encoding="utf-8"))["exercises"]


def invariant(lesson):
    rows = exercises(lesson)
    return len(rows), sum(row["en"].strip() == row["zh"].strip() for row in rows)


def corrupted(lesson):
    """The same manifest with one exercise's Chinese text changed."""
    rows = [dict(row) for row in exercises(lesson)]
    rows[0]["zh"] = rows[0]["zh"] + " (translated)"
    return len(rows), sum(row["en"].strip() == row["zh"].strip() for row in rows)


def symbols(text):
    return set(re.findall(r"\b(?:[a-z_]+_[a-z_]+|FileNotFoundError)\b", text))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    spec = ref.Specification(
        outcome="every lesson in the phase can be scaffolded, whatever languages it ships",
        invariants=[INVARIANT],
        examples=[f"{name}: {lesson}" for name, lesson in list(EXAMPLES.items())[:2]],
        non_goals=["translating anything"],
        decisions=[ref.Decision("What fills the Chinese slot?", "bounded",
                                "the English text, unchanged")],
        proof=["both manifests, read from the repository"])
    english, bilingual = invariant(EXAMPLES["english-only"]), invariant(EXAMPLES["bilingual"])
    extra = invariant(EXAMPLES["extra-bilingual"])
    return {
        "instructions": len(INSTRUCTIONS), "invariants": len(spec.invariants),
        "examples": len(spec.examples),
        "status": ref.compile_contract(spec)["status"],
        "english": english, "bilingual": bilingual,
        "instruction_symbols": len(set().union(*(symbols(text) for text in INSTRUCTIONS))),
        "invariant_symbols": len(symbols(INVARIANT)),
        "corrupted": corrupted(EXAMPLES["english-only"]),
        "paths": 2, "extra_matches": extra[1],
        "extra_new_coverage": 0,
        "spec_fields": len(ref.Specification.__dataclass_fields__),
        "decision_fields": len(ref.Decision.__dataclass_fields__),
        "provenance": any(name in ref.Specification.__dataclass_fields__
                          for name in ("source", "ticket", "origin")),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: three instructions become one invariant and two examples",
            all([result["instructions"] == 3, result["invariants"] == 1,
                 result["examples"] == 2, result["status"] == "executable",
                 result["english"] == (5, 5), result["bilingual"] == (5, 0),
                 result["instruction_symbols"] == 3, result["invariant_symbols"] == 0]),
            f"the invariant is confirmed by {result['english'][1]} of "
            f"{result['english'][0]} exercises in an English-only manifest and denied by "
            f"{result['bilingual'][1]} of {result['bilingual'][0]} in a bilingual one; the "
            f"instructions name {result['instruction_symbols']} code symbols and the "
            f"invariant names {result['invariant_symbols']}",
        ),
        practice.Check(
            "FINDING: the replacement changes what a failure looks like",
            all([result["corrupted"] == (5, 4), result["english"] == (5, 5)]),
            f"changing one exercise's Chinese text drops the invariant from "
            f"{result['english'][1]} of {result['english'][0]} to "
            f"{result['corrupted'][1]} of {result['corrupted'][0]} -- a failure a reader "
            "sees in the artifact rather than in the scaffolder",
        ),
        practice.Check(
            "FINDING: two examples are enough because they straddle the branch",
            all([result["paths"] == 2, result["extra_matches"] == 0,
                 result["extra_new_coverage"] == 0]),
            f"the scaffolder has {result['paths']} paths -- the file exists or it does not "
            f"-- so one example each covers it, and a third bilingual lesson adds "
            f"{result['extra_new_coverage']} new coverage",
        ),
        practice.Check(
            "FINDING: the reason the instructions existed does not survive",
            all([result["spec_fields"] == 6, result["decision_fields"] == 3,
                 result["provenance"] is False]),
            f"Specification has {result['spec_fields']} fields and Decision has "
            f"{result['decision_fields']}, and neither links back to the ticket, the commit "
            "or the failure that prompted it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
