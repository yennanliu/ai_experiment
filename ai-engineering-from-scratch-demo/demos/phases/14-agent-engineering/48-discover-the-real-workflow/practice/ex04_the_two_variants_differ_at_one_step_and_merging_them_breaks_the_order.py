"""Exercise 4 — the two variants differ at one step, and merging them breaks the order.

    Model two workflow variants without merging them.

Reading of the exercise: the variants have to be real and they have to differ
for a reason. In this curriculum they do: **42** lessons ship documentation in
two languages and **12** ship English only, and the scaffolding step behaves
differently for each.

**ANSWER: both variants audit clean on their own and merging them produces an
ordering issue.** Each is a 4-step workflow differing only at step 2 -- the
bilingual variant reads two documents, the English-only variant reads one and
needs the scaffolder's missing-file branch. Audited separately they return
**0** issues each; concatenated into one list of **8** steps the orders read
`[1, 2, 3, 4, 1, 2, 3, 4]` and `audit` reports **1** issue. The model has no
way to say "these are two paths".

**FINDING: the difference is one step and the cause is a population split.**
**42** of the **54** lessons in this phase carry a `zh.md`, so the variant is
not an edge case, a legacy process or an expertise difference -- it is a
policy that changed partway through the curriculum. Averaging the two would
describe a workflow that reads **1.8** documents, which nobody does.

**FINDING: merging renumbers, and renumbering hides the variant.** Renumbering
the second variant to 5-8 makes the audit pass with **0** issues, at the cost
of asserting that the eight steps happen in sequence. A clean report is
available by making the model wrong, which is the failure mode the lesson
warns about.

**FINDING: the variant is visible in the evidence, not in the structure.**
The two workflows differ in **1** of **4** actions and in **1** evidence
source; everything else is byte-identical. Whoever reads the JSON sees two
`steps` arrays and has to diff them, because `audit` returns no field naming
the variant.

Structure: `variant()` builds either path; `merged()` shows what happens when
they are concatenated and when they are renumbered.
"""

from __future__ import annotations

from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "48-discover-the-real-workflow"
PHASE_DIR = "phases/14-agent-engineering"
BILINGUAL = "reads en.md and zh.md, emitting a manifest with both texts"
ENGLISH_ONLY = "reads en.md only, taking the scaffolder's missing-file branch"


def step(ref, order, action, source, direct=True):
    return ref.WorkflowStep(order, "author", action,
                            (ref.Evidence(source, action, direct, 0.9),))


def variant(ref, bilingual, offset=0):
    """One path through the same four steps."""
    scaffold = BILINGUAL if bilingual else ENGLISH_ONLY
    source = "docs/zh.md" if bilingual else "scripts/scaffold_practice.py"
    return [step(ref, 1 + offset, "opens the lesson's documentation", "docs/en.md"),
            step(ref, 2 + offset, scaffold, source),
            step(ref, 3 + offset, "writes one solution per exercise", "practice/ex01.py"),
            step(ref, 4 + offset, "finalizes and audits the lesson", "practice.yaml")]


def population():
    reference = parity.find_reference_root() / PHASE_DIR
    en = len(list(reference.glob("*/docs/en.md")))
    zh = len(list(reference.glob("*/docs/zh.md")))
    return en, zh, en - zh


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    both, only = variant(ref, True), variant(ref, False)
    concatenated = both + only
    renumbered = both + variant(ref, False, offset=4)
    en, zh, english_only = population()
    differing = [index for index, (left, right) in enumerate(zip(both, only), 1)
                 if left.action != right.action]
    return {
        "bilingual_issues": ref.audit(both)["issues"],
        "english_issues": ref.audit(only)["issues"],
        "bilingual_status": ref.audit(both)["status"],
        "merged_orders": [row.order for row in concatenated],
        "merged_issues": ref.audit(concatenated)["issues"],
        "renumbered_issues": ref.audit(renumbered)["issues"],
        "renumbered_steps": len(renumbered),
        "lessons": en, "zh": zh, "english_only": english_only,
        "average_docs": round((zh * 2 + english_only) / en, 1),
        "differing_steps": differing, "steps": len(both),
        "evidence_sources": len({item.source for row in both + only
                                 for item in row.evidence}),
        "names_variant": any("variant" in key for key in ref.audit(both)),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: both variants audit clean and merging them breaks the order",
            all([result["bilingual_issues"] == [], result["english_issues"] == [],
                 result["bilingual_status"] == "grounded",
                 result["merged_orders"] == [1, 2, 3, 4, 1, 2, 3, 4],
                 result["merged_issues"] == ["workflow order must be contiguous from one"]]),
            f"each variant returns {len(result['bilingual_issues'])} issues on its own; "
            f"concatenated the orders read {result['merged_orders']} and the audit reports "
            f"{result['merged_issues']} -- the model has no way to say 'these are two paths'",
        ),
        practice.Check(
            "FINDING: the difference is one step and the cause is a population split",
            all([result["lessons"] == 54, result["zh"] == 42,
                 result["english_only"] == 12, result["differing_steps"] == [2],
                 result["average_docs"] == 1.8]),
            f"{result['zh']} of {result['lessons']} lessons carry a Chinese document and "
            f"{result['english_only']} do not, so the variant is a policy that changed "
            f"partway through; the paths differ at step {result['differing_steps'][0]} and "
            f"an averaged workflow would read {result['average_docs']} documents",
        ),
        practice.Check(
            "FINDING: merging renumbers, and renumbering hides the variant",
            all([result["renumbered_issues"] == [], result["renumbered_steps"] == 8]),
            f"renumbering the second path to 5-8 makes the audit return "
            f"{result['renumbered_issues']} over {result['renumbered_steps']} steps, at the "
            "cost of asserting they happen in sequence: a clean report bought by making "
            "the model wrong",
        ),
        practice.Check(
            "FINDING: the variant is visible in the evidence, not in the structure",
            all([len(result["differing_steps"]) == 1, result["steps"] == 4,
                 result["evidence_sources"] == 5, result["names_variant"] is False]),
            f"{len(result['differing_steps'])} of {result['steps']} actions differ and the "
            f"two paths cite {result['evidence_sources']} sources between them; audit "
            "returns no field naming the variant, so a reader has to diff two steps arrays",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
