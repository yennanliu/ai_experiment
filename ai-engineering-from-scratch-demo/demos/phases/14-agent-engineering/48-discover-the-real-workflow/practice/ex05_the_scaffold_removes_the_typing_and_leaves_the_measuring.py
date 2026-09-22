"""Exercise 5 — the scaffold removes the typing and leaves the measuring.

    Identify a proposed feature that removes a visible step but leaves hidden
    work untouched.

Reading of the exercise: the feature has to be one somebody would actually
propose. This one already shipped: `scaffold_practice.py` writes the manifest
and the stubs, so the author never creates a file by hand. It removes the
visible step and leaves the step that costs the time.

**ANSWER: the scaffold writes every field except the one nobody can
generate.** Its template emits `verifies:` with a placeholder, and across the
**5** most recently finished lessons **0** manifests still carry it -- **25**
exercises whose threshold was written by hand after the answer was measured.
The visible step (creating **5** files per lesson) is gone; the hidden step
(deciding what number the answer asserts) is untouched.

**FINDING: the placeholder is a banned string, so the hidden work is
enforced rather than removed.** `audit_practice.py` refuses any file
containing the scaffold marker, which means the generated placeholder cannot
ship. The system's response to un-done hidden work is to block, not to do it.

**FINDING: the removed step was never the friction.** Modelling both
workflows, the pre-scaffold path has **5** steps and the scaffolded path
**4**; the friction entries -- "the threshold is only knowable after the first
run" -- are attached to the step both versions share. A feature that deletes a
step with no friction on it changes the count and not the experience.

**FINDING: the audit cannot see the saving either.** `direct_evidence_ratio`
is **1.0** for both workflows and the status is `grounded` for both, so the
artifact that is supposed to describe the workflow reports the same numbers
before and after the feature. What changed is the step list, which is the one
thing a reader has to read rather than compare.

Structure: `before()` and `after()` model the two paths; `thresholds()` counts
what the scaffold left for a human.
"""

from __future__ import annotations

from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "48-discover-the-real-workflow"
BASE = Path(__file__).resolve().parents[2]
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "demos").is_dir())
FINISHED = ("43-frame-the-task-before-code", "44-plan-from-evidence",
            "45-delegate-with-isolation", "46-turn-feedback-into-system",
            "47-outcomes-before-output")
PLACEHOLDER = "TO" + "DO state the threshold"
MEASURED = "the threshold is only knowable after the first run"


def step(ref, order, action, source, friction=""):
    return ref.WorkflowStep(order, "author", action,
                            (ref.Evidence(source, action, True, 0.9),), friction)


def before(ref):
    """The workflow the scaffold was proposed to improve."""
    return [step(ref, 1, "creates the manifest by hand", "practice.yaml"),
            step(ref, 2, "creates one stub per exercise", "practice/ex01.py"),
            step(ref, 3, "writes each answer", "practice/ex01.py"),
            step(ref, 4, "measures what the answer asserts", "tests/test_practice.py",
                 friction=MEASURED),
            step(ref, 5, "fills the threshold into the manifest", "practice.yaml",
                 friction=MEASURED)]


def after(ref):
    """The same work with the scaffold in place."""
    return [step(ref, 1, "runs the scaffolder", "scripts/scaffold_practice.py"),
            step(ref, 2, "writes each answer", "practice/ex01.py"),
            step(ref, 3, "measures what the answer asserts", "tests/test_practice.py",
                 friction=MEASURED),
            step(ref, 4, "fills the threshold into the manifest", "practice.yaml",
                 friction=MEASURED)]


def thresholds():
    """What the scaffold left for a human, across the finished lessons."""
    manifests = [BASE / lesson / "practice" / "practice.yaml" for lesson in FINISHED]
    texts = [path.read_text(encoding="utf-8") for path in manifests]
    return {"manifests": len(texts),
            "with_placeholder": sum(PLACEHOLDER in text for text in texts),
            "exercises": sum(text.count("  - index:") for text in texts),
            "verifies": sum(text.count("    verifies:") for text in texts)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    old, new = ref.audit(before(ref)), ref.audit(after(ref))
    counts = thresholds()
    template = (ROOT / "scripts" / "scaffold_practice.py").read_text(encoding="utf-8")
    audit_source = (ROOT / "scripts" / "audit_practice.py").read_text(encoding="utf-8")
    return {
        **counts,
        "template_has_placeholder": PLACEHOLDER in template,
        "banned": PLACEHOLDER.split()[0] in audit_source,
        "before_steps": len(before(ref)), "after_steps": len(after(ref)),
        "before_friction": len(old["friction_points"]),
        "after_friction": len(new["friction_points"]),
        "shared_friction": MEASURED in old["friction_points"]
        and MEASURED in new["friction_points"],
        "before_ratio": old["direct_evidence_ratio"], "after_ratio": new["direct_evidence_ratio"],
        "before_status": old["status"], "after_status": new["status"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the scaffold writes every field except the one nobody can generate",
            all([result["manifests"] == 5, result["with_placeholder"] == 0,
                 result["exercises"] == 25, result["verifies"] == 25,
                 result["template_has_placeholder"] is True]),
            f"the template emits a placeholder for verifies and "
            f"{result['with_placeholder']} of {result['manifests']} finished manifests "
            f"still carry it: {result['exercises']} exercises with {result['verifies']} "
            "thresholds, each written by hand after the answer was measured",
        ),
        practice.Check(
            "FINDING: the placeholder is a banned string",
            all([result["banned"] is True, result["with_placeholder"] == 0]),
            "audit_practice.py refuses any file carrying the scaffold marker, so the "
            "generated placeholder cannot ship; the system's response to un-done hidden "
            "work is to block rather than to do it",
        ),
        practice.Check(
            "FINDING: the removed step was never the friction",
            all([result["before_steps"] == 5, result["after_steps"] == 4,
                 result["before_friction"] == 2, result["after_friction"] == 2,
                 result["shared_friction"] is True]),
            f"the path goes from {result['before_steps']} steps to "
            f"{result['after_steps']} while the friction entries stay at "
            f"{result['after_friction']}, both attached to steps the two versions share",
        ),
        practice.Check(
            "FINDING: the audit cannot see the saving either",
            all([result["before_ratio"] == result["after_ratio"] == 1.0,
                 result["before_status"] == result["after_status"] == "grounded"]),
            f"both workflows report a ratio of {result['after_ratio']} at status "
            f"{result['after_status']!r}, so the artifact describing the workflow reports "
            "the same numbers before and after the feature",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
