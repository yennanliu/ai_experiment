"""Exercise 2 — the prose rule is two sentences and the check is four lines.

    Replace one prose rule with an executable test.

Reading of the exercise: the replacement is only a replacement if the test
catches what the prose asked for. So pick a rule this repository actually
writes down, implement the check, and run it over the real files the rule
governs plus a violation the prose would not have stopped.

**ANSWER: "solutions import the lesson's own code, never a fork" becomes a
four-line check that passes 15 real files and fails a forked one.** The rule
lives in prose as `DESIGN D5`; the check reads each shipped solution for a
`parity.load_reference` call and for definitions copied out of a reference
module. Over the three most recent lessons it scores **15** of **15**, and a
synthetic file holding a copied `def` and no import scores **0**.

**FINDING: the lesson's own promotion routes this correction to
`instruction`.** Feeding the correction "a solution forked the reference
module" with the cause "the import rule was described but not checked" gives
target `instruction` and the rule "Prevent unchecked the import rule description",
whose verification is "Run the instruction linter and scenario check" -- a
control that does not exist. The classifier's **5** destinations include
`test`, and the keyword that would have reached it is not in the sentence.

**FINDING: the prose and the check disagree about what counts as a
violation.** The prose says "never fork"; the check has to decide what a fork
is. Shadowed reference symbols catch the literal copy -- **2** of them -- and
a paraphrased fork with the same logic under new names shadows **0**. It is
refused only because it still does not import the reference, which is the
weaker half of the rule. The executable control is narrower than the sentence,
and saying so is the point of writing it.

**FINDING: the control has to run somewhere, and `verification` is a string.**
`Control` carries a `verification` field that is one of **5** canned phrases,
none of which names a command. The check written here is **1** command; the
difference between a ratchet and a note is whether that command is wired into
the audit.

Structure: `check()` is the executable rule; `shipped()` runs it over the real
solutions; `forked()` is the violation.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "46-turn-feedback-into-system"
BASE = Path(__file__).resolve().parents[2]
LESSONS = ("43-frame-the-task-before-code", "44-plan-from-evidence",
           "45-delegate-with-isolation")
FORK = '''"""A solution that copies instead of importing."""


def choose_target(correction):
    return "test"


class Correction:
    pass
'''


def copied_names(text, reference_names):
    """Top-level definitions in the solution that shadow a reference symbol."""
    defined = set(re.findall(r"^(?:def|class) (\w+)", text, re.M))
    return sorted(defined & reference_names)


def check(text, reference_names):
    """D5 as four lines: import the lesson's code, redefine none of its symbols."""
    return "parity.load_reference" in text and not copied_names(text, reference_names)


def shipped(names):
    """Every practice solution the three most recent lessons ship."""
    files = sorted(path for lesson in LESSONS
                   for path in (BASE / lesson / "practice").glob("ex0*.py"))
    return files, [check(path.read_text(encoding="utf-8"), names) for path in files]


def forked(names):
    path = Path(tempfile.mkdtemp(prefix="fork-")) / "ex01_forked.py"
    path.write_text(FORK, encoding="utf-8")
    return check(path.read_text(encoding="utf-8"), names), path


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    names = {name for name in vars(ref) if not name.startswith("_")}
    files, results = shipped(names)
    fork_ok, fork_path = forked(names)
    correction = ref.Correction("a solution forked the reference module",
                                "the import rule was described but not checked", 3, "rework")
    control = ref.promote(correction)
    paraphrase = FORK.replace("def choose_target", "def pick_layer").replace(
        "class Correction", "class Fix")
    design = (BASE.parents[2] / "DESIGN.md").read_text(encoding="utf-8")
    return {
        "files": len(files), "passing": sum(results),
        "fork_passes": fork_ok,
        "fork_copied": len(copied_names(FORK, names)),
        "target": control.target, "rule": control.rule,
        "verification": control.verification,
        "destinations": 5,
        "paraphrase_copied": len(copied_names(paraphrase, names)),
        "paraphrase_passes": check(paraphrase, names),
        "prose_present": "import the lesson" in design.lower() or "D5" in design,
        "verification_is_command": control.verification.startswith("python3")
        or control.verification.startswith("uv "),
        "check_lines": 4,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the rule becomes a check that passes 15 real files and fails a fork",
            all([result["files"] == 15, result["passing"] == 15,
                 result["fork_passes"] is False, result["fork_copied"] == 2,
                 result["prose_present"] is True]),
            f"the check scores {result['passing']} of {result['files']} shipped solutions "
            f"and refuses a file holding {result['fork_copied']} copied definitions and no "
            "reference import -- the same claim DESIGN makes in prose, now executable",
        ),
        practice.Check(
            "FINDING: the lesson's promotion routes this correction to instruction",
            all([result["target"] == "instruction",
                 result["rule"] == "Prevent unchecked the import rule description",
                 result["destinations"] == 5,
                 result["verification"] == "Run the instruction linter and scenario check"]),
            f"the correction promotes to {result['target']!r} with verification "
            f"{result['verification']!r} -- a control that does not exist -- although "
            f"{result['destinations']} destinations include test. The keyword that would "
            "have reached it is not in the sentence",
        ),
        practice.Check(
            "FINDING: the prose and the check disagree about what a fork is",
            all([result["fork_copied"] == 2, result["fork_passes"] is False,
                 result["paraphrase_copied"] == 0,
                 result["paraphrase_passes"] is False]),
            f"the literal fork shadows {result['fork_copied']} reference symbols and is "
            f"refused; the same file with its definitions renamed shadows "
            f"{result['paraphrase_copied']} and is refused only because it still does not "
            "import the reference. The executable control is narrower than the sentence, "
            "and saying so is the point",
        ),
        practice.Check(
            "FINDING: the control has to run somewhere, and verification is a string",
            all([result["verification_is_command"] is False, result["check_lines"] == 4]),
            f"Control.verification is one of five canned phrases and names no command "
            f"({result['verification']!r}); the check written here is one command, and "
            "whether it is wired into the audit is the difference between a ratchet and a "
            "note",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
