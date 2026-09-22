"""Exercise 1 — a frame whose receipts all resolve, and a validator that never checks.

    Frame a real bug from one of your repositories without proposing a
    solution.

Reading of the exercise: "without proposing a solution" is the constraint that
makes the frame worth writing, and "real" is the one that makes it checkable.
The bug framed here is one this repository already measured: Lesson 38's
verification gate writes a signed override log and never reads it, so a valid
override changes no verdict.

**ANSWER: the frame validates READY with 3 facts whose receipts all resolve to
the lines they cite.** Each `path:line` is opened and the claimed symbol is
found on that line: `OVERRIDES_PATH` is defined at line **25**,
`record_override` at **174**, `verify_signature` at **194**, and
`inspect.getsource(verify)` contains `OVERRIDES_PATH` **0** times. The goal
names a behaviour -- a signed override must change the verdict -- and no
allowed path is a proposal about how.

**FINDING: `validate` accepts any non-empty string as evidence.** The check is
`if not fact.evidence.strip()`, so a fact citing "obviously" passes and the
frame renders `Status: READY`. **5** of the **6** validator branches are
emptiness tests -- four on lists and one on a fact's evidence string -- and
**0** of them resolve anything.

**FINDING: resolving receipts is 4 lines and rejects the frame that lies.**
Opening the cited file and looking for the claimed symbol on the cited line
turns a fact into a check: the honest frame scores **3** of **3** and a frame
whose line numbers are off by ten scores **0**. The lesson's own rule -- "a
file path and line is enough" -- is only true if something reads it.

**FINDING: the frame that proposes a solution still validates.** Replacing the
goal with "make verify() read overrides.jsonl" leaves **0** blocking issues,
because nothing in the validator distinguishes a behaviour from an
implementation. The discipline the lesson teaches is entirely in the author.

Structure: `frame()` builds the real frame; `resolve()` opens each receipt and
checks the claim against the line it names.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "43-frame-the-task-before-code"
GATE = "38-verification-gates"

# (claim, path:line, the symbol that must appear on that line)
FACTS = [
    ("The gate declares an override log path", "code/main.py:25", "OVERRIDES_PATH"),
    ("Overrides are written and signed", "code/main.py:174", "def record_override"),
    ("Signatures are verified on read", "code/main.py:194", "def verify_signature"),
]


def gate_root():
    module = parity.load_reference(PHASE, GATE, "main")
    return Path(module.__file__).resolve().parents[1], module


def frame(ref, goal=None, facts=None):
    return ref.TaskFrame(
        goal=goal or "A signed override lifts the finding it names on the verdict it names",
        allowed_paths=["code/main.py", "code/tests/test_main.py"],
        forbidden_paths=["docs/en.md", "outputs/**"],
        acceptance=["python3 -m unittest discover code/tests"],
        facts=[ref.RepositoryFact(claim, evidence)
               for claim, evidence, _ in (facts if facts is not None else FACTS)],
        unknowns=["Whether an override is scoped to one head_commit or to every future one"])


def resolve(root, facts):
    """Open each receipt and confirm the claimed symbol sits on the cited line."""
    resolved = []
    for _, evidence, symbol in facts:
        path, _, number = evidence.rpartition(":")
        lines = (root / path).read_text(encoding="utf-8").splitlines()
        index = int(number) - 1
        resolved.append(0 <= index < len(lines) and symbol in lines[index])
    return resolved


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    root, gate = gate_root()
    honest = frame(ref)
    shifted = [(claim, f"{path}:{int(number) + 10}", symbol)
               for claim, evidence, symbol in FACTS
               for path, _, number in [evidence.rpartition(":")]]
    vague = [(claim, "obviously", symbol) for claim, _, symbol in FACTS]
    validator = inspect.getsource(ref.validate)
    return {
        "issues": ref.validate(honest),
        "status_line": [line for line in ref.render(honest).splitlines()
                        if line.startswith("Status")][0],
        "resolved": resolve(root, FACTS), "shifted": resolve(root, shifted),
        "reads_overrides": inspect.getsource(gate.verify).count("OVERRIDES_PATH"),
        "vague_issues": ref.validate(frame(ref, facts=vague)),
        "branches": len(re.findall(r"issues\.append", validator)),
        "emptiness": validator.count("if not "),
        "resolves": validator.count("read_text") + validator.count("open("),
        "proposal_issues": ref.validate(frame(ref, goal="make verify() read overrides.jsonl")),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the frame validates READY with 3 receipts that all resolve",
            all([result["issues"] == [], result["status_line"] == "Status: READY",
                 result["resolved"] == [True, True, True],
                 result["reads_overrides"] == 0]),
            f"validate returns {result['issues']} and the frame renders "
            f"{result['status_line']!r}; the three receipts resolve "
            f"{result['resolved']} against the cited lines, and verify() mentions "
            f"OVERRIDES_PATH {result['reads_overrides']} times -- the bug, stated as a "
            "behaviour rather than a patch",
        ),
        practice.Check(
            "FINDING: validate accepts any non-empty string as evidence",
            all([result["vague_issues"] == [], result["branches"] == 6,
                 result["emptiness"] == 5, result["resolves"] == 0]),
            f"a fact citing 'obviously' yields {result['vague_issues']}: of the "
            f"{result['branches']} validator branches, {result['emptiness']} are emptiness "
            f"tests and {result['resolves']} open a file",
        ),
        practice.Check(
            "FINDING: resolving receipts is four lines and rejects the frame that lies",
            all([sum(result["resolved"]) == 3, sum(result["shifted"]) == 0]),
            f"opening the cited file and looking for the claimed symbol scores "
            f"{sum(result['resolved'])} of 3 on the honest frame and "
            f"{sum(result['shifted'])} when every line number is off by ten -- 'a file path "
            "and line is enough' holds only if something reads it",
        ),
        practice.Check(
            "FINDING: the frame that proposes a solution still validates",
            result["proposal_issues"] == [],
            f"replacing the goal with an implementation instruction leaves "
            f"{result['proposal_issues']} blocking issues, because nothing in the validator "
            "separates a behaviour from a patch. That discipline lives in the author",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
