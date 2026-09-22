"""Exercise 2 — the shipped example has two facts and no resolvable receipt.

    Find one claim in the frame that is actually an assumption. Replace it
    with evidence.

Reading of the exercise: an assumption is a claim whose receipt does not
resolve. The lesson ships a frame to practise on, so the honest place to look
for one is there -- and the test is mechanical: open the cited path, find the
cited line, look for the claim.

**ANSWER: both facts in `example()` are assumptions, and the second one is the
load-bearing one.** `app/accounts.py:18` and `tests/test_accounts.py:44` name
files that do not exist anywhere in the lesson directory -- **0** of the **2**
receipts resolve. "Duplicate errors use status 409" is the claim the whole
frame turns on, because it fixes the public contract the change must match.
Replaced with a receipt from this repository, the frame keeps its shape and
gains a check.

**FINDING: the frame still renders `Status: READY`.** `validate` returns **0**
issues for a frame whose every receipt is fictional, so the difference between
a researched frame and an invented one is invisible to the tool that exists to
tell them apart.

**FINDING: the lesson directory contains 2 Python files and neither is
`app/accounts.py`.** The example's paths look like a real project and belong
to none: `code/main.py` and `code/tests/test_main.py` are what ships. An
example that cannot resolve teaches the shape of a receipt and not the habit
of checking one.

**FINDING: the replacement receipt costs one line and changes the status when
it is wrong.** Citing a real path and line in this repository -- with the
claimed symbol on it -- resolves **1** of **1**; shifting that line by five
resolves **0**. That difference is the entire value of the exercise, and it is
the difference `validate` does not see.

Structure: `resolve()` opens a receipt; `shipped()` runs it over the lesson's
own example; `replacement()` builds the evidence-backed fact.
"""

from __future__ import annotations

from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "43-frame-the-task-before-code"
HERE = Path(__file__).resolve()
REPLACEMENT = ("Practice solutions import the reference module rather than forking it",
               f"{HERE.name}:1", "Exercise 2")


def lesson_root(ref):
    return Path(ref.__file__).resolve().parents[1]


def resolve(root, evidence, needle):
    """True when the cited line exists and carries the claimed text."""
    path, _, number = evidence.rpartition(":")
    target = root / path
    if not target.exists() or not number.isdigit():
        return False
    lines = target.read_text(encoding="utf-8").splitlines()
    index = int(number) - 1
    return 0 <= index < len(lines) and needle in lines[index]


def shipped(ref):
    """Every fact in the lesson's own example, with whether its receipt resolves."""
    root = lesson_root(ref)
    rows = []
    for fact in ref.example().facts:
        path, _, _ = fact.evidence.rpartition(":")
        rows.append({"claim": fact.claim, "evidence": fact.evidence,
                     "file_exists": (root / path).exists(),
                     "resolves": resolve(root, fact.evidence, fact.claim.split()[0])})
    return rows


def replacement(ref):
    claim, evidence, needle = REPLACEMENT
    here = HERE.parent
    shifted = f"{HERE.name}:{int(evidence.rpartition(':')[2]) + 5}"
    return {"resolves": resolve(here, evidence, needle),
            "shifted": resolve(here, shifted, needle),
            "fact": ref.RepositoryFact(claim, evidence)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    root = lesson_root(ref)
    rows = shipped(ref)
    swapped = replacement(ref)
    example = ref.example()
    patched = ref.TaskFrame(goal=example.goal, allowed_paths=example.allowed_paths,
                            forbidden_paths=example.forbidden_paths,
                            acceptance=example.acceptance,
                            facts=[example.facts[0], swapped["fact"]],
                            unknowns=example.unknowns)
    return {
        "facts": len(rows), "resolved": sum(row["resolves"] for row in rows),
        "files_exist": sum(row["file_exists"] for row in rows),
        "evidence": [row["evidence"] for row in rows],
        "issues": ref.validate(example),
        "status": [line for line in ref.render(example).splitlines()
                   if line.startswith("Status")][0],
        "lesson_files": sorted(path.relative_to(root).as_posix()
                               for path in root.rglob("*.py")),
        "replacement": swapped["resolves"], "shifted": swapped["shifted"],
        "patched_issues": ref.validate(patched),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: both facts in example() are assumptions",
            all([result["facts"] == 2, result["resolved"] == 0,
                 result["files_exist"] == 0,
                 "tests/test_accounts.py:44" in result["evidence"]]),
            f"{result['resolved']} of {result['facts']} receipts resolve and "
            f"{result['files_exist']} of the cited files exist: {result['evidence']}. "
            "'Duplicate errors use status 409' is the load-bearing one, because it fixes "
            "the public contract the change has to match",
        ),
        practice.Check(
            "FINDING: the frame still renders Status: READY",
            all([result["issues"] == [], result["status"] == "Status: READY"]),
            f"validate returns {result['issues']} for a frame whose every receipt is "
            f"fictional and render prints {result['status']!r}, so a researched frame and "
            "an invented one are indistinguishable to the tool meant to tell them apart",
        ),
        practice.Check(
            "FINDING: the lesson directory contains 2 Python files and neither is cited",
            all([result["lesson_files"] == ["code/main.py", "code/tests/test_main.py"],
                 result["files_exist"] == 0]),
            f"what ships is {result['lesson_files']}, and the example cites app/accounts.py "
            "and tests/test_accounts.py -- paths that look like a real project and belong "
            "to none",
        ),
        practice.Check(
            "FINDING: the replacement costs one line and fails when it is wrong",
            all([result["replacement"] is True, result["shifted"] is False,
                 result["patched_issues"] == []]),
            f"a receipt naming a real path and line in this repository resolves "
            f"({result['replacement']}) and stops resolving when the line moves by five "
            f"({result['shifted']}) -- while validate reports {result['patched_issues']} "
            "either way",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
