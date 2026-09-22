"""Exercise 4 — the broad path is 405 files and the receipts name two.

    Split one broad allowed path into the smallest safe set.

Reading of the exercise: "smallest safe" is derivable rather than chosen. The
frame already says which files the change must touch -- its receipts name the
code and its acceptance command names the test -- so the allowed set is that
union and nothing else.

**ANSWER: `phases/14-agent-engineering/**` matches 405 files and the frame's
own evidence names 2.** Splitting the broad path by the receipts leaves
`38-verification-gates/code/main.py` and the test the acceptance command runs:
a **202.5x** reduction, derived from the frame rather than negotiated.

**FINDING: the overlap check is a literal set intersection, so a glob hides a
forbidden file.** `validate` computes
`set(allowed_paths) & set(forbidden_paths)`, which means allowing
`code/**` while forbidding `code/main.py` produces **0** issues -- the
forbidden file sits inside the allowed glob and the strings never match. The
one check that guards the boundary is blind to the notation the boundary is
written in.

**FINDING: the shipped example writes negative space in globs and positive
space in exact paths.** Of its **4** paths, the **2** forbidden entries are
globs (`migrations/**`, `deploy/**`) and the **2** allowed entries are exact
files. That asymmetry is good practice and it is also what makes the
intersection test meaningless: the two sets are written in different
languages.

**FINDING: a glob-aware overlap check finds 1 conflict the shipped one
misses.** Expanding both sides against the real directory and intersecting
the *file sets* catches `code/**` against `code/main.py` immediately. The fix
is to compare what the patterns match, not how they are spelled.

Structure: `expand()` resolves a pattern against a real tree; `smallest()`
derives the allowed set from the frame's receipts and acceptance.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "43-frame-the-task-before-code"
GATE = "38-verification-gates"
RECEIPTS = ["code/main.py:25", "code/main.py:174", "code/main.py:194"]
ACCEPTANCE = ["python3 -m unittest discover code/tests"]


def phase_root(ref):
    return Path(ref.__file__).resolve().parents[2]


def files(root):
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*")
                  if path.is_file() and "__pycache__" not in path.parts)


def expand(root, pattern):
    """Every real file a pattern matches, so two patterns can be compared by effect."""
    return {name for name in files(root)
            if fnmatch.fnmatch(name, pattern) or name.startswith(pattern.rstrip("*"))}


def smallest(receipts, acceptance):
    """The allowed set the frame's own evidence implies."""
    cited = {evidence.rpartition(":")[0] for evidence in receipts}
    tests = {word for command in acceptance for word in command.split()
             if "/" in word or word.endswith("tests")}
    return sorted(cited | tests)


def glob_overlap(root, allowed, forbidden):
    """Overlap by what the patterns match, not by how they are spelled."""
    left = set().union(*(expand(root, pattern) for pattern in allowed))
    right = set().union(*(expand(root, pattern) for pattern in forbidden))
    return sorted(left & right)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    root = phase_root(ref)
    broad = expand(root, f"{GATE}/../*")
    everything = files(root)
    lesson_files = expand(root, f"{GATE}/*")
    tight = smallest(RECEIPTS, ACCEPTANCE)
    blind = ref.TaskFrame(goal="g", allowed_paths=["code/**"],
                          forbidden_paths=["code/main.py"],
                          acceptance=ACCEPTANCE, facts=[], unknowns=[])
    gate_root = root / GATE
    example = ref.example()
    return {
        "broad": len(everything), "tight": len(tight), "tight_paths": tight,
        "ratio": round(len(everything) / len(tight), 1),
        "lesson_files": len(lesson_files),
        "blind_issues": ref.validate(blind),
        "glob_conflicts": glob_overlap(gate_root, ["code/**"], ["code/main.py"]),
        "example_paths": len(example.allowed_paths) + len(example.forbidden_paths),
        "example_globs": sum("*" in path for path in
                             example.allowed_paths + example.forbidden_paths),
        "forbidden_globs": sum("*" in path for path in example.forbidden_paths),
        "allowed_globs": sum("*" in path for path in example.allowed_paths),
        "broad_sample": len(broad) > 0,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the broad path is 405 files and the receipts name two",
            all([result["broad"] == 405, result["tight"] == 2, result["ratio"] == 202.5,
                 result["tight_paths"] == ["code/main.py", "code/tests"]]),
            f"the phase directory holds {result['broad']} files against the "
            f"{result['tight']} the frame's own evidence names ({result['tight_paths']}) -- "
            f"a {result['ratio']}x reduction derived from the receipts and the acceptance "
            "command rather than negotiated",
        ),
        practice.Check(
            "FINDING: the overlap check is a literal set intersection",
            all([result["blind_issues"] == [], len(result["glob_conflicts"]) == 1]),
            f"allowing code/** while forbidding code/main.py yields "
            f"{result['blind_issues']} from validate, because the check intersects strings; "
            f"expanding both sides against the real tree finds {result['glob_conflicts']}",
        ),
        practice.Check(
            "FINDING: the example writes negative space in globs and positive space in files",
            all([result["example_paths"] == 4, result["example_globs"] == 2,
                 result["forbidden_globs"] == 2, result["allowed_globs"] == 0]),
            f"of the example's {result['example_paths']} paths, the "
            f"{result['forbidden_globs']} forbidden entries are globs and the allowed ones "
            f"are exact files ({result['allowed_globs']} globs) -- good practice, and "
            "exactly what makes a string intersection meaningless",
        ),
        practice.Check(
            "FINDING: a glob-aware check finds the conflict the shipped one misses",
            all([result["glob_conflicts"] == ["code/main.py"],
                 result["lesson_files"] == 7]),
            f"comparing what the patterns match rather than how they are spelled surfaces "
            f"{result['glob_conflicts']} inside a lesson directory of "
            f"{result['lesson_files']} files",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
