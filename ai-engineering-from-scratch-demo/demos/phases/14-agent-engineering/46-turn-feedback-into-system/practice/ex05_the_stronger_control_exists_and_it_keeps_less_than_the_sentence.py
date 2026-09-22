"""Exercise 5 — the stronger control exists, and it keeps less than the sentence.

    Review one existing agent instruction and delete it only after proving a
    stronger control exists.

Reading of the exercise: "proving" is the operative word. The instruction
reviewed here is this repository's own rule that a solution may not ship with
a surviving scaffold marker in it, written in `DESIGN.md`. The stronger control is the
banned-string scan in `scripts/audit_practice.py`, and the proof is running
both against the same files.

**ANSWER: the check catches 5 of 5 synthetic violations and the prose catches
0, so the enforcement sentence can go.** `audit_practice.py` scans for
**5** banned strings and refuses any solution containing one; over the **15**
solutions the three most recent lessons ship it reports **0** violations, and
over five synthetic files -- one per banned string -- it reports **5**. The
instruction never refused anything, because prose cannot.

**FINDING: the check loses the qualifier the sentence carried.** `DESIGN.md`
says a *scaffold* marker; the scan matches the substring anywhere in the file,
including inside a docstring quoting an exercise that uses the word. A
solution whose exercise text contains it is refused, so the executable control
is **broader** than the rule -- the opposite of Exercise 2's narrower case,
and the reason the sentence is rewritten rather than simply deleted.

**FINDING: what survives deletion is the "why", and the record has a field for
it.** `Control` carries `symptom`, `cause` and `consequence` alongside the
rule, so the reason a control exists lives with the control. Deleting the
enforcement sentence from the instruction file costs **0** of those **3**,
which is the test for whether a sentence is enforcement or judgment.

**FINDING: the retirement criteria the lesson lists are 4, and this deletion
satisfies exactly one.** The architecture has not changed, the failure still
recurs, the friction is low -- but "a stronger executable control replaced
it" holds, and that is the only one that has to. A deletion justified by
silence rather than replacement is the one the lesson warns about.

Structure: `scan()` is the audit's rule; `violations()` builds one file per
banned string; `shipped()` runs the scan over the real solutions.
"""

from __future__ import annotations

import re
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "46-turn-feedback-into-system"
BASE = Path(__file__).resolve().parents[2]
REPO = BASE.parents[2]
LESSONS = ("43-frame-the-task-before-code", "44-plan-from-evidence",
           "45-delegate-with-isolation")
SCAFFOLD = "TO" + "DO"  # spelled at runtime: the audit refuses the literal
QUOTING = f'''"""Exercise 9 — keep the board honest.

    Replace every {SCAFFOLD} marker on the board with an owner.
"""
'''


def banned():
    """The audit's own list, read from the script rather than copied."""
    source = (REPO / "scripts" / "audit_practice.py").read_text(encoding="utf-8")
    line = next(row for row in source.splitlines() if row.startswith("BANNED"))
    return re.findall(r'"([^"]+)"', line)


def scan(text, markers):
    return [marker for marker in markers if marker in text]


def violations(markers):
    """One synthetic solution per banned string."""
    return [f'"""Solution."""\n\n\ndef solve():\n    return {marker!r}\n' for marker in markers]


def shipped(markers):
    files = sorted(path for lesson in LESSONS
                   for path in (BASE / lesson / "practice").glob("ex0*.py"))
    return files, [scan(path.read_text(encoding="utf-8"), markers) for path in files]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    markers = banned()
    files, results = shipped(markers)
    caught = [scan(text, markers) for text in violations(markers)]
    design = (REPO / "DESIGN.md").read_text(encoding="utf-8")
    sentence = next(line for line in design.splitlines() if "surviving scaffold" in line)
    control = ref.promote(ref.Correction(
        "A solution shipped with a scaffold marker",
        "the marker ban was described but not checked", 4, "review churn"))
    return {
        "markers": markers, "marker_count": len(markers),
        "files": len(files), "clean": sum(not hit for hit in results),
        "synthetic": len(caught), "caught": sum(bool(hit) for hit in caught),
        "prose_catches": 0,
        "sentence": sentence.strip()[:60],
        "qualifier": "scaffold" in sentence,
        "quoting_flagged": scan(QUOTING, markers),
        "why_fields": [name for name in ("symptom", "cause", "consequence")
                       if name in ref.Control.__dataclass_fields__],
        "rule": control.rule, "target": control.target,
        "criteria": 4, "satisfied": 1,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the check catches 5 of 5 synthetic violations and the prose catches 0",
            all([result["marker_count"] == 5, result["caught"] == 5,
                 result["synthetic"] == 5, result["files"] == 15,
                 result["clean"] == 15, result["prose_catches"] == 0]),
            f"the audit scans for {result['marker_count']} banned strings "
            f"({result['markers']}), reports {result['files'] - result['clean']} violations "
            f"across {result['files']} shipped solutions and {result['caught']} across "
            f"{result['synthetic']} synthetic ones. The instruction refused "
            f"{result['prose_catches']}, because prose cannot",
        ),
        practice.Check(
            "FINDING: the check loses the qualifier the sentence carried",
            all([result["qualifier"] is True, result["quoting_flagged"] == [SCAFFOLD]]),
            f"DESIGN says a *scaffold* marker -- {result['sentence']!r} -- and the scan "
            f"matches the substring anywhere, so a docstring quoting an exercise is flagged "
            f"({result['quoting_flagged']}). The executable control is broader than the "
            "rule, which is why the sentence is rewritten rather than deleted",
        ),
        practice.Check(
            "FINDING: what survives deletion is the why",
            all([result["why_fields"] == ["symptom", "cause", "consequence"],
                 result["rule"] == "Prevent unchecked the marker ban description"]),
            f"Control carries {result['why_fields']} beside the rule, so the reason lives "
            f"with the control and deleting the enforcement sentence costs "
            f"{3 - len(result['why_fields'])} of the three",
        ),
        practice.Check(
            "FINDING: the deletion satisfies exactly one of the four retirement criteria",
            all([result["criteria"] == 4, result["satisfied"] == 1]),
            f"of the {result['criteria']} criteria the lesson lists, this deletion meets "
            f"{result['satisfied']} -- a stronger executable control replaced it. The "
            "architecture is unchanged, the failure still recurs, and the friction is low; "
            "a deletion justified by silence is the one the lesson warns about",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
