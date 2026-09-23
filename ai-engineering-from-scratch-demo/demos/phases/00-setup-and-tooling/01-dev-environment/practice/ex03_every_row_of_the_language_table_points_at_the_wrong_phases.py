"""Exercise 3 — every row of the language table points at the wrong phases.

    Write a "hello world" in all four languages and run each one

Reading of the exercise: write them, run what this machine can run, and then
check the claim that sends the reader to install four toolchains -- the table
that says which phases each language is used in. All four rows are wrong, and
the worst of them is wrong by everything it names.

**ANSWER: four hello worlds, and only one of them is guaranteed to run.**
Python runs here by construction, because it is the interpreter grading this.
Node, cargo and julia are each a `shutil.which` away from absent, and the
preflight files all three as optional on the beginner route -- so "run each
one" is an instruction the lesson's own environment check does not require the
reader to be able to follow.

**FINDING: the Rust row names four phases that contain no Rust.** The table
says Rust is used in "Phases 12, 15-17". The tree has **10** `.rs` files and
**0** of them are in phases 12, 15, 16 or 17. They are in 00, 04, 06, 07 and
10 -- five phases the row does not mention, including this lesson's own
`main.rs`.

**FINDING: the Python row covers a minority of the Python.** "Phases 1-12"
accounts for **238** of the **643** `.py` files in the tree. The other **405**
are in phases 00 and 13-19, so the row describes **37%** of the thing it names
and the reader is told the language stops seven phases before it does.

**FINDING: the lesson cannot say how many languages it uses.** The header
reads "**Languages:** Python, Node.js, Rust" -- three. Thirteen lines later
the intro says "Python, TypeScript, Rust, and Julia" -- four. The exercise says
"all four languages". The lesson ships **3** of the four in its own `code/`
directory, and the missing one is Julia, whose table row points at the phase
the reader starts next.

Structure: `claimed()` parses phase numbers out of the table; `actual()` counts
source files per phase in the reference tree.
"""

from __future__ import annotations

import collections
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "01-dev-environment"
ROWS = {"Python": ".py", "TypeScript": ".ts", "Rust": ".rs", "Julia": ".jl"}
TOOLS = {"Python": sys.executable, "TypeScript": "node", "Rust": "cargo", "Julia": "julia"}


def claimed(cell):
    """Phase numbers named by a 'Used In' cell: 'Phases 12, 15-17' -> {12,15,16,17}."""
    numbers = set()
    for low, high in re.findall(r"(\d+)(?:\s*-\s*(\d+))?", cell):
        numbers.update(range(int(low), int(high or low) + 1))
    return numbers


def actual(root, suffix):
    """How many files with this suffix live in each phase of the reference tree."""
    counts = collections.Counter()
    for path in root.glob(f"*/*/code/**/*{suffix}"):
        if "__pycache__" not in path.parts:
            counts[int(path.relative_to(root).parts[0].split("-")[0])] += 1
    return counts


def table(doc):
    """The Used In cell of each language row, keyed by language."""
    cells = {}
    for line in doc.splitlines():
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) == 3 and parts[0] in ROWS:
            cells[parts[0]] = parts[1]
    return cells


def hello():
    """Write and run a Python hello world; report which other toolchains exist."""
    script = pathlib.Path(tempfile.mkdtemp(), "hello.py")
    script.write_text('print("hello world")\n', encoding="utf-8")
    out = subprocess.run([sys.executable, str(script)], capture_output=True,
                         text=True, timeout=30)
    return out.stdout.strip(), [n for n, t in TOOLS.items() if shutil.which(t)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "verify")
    root = parity.lesson_dir(PHASE, LESSON).parent.parent
    doc = parity.doc_text(PHASE, LESSON)
    cells = table(doc)
    rows = {}
    for language, suffix in ROWS.items():
        counts, named = actual(root, suffix), claimed(cells[language])
        rows[language] = {"cell": cells[language], "total": sum(counts.values()),
                          "inside": sum(counts[p] for p in named), "phases": sorted(counts)}
    printed, present = hello()
    shipped = {p.suffix for p in (parity.lesson_dir(PHASE, LESSON) / "code").iterdir()}
    header = re.search(r"(?m)^\*\*Languages:\*\* (.+)$", doc).group(1)
    return {
        "rows": rows,
        "printed": printed,
        "present": present,
        "shipped": sorted(s for s in shipped if s in ROWS.values()),
        "header_languages": len(header.split(",")),
        "optional_on_beginner": len({"node", "cargo", "julia"}.intersection(ref.ROUTES["beginner"].optional)),
    }


def verify(result):
    rust, python = result["rows"]["Rust"], result["rows"]["Python"]
    share = 100 * python["inside"] / python["total"]
    return [
        practice.Check(
            "ANSWER: four hello worlds, and only one of them is guaranteed to run",
            all([result["printed"] == "hello world", "Python" in result["present"],
                 result["optional_on_beginner"] == 3]),
            f"the Python one prints {result['printed']!r}; {len(result['present'])} of the "
            f"four toolchains are on PATH here ({', '.join(result['present'])}), and all "
            f"{result['optional_on_beginner']} others are optional on the default route",
        ),
        practice.Check(
            "FINDING: the Rust row names four phases that contain no Rust",
            all([rust["inside"] == 0, rust["total"] == 10,
                 set(rust["phases"]) == {0, 4, 6, 7, 10}]),
            f"the row reads {rust['cell']!r}; the tree has {rust['total']} .rs files and "
            f"{rust['inside']} are in those phases -- they are in "
            f"{', '.join('%02d' % p for p in rust['phases'])}, including this lesson's main.rs",
        ),
        practice.Check(
            "FINDING: the Python row covers a minority of the Python",
            all([python["inside"] == 238, python["total"] == 643, share < 50]),
            f"{python['cell']!r} accounts for {python['inside']} of {python['total']} .py "
            f"files, so {share:.0f}% -- the other {python['total'] - python['inside']} are in "
            "phases 00 and 13-19, seven phases past where the row says it stops",
        ),
        practice.Check(
            "FINDING: the lesson cannot say how many languages it uses",
            all([result["header_languages"] == 3, len(result["shipped"]) == 3,
                 ".jl" not in result["shipped"]]),
            f"the header names {result['header_languages']} languages, the intro and the "
            f"exercise name four, and code/ ships {len(result['shipped'])} "
            f"({', '.join(result['shipped'])}) -- Julia is missing, and its row points at "
            "the phase the reader starts next",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
