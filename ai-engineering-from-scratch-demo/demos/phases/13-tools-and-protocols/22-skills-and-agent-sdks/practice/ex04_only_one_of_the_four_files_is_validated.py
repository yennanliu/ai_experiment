"""Exercise 4 — only one of the four files is validated.

    Split a 400-line prompt into `SKILL.md`, one reference, one script
    contract, and one output template. Keep every file responsible for one
    kind of information.

Reading of the exercise: the split is the easy half; "keep every file
responsible for one kind" is a property that has to survive someone else
editing it, so the question is what enforces it. The answer is nothing --
`validate_skill_text` takes one string and a directory name, so the other
three files are invisible to the only checker in the module. The split is
therefore written *and* a checker for it is written, because otherwise the
exercise's requirement has no way to fail.

**ANSWER: 400 lines become 4 files, each holding one kind.** `SKILL.md`
keeps routing and procedure at **36** lines; `reference.md` holds the domain
facts; `script-contract.md` holds the interface; `output-template.md` holds
the shape of the answer. Every line lands in exactly one file and the totals
reconcile.

**FINDING: the validator sees one of the four.** `validate_skill_text(text,
directory_name, allowed_runtime_extensions)` takes **3** parameters and
**0** of them is a file list, so the three companion files could be empty,
contradictory or absent and the skill still reports `valid=True`. The one-
responsibility rule is a convention with no checker, which is why this
solution writes one.

**FINDING: the frontmatter is the only part that is machine-checked at all.**
The body is checked for being non-empty and nothing else -- so a `SKILL.md`
containing the whole 400 lines validates exactly as the 38-line one does.
Splitting improves what a reader and a model see and changes **0** of the
validator's verdicts.

**FINDING: the split is load-bearing for the description, which is the only
routing signal.** `description` is what decides whether the skill is reached
at all, and it survives at **60** characters while the body it fronts shrank
by **91%**. Progressive disclosure works because the router reads one field
and the procedure is behind it.

Structure: `SECTIONS` is the source prompt classified line-kind by line-kind,
`split` assigns each kind to a file, and `misplaced` is the checker the
module does not have.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "22-skills-and-agent-sdks"
NAME = "incident-report"
DESCRIPTION = "Use when writing an incident report from an outage timeline."
KINDS = {"routing": "SKILL.md", "procedure": "SKILL.md", "reference": "reference.md",
         "contract": "script-contract.md", "template": "output-template.md"}
SOURCE = ([("routing", 6), ("procedure", 30), ("reference", 210),
           ("contract", 84), ("template", 70)])


def lines_of(kind, count):
    return [f"[{kind}] line {index}" for index in range(count)]


def split(source=SOURCE):
    """Each kind of line into the one file responsible for it."""
    files = {}
    for kind, count in source:
        files.setdefault(KINDS[kind], []).extend(lines_of(kind, count))
    return files


def misplaced(files):
    """The checker the module does not have: a line in the wrong file."""
    return {path: [line for line in body
                   if KINDS[line.split("]")[0][1:]] != path]
            for path, body in files.items()}


def skill_md(body_lines):
    return "\n".join(["---", f"name: {NAME}", f"description: {DESCRIPTION}", "---", ""]
                     + body_lines + [""])


def all_lines(files):
    return [line for body in files.values() for line in body]


def counts(files):
    return {path: len(body) for path, body in files.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    files = split()
    total = sum(count for _, count in SOURCE)
    report = ref.validate_skill_text(skill_md(files["SKILL.md"]), NAME)
    monolith = skill_md(all_lines(files))
    monolith_report = ref.validate_skill_text(monolith, NAME)
    companions = [path for path in files if path != "SKILL.md"]
    scrambled = {**files, "reference.md": files["reference.md"] + ["[template] stray"]}
    signature = inspect.signature(ref.validate_skill_text).parameters
    return {
        "files": sorted(files), "counts": counts(files),
        "total": total, "reconciles": len(all_lines(files)) == total,
        "skill_lines": len(files["SKILL.md"]),
        "valid": report.valid, "issues": [i.code for i in report.issues],
        "monolith_valid": monolith_report.valid,
        "monolith_lines": len(monolith.splitlines()),
        "params": list(signature), "file_param": [p for p in signature if "file" in p],
        "companions_ignored": len(companions),
        "misplaced_clean": counts(misplaced(files)),
        "misplaced_scrambled": sum(len(v) for v in misplaced(scrambled).values()),
        "description": report.description, "description_len": len(DESCRIPTION),
        "shrunk": round(1 - len(files["SKILL.md"]) / total, 2),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 400 lines become 4 files, each holding one kind",
            all([result["files"] == ["SKILL.md", "output-template.md", "reference.md",
                                     "script-contract.md"],
                 result["total"] == 400, result["reconciles"],
                 result["skill_lines"] == 36, result["valid"],
                 sum(result["misplaced_clean"].values()) == 0]),
            f"{result['total']} lines split into {len(result['files'])} files "
            f"{result['counts']}, reconciling exactly, with SKILL.md keeping routing and "
            f"procedure at {result['skill_lines']} lines. The checker finds "
            f"{sum(result['misplaced_clean'].values())} misplaced lines",
        ),
        practice.Check(
            "FINDING: the validator sees one of the four",
            all([result["params"] == ["text", "directory_name",
                                      "allowed_runtime_extensions"],
                 result["file_param"] == [], result["companions_ignored"] == 3,
                 result["valid"]]),
            f"validate_skill_text takes {result['params']} -- {len(result['params'])} "
            f"parameters, {len(result['file_param'])} of them a file list -- so the "
            f"{result['companions_ignored']} companion files could be empty, contradictory "
            "or absent and the skill still validates. The one-responsibility rule is a "
            "convention with no checker, which is why this solution writes one",
        ),
        practice.Check(
            "FINDING: the frontmatter is the only part that is machine-checked",
            all([result["monolith_valid"] == result["valid"],
                 result["monolith_lines"] > result["skill_lines"]]),
            f"a SKILL.md carrying all {result['total']} lines validates exactly as the "
            f"{result['skill_lines']}-line one does -- {result['monolith_valid']} against "
            f"{result['valid']} -- because the body is checked for being non-empty and "
            "nothing else. Splitting changes what a reader sees and none of the verdicts",
        ),
        practice.Check(
            "FINDING: the split is load-bearing for the description, the only routing signal",
            all([result["description"] == DESCRIPTION,
                 result["description_len"] == 60, result["shrunk"] >= 0.9,
                 result["misplaced_scrambled"] == 1]),
            f"description survives at {result['description_len']} characters while the body "
            f"it fronts shrank by {result['shrunk']:.0%}. Progressive disclosure works "
            f"because the router reads one field and the procedure is behind it -- and the "
            f"checker catches {result['misplaced_scrambled']} stray line when one is moved",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
