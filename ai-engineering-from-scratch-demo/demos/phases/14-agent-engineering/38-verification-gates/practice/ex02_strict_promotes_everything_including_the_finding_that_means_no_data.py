"""Exercise 2 — strict promotes everything, including the finding that means "no data".

    Support a `--strict` mode that promotes every `warn` to `block`. Document
    the cases where strict mode is the right default.

Reading of the exercise: `verify` ships with the promotion and a comment
saying "opt-in by release branch only", so the mode is there and the
documentation is the work. Doing it properly means knowing what strict
actually promotes, and on the shipped findings the answer is **3** codes --
two that mean "the agent did something outside its lane" and one that means
"the gate had nothing to measure". Those want opposite defaults.

**ANSWER: strict promotes 3 of the 8 shipped codes and flips 1 of 3 demo
tasks.** The warn-severity codes are `coverage.minor_regression`,
`coverage.missing` and `scope.off_scope`; the other **5** are already
blocking. The lesson's three demo tasks go `[True, False, False]` to
`[False, False, False]`, so the mode changes exactly the task that was
passing with a warning -- which is the only task it *can* change.

**FINDING: strict promotes "no coverage data" to a blocking failure.**
`coverage.missing` fires when `coverage_report` is `None`, so under strict a
task whose coverage tool crashed is indistinguishable from one that failed
its floor. **1** of the **3** demo tasks blocks for that reason alone, and
the detail string -- "cannot enforce floor" -- is the gate admitting it does
not know.

**FINDING: promotion is all-or-nothing, so the useful middle is
unreachable.** `verify` rewrites every `warn` in one comprehension, so a team
wanting "off-scope blocks, missing coverage warns" cannot express it: of the
**8** subsets of the three warn codes, the flag reaches **2**. Per-code
severity -- the shape Lesson 33 gave rules -- costs one dict and makes strict
a policy rather than a switch.

**FINDING: the report records that strict ran and not what it changed.**
`VerdictReport` has **6** fields including `strict`, and the promoted
findings carry `severity="block"` with no memory of having been warnings. So
a reader of a strict report cannot separate the **1** genuinely blocking
finding from the **1** promoted one without re-running non-strict.

Structure: `demo_tasks()` rebuilds the lesson's three fixtures; `sweep()`
runs them under both modes.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "38-verification-gates"
ACCEPT = ["pytest -x test_app.py::test_signup"]


def art(ref, task_id, **kw):
    """One Artifacts with the lesson's happy-path defaults, overridden per task."""
    base = dict(acceptance_commands=ACCEPT, head_commit="a1b2c3d",
                feedback=[{"command": ACCEPT[0], "exit_code": 0}],
                scope_report={"forbidden_writes": [], "off_scope_writes": []},
                rule_report=[{"slug": "done/tests-pass", "passed": True}])
    return ref.Artifacts(task_id=task_id, **{**base, **kw})


def demo_tasks(ref):
    return [
        art(ref, "T-001", coverage_report={"current": 0.84, "previous": 0.85},
            scope_report={"forbidden_writes": [], "off_scope_writes": ["README.md"]}),
        art(ref, "T-002", coverage_report={"current": 0.62, "previous": 0.80},
            scope_report={"forbidden_writes": ["scripts/release.sh"],
                          "off_scope_writes": ["README.md"]},
            rule_report=[{"slug": "forbidden/no-release", "passed": False}]),
        art(ref, "T-003", feedback=[],
            rule_report=[{"slug": "done/tests-pass", "passed": False}]),
    ]


def sweep(ref, strict):
    return [ref.verify(art, strict=strict) for art in demo_tasks(ref)]


def code_severities(ref):
    """Every code the gate can emit, with the severity it is born with."""
    rows = {}
    for report in sweep(ref, strict=False):
        for finding in report.findings:
            rows[finding.code] = finding.severity
    extra = ref.verify(demo_tasks(ref)[0], coverage_floor=0.99)
    for finding in extra.findings:
        rows.setdefault(finding.code, finding.severity)
    return rows


def coverage_only(ref):
    """A task blocking under strict for missing coverage and nothing else."""
    task = art(ref, "T-cov", coverage_report=None)
    plain, hard = ref.verify(task), ref.verify(task, strict=True)
    return {"plain": plain.passed, "strict": hard.passed,
            "codes": [f.code for f in hard.findings],
            "detail": hard.findings[0].detail if hard.findings else ""}


def buckets(severities):
    """Split the emitted codes into the two severities the gate uses."""
    warn = sorted(code for code, sev in severities.items() if sev == "warn")
    block = sorted(code for code, sev in severities.items() if sev == "block")
    return warn, block


def promotion(ref):
    """What a promoted finding looks like once strict has rewritten it."""
    report = ref.verify(demo_tasks(ref)[0], strict=True)
    return {"marked": any("warn" in f.detail for f in report.findings),
            "severities": sorted({f.severity for f in report.findings})}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    severities = code_severities(ref)
    warn, block = buckets(severities)
    plain = [report.passed for report in sweep(ref, strict=False)]
    hard = [report.passed for report in sweep(ref, strict=True)]
    return {
        "severities": severities, "warn_codes": warn, "block_codes": block,
        "plain": plain, "strict": hard,
        "flipped": sum(a != b for a, b in zip(plain, hard)),
        "coverage_only": coverage_only(ref),
        "one_comprehension": inspect.getsource(ref.verify).count("if strict") == 1,
        "expressible": 2, "subsets": 2 ** len(warn),
        "report_fields": list(ref.VerdictReport.__dataclass_fields__),
        "promoted": promotion(ref),
    }


def verify(result):
    cov = result["coverage_only"]
    return [
        practice.Check(
            "ANSWER: strict promotes 3 of 8 codes and flips 1 of 3 demo tasks",
            all([len(result["warn_codes"]) == 3,
                 "scope.off_scope" in result["warn_codes"],
                 "coverage.missing" in result["warn_codes"],
                 result["plain"] == [True, False, False],
                 result["strict"] == [False, False, False],
                 result["flipped"] == 1]),
            f"the warn-severity codes are {result['warn_codes']} and the other "
            f"{len(result['block_codes'])} are already blocking. The three demo tasks go "
            f"{result['plain']} to {result['strict']} under strict -- "
            f"{result['flipped']} flip, and it is the one that was passing with a warning",
        ),
        practice.Check(
            "FINDING: strict promotes 'no coverage data' to a blocking failure",
            all([cov["plain"] is True, cov["strict"] is False,
                 cov["codes"] == ["coverage.missing"],
                 "cannot enforce floor" in cov["detail"]]),
            f"coverage.missing fires when coverage_report is None, so under strict a task "
            f"whose coverage tool crashed goes from {cov['plain']} to {cov['strict']} on "
            f"{cov['codes']} alone -- detail {cov['detail']!r}, the gate admitting it does "
            "not know",
        ),
        practice.Check(
            "FINDING: promotion is all-or-nothing",
            all([result["one_comprehension"] is True,
                 result["expressible"] == 2, result["subsets"] == 8]),
            f"verify rewrites every warn in one comprehension, so of the "
            f"{result['subsets']} subsets of the warn codes the flag reaches "
            f"{result['expressible']}. Per-code severity -- the shape Lesson 33 gave "
            "rules -- costs one dict and makes strict a policy rather than a switch",
        ),
        practice.Check(
            "FINDING: the report records that strict ran and not what it changed",
            all([len(result["report_fields"]) == 6,
                 "strict" in result["report_fields"],
                 result["promoted"]["marked"] is False,
                 result["promoted"]["severities"] == ["block"]]),
            f"VerdictReport carries {result['report_fields']}, and promoted findings "
            f"arrive as {result['promoted']['severities']} with no memory of having been "
            "warnings. A reader cannot separate a genuinely blocking finding from a "
            "promoted one without re-running non-strict",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
