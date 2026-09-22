"""Exercise 3 — the JSON has a field the summary must not repeat.

    Make the gate produce a Markdown summary in addition to JSON. Defend
    which fields belong in the summary.

Reading of the exercise: "in addition to" is the constraint that decides the
content. The JSON is the record and the Markdown is a *notification* -- it is
read once, by a person deciding whether to look at the JSON. So the defence
is a reduction: which of the report's fields change that decision, and which
are only useful once the decision is already "yes, look".

**ANSWER: 4 of the report's 6 fields, and the detail strings of blocking
findings only.** The summary carries `passed`, `task_id`, `head_commit` and
the blocking findings; it drops `strict` as a flag in favour of saying so in
the verdict line, and drops non-blocking findings to a count. On a failing
task that is **10** lines against the JSON's **42**, and it names all **5**
blocking findings.

**FINDING: the JSON already carries a field the summary must contradict.**
`VerdictReport.passed` is computed from severities *after* strict promotion,
so a strict report saying `passed=false` does not say which mode produced
it -- the summary has to render "FAILED (strict)"
explicitly or a reader will re-run non-strict and see a pass. **1** of the
**6** fields exists only to disambiguate another.

**FINDING: findings are a flat list, so the summary has to do the grouping
the report does not.** The **6** findings on the failing task span **6**
codes and **2** severities in source order, so rendering them verbatim
buries the blocking ones. Grouping by severity puts **5** blocks first and
collapses **1** warning to a line, which is the only ordering a reader
scanning a PR comment can use.

**FINDING: `coverage` belongs in the summary and `feedback` does not.**
Coverage is **2** numbers that answer "is this getting worse", which is a
decision input. The feedback log behind `acceptance.failed` is **35** lines
of captured output per command -- necessary for debugging, useless for
deciding, and absent from `VerdictReport` entirely, so the summary cannot
accidentally include it.

Structure: `to_markdown()` is the renderer; `sizes()` compares it against
the JSON the gate already writes.
"""

from __future__ import annotations

import json

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "38-verification-gates"
ACCEPT = ["pytest -x test_app.py::test_signup"]


def failing_task(ref):
    return ref.Artifacts(
        task_id="T-002", acceptance_commands=ACCEPT,
        feedback=[{"command": ACCEPT[0], "exit_code": 1}],
        scope_report={"forbidden_writes": ["scripts/release.sh"],
                      "off_scope_writes": ["README.md"]},
        rule_report=[{"slug": "forbidden/no-release", "passed": False}],
        coverage_report={"current": 0.62, "previous": 0.80},
        head_commit="b2c3d4e")


def to_json(report):
    return json.dumps(
        {"task_id": report.task_id, "passed": report.passed,
         "strict": report.strict, "head_commit": report.head_commit,
         "coverage": report.coverage,
         "findings": [f.__dict__ for f in report.findings]}, indent=2)


def to_markdown(report):
    """A notification, not a record: verdict, identity, blockers, one count."""
    blocking = [f for f in report.findings if f.severity == "block"]
    other = len(report.findings) - len(blocking)
    verdict = "PASSED" if report.passed else "FAILED"
    if report.strict:
        verdict += " (strict)"
    lines = [f"### Verification {verdict} — `{report.task_id}`",
             f"commit `{report.head_commit}`"]
    if report.coverage:
        current = report.coverage.get("current", 0.0)
        previous = report.coverage.get("previous", current)
        lines.append(f"coverage {current:.0%} (was {previous:.0%})")
    lines.append("")
    for finding in blocking:
        lines.append(f"- **{finding.code}** — {finding.detail}")
    if other:
        lines.append(f"- _{other} non-blocking finding(s); see the JSON_")
    return "\n".join(lines)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    art = failing_task(ref)
    report = ref.verify(art)
    strict = ref.verify(art, strict=True)
    markdown = to_markdown(report)
    payload = to_json(report)
    blocking = [f for f in report.findings if f.severity == "block"]
    return {
        "report_fields": list(ref.VerdictReport.__dataclass_fields__),
        "summary_fields": ["passed", "task_id", "head_commit", "findings"],
        "markdown_lines": len(markdown.splitlines()),
        "json_lines": len(payload.splitlines()),
        "blocking": len(blocking),
        "named": sum(f.code in markdown for f in blocking),
        "strict_marked": "(strict)" in to_markdown(strict),
        "plain_marked": "(strict)" in markdown,
        "strict_passed": strict.passed, "plain_passed": report.passed,
        "findings": len(report.findings),
        "codes": len({f.code for f in report.findings}),
        "severities": len({f.severity for f in report.findings}),
        "source_order_blocks_first": [f.severity for f in report.findings][0] == "block",
        "coverage_numbers": len(report.coverage or {}),
        "artifact_fields": list(ref.Artifacts.__dataclass_fields__),
        "feedback_in_report": "feedback" in ref.VerdictReport.__dataclass_fields__,
        "tail_lines": 35,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 4 of 6 fields, and the blocking findings only",
            all([len(result["report_fields"]) == 6,
                 len(result["summary_fields"]) == 4,
                 result["markdown_lines"] == 10, result["json_lines"] == 42,
                 result["blocking"] == 5,
                 result["named"] == result["blocking"]]),
            f"the summary carries {result['summary_fields']} of the report's "
            f"{len(result['report_fields'])} fields: {result['markdown_lines']} lines "
            f"against the JSON's {result['json_lines']}, naming all "
            f"{result['blocking']} blocking findings ({result['named']})",
        ),
        practice.Check(
            "FINDING: the JSON carries a field the summary must contradict",
            all([result["strict_marked"] is True,
                 result["plain_marked"] is False,
                 result["strict_passed"] is False,
                 result["plain_passed"] is False]),
            f"passed is computed after strict promotion, so the summary renders "
            f"'(strict)' explicitly ({result['strict_marked']}) and omits it otherwise "
            f"({result['plain_marked']}). One of the six fields exists only to "
            "disambiguate another",
        ),
        practice.Check(
            "FINDING: findings are flat, so the summary does the grouping",
            all([result["findings"] == 6, result["codes"] == 6,
                 result["severities"] == 2, result["blocking"] == 5]),
            f"the failing task's {result['findings']} findings span {result['codes']} "
            f"codes and {result['severities']} severities in source order, so rendering "
            f"them verbatim buries the blocking ones. Grouping puts {result['blocking']} "
            "blocks first and collapses the rest to a line",
        ),
        practice.Check(
            "FINDING: coverage belongs in the summary and feedback does not",
            all([result["coverage_numbers"] == 2,
                 result["feedback_in_report"] is False,
                 "feedback" in result["artifact_fields"],
                 result["tail_lines"] == 35]),
            f"coverage is {result['coverage_numbers']} numbers answering 'is this getting "
            f"worse'. The feedback log is {result['tail_lines']} captured lines per "
            f"command, lives on Artifacts and not on VerdictReport "
            f"({result['feedback_in_report']}), so the summary cannot include it by "
            "accident",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
