"""Exercise 3 — a malformed rule is dropped, and the build goes green.

    Wire the checker into CI: fail the build if a block-severity rule fails
    on the latest agent run.

Reading of the exercise: the gate is a filter and an `any`, which is three
lines. What makes it worth writing is that `parse_rules` drops any block it
cannot parse without saying so, so the set of rules the gate enforces is
whatever survived parsing -- and a rule that vanishes cannot fail.

**ANSWER: the gate blocks the bad run and allows the good one, on 3 block
rules.** Filtering to `block` severity and failing on any `passed == False`
gives `blocked` on the bad trace, naming **3** rules, and `allowed` on the
good one. Warn and info failures are reported and do not gate: the bad trace
fails **5** rules and **3** of them stop the build.

**FINDING: deleting the `check:` line makes a blocking rule disappear.** A
run whose *only* violation is editing `scripts/release.sh` is **blocked** by
the intact rule set, naming exactly that rule. `parse_rules` requires both
`category:` and `check:` and `continue`s otherwise, so removing that rule's
check line drops it silently -- **4** rules, **2** block rules -- and the
identical run is **allowed**. The build goes green because a rule was
malformed, which is the opposite of what a gate is for.

**FINDING: the parser reports 0 of the 2 ways it can lose a rule.**
`parse_rules` returns a list and nothing else, so a caller sees **4** rules
where it wrote **5** and has no count of blocks it skipped. Comparing
`len(rules)` against the `## ` heading count recovers it in **1** line:
**5** headings, **4** rules, **1** silently dropped.

**FINDING: the gate is only as current as the trace it reads.** It takes the
latest run's `TurnTrace`, which has **7** fields and **0** timestamp, so a
CI job handed last week's trace blocks or allows on stale evidence and cannot
tell. The same gap Lesson 32's state file has, in the artifact the build
decision is made from.

Structure: `ci_gate()` is the three lines; `heading_count()` is the missing
parser diagnostic.
"""

from __future__ import annotations

import pathlib
import re
import tempfile

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "33-instructions-as-executable-constraints"
BLOCKING = ("startup/state-file-fresh", "forbidden/no-release-script-edits",
            "done/tests-pass")


def severity_of(slug):
    return "block" if slug in BLOCKING else "warn"


def parse_with(ref, text):
    path = pathlib.Path(tempfile.mkdtemp()) / "agent-rules.md"
    path.write_text(text)
    original = ref.RULES_PATH
    ref.RULES_PATH = path
    try:
        return ref.parse_rules()
    finally:
        ref.RULES_PATH = original


def heading_count(text):
    return len(re.findall(r"^## ", text, flags=re.MULTILINE))


def traces(ref):
    return {
        "bad": ref.TurnTrace(read_state_file=False,
                             edited_files=["app.py", "scripts/release.sh"],
                             confidence=0.4, asked_for_help=False,
                             tests_exit_code=1, added_dependencies=["fastapi"]),
        "good": ref.TurnTrace(read_state_file=True,
                              edited_files=["app.py", "test_app.py"],
                              confidence=0.9, asked_for_help=False,
                              tests_exit_code=0, added_dependencies=[]),
        "release_only": ref.TurnTrace(
            read_state_file=True,
            edited_files=["app.py", "scripts/release.sh"],
            confidence=0.9, asked_for_help=False, tests_exit_code=0,
            added_dependencies=[]),
    }


def ci_gate(ref, rules, trace):
    """Fail the build if any block-severity rule failed on the latest run."""
    rows = ref.score(rules, ref.RuleChecker(), trace)
    blocking = [r for r in rows if severity_of(r["slug"]) == "block"]
    failed = [r["slug"] for r in blocking if not r["passed"]]
    return {"verdict": "blocked" if failed else "allowed", "failed": failed,
            "block_rules": len(blocking),
            "total_failed": sum(not r["passed"] for r in rows)}


def strip_check(text, slug):
    out, inside = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            inside = line[3:].strip() == slug
        if not (inside and line.strip().startswith("- check:")):
            out.append(line)
    return "\n".join(out) + "\n"


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = traces(ref)
    rules = parse_with(ref, ref.SEED_RULES)
    healthy = {name: ci_gate(ref, rules, trace) for name, trace in rows.items()}
    broken_text = strip_check(ref.SEED_RULES, "forbidden/no-release-script-edits")
    broken_rules = parse_with(ref, broken_text)
    broken = ci_gate(ref, broken_rules, rows["release_only"])
    intact = ci_gate(ref, rules, rows["release_only"])
    return {
        "rules": len(rules), "healthy": healthy, "intact": intact,
        "bad_failed": healthy["bad"]["total_failed"],
        "broken_rules": len(broken_rules),
        "broken_headings": heading_count(broken_text),
        "broken_gate": broken,
        "still_edits_release": "scripts/release.sh"
        in rows["release_only"].edited_files,
        "parse_returns": ref.parse_rules.__annotations__.get("return"),
        "dropped": heading_count(broken_text) - len(broken_rules),
        "trace_fields": list(ref.TurnTrace.__dataclass_fields__),
        "timestamps": [f for f in ref.TurnTrace.__dataclass_fields__
                       if "time" in f or "at" == f[-2:] or "run" in f],
    }


def verify(result):
    healthy, broken = result["healthy"], result["broken_gate"]
    return [
        practice.Check(
            "ANSWER: the gate blocks the bad run and allows the good one",
            all([healthy["bad"]["verdict"] == "blocked",
                 healthy["good"]["verdict"] == "allowed",
                 healthy["bad"]["block_rules"] == 3,
                 len(healthy["bad"]["failed"]) == 3,
                 result["bad_failed"] == 5, result["rules"] == 5]),
            f"filtering to block severity gives {healthy['bad']['verdict']} on the bad "
            f"trace, naming {len(healthy['bad']['failed'])} rules, and "
            f"{healthy['good']['verdict']} on the good one: {result['bad_failed']} rules "
            f"fail and {healthy['bad']['block_rules']} of them gate",
        ),
        practice.Check(
            "FINDING: deleting the check line makes a blocking rule disappear",
            all([result["broken_rules"] == 4, broken["block_rules"] == 2,
                 broken["verdict"] == "allowed",
                 result["intact"]["verdict"] == "blocked",
                 result["intact"]["failed"] == [
                     "forbidden/no-release-script-edits"],
                 result["still_edits_release"] is True]),
            f"a run whose only violation is editing scripts/release.sh is "
            f"{result['intact']['verdict']} by the intact rule set, naming "
            f"{result['intact']['failed']}. Removing that rule's check line drops it "
            f"silently -- {result['broken_rules']} rules, {broken['block_rules']} block "
            f"rules -- and the identical run is {broken['verdict']}",
        ),
        practice.Check(
            "FINDING: the parser reports nothing about what it dropped",
            all([result["broken_headings"] == 5, result["dropped"] == 1,
                 result["parse_returns"] == "list[Rule]"]),
            f"parse_rules returns {result['parse_returns']} and nothing else, so a "
            f"caller sees {result['broken_rules']} rules where the file has "
            f"{result['broken_headings']} headings -- comparing the two recovers the "
            f"{result['dropped']} dropped block in one line",
        ),
        practice.Check(
            "FINDING: the gate is only as current as the trace it reads",
            all([len(result["trace_fields"]) == 7, result["timestamps"] == []]),
            f"TurnTrace has {len(result['trace_fields'])} fields and "
            f"{len(result['timestamps'])} of them a timestamp, so a CI job handed last "
            "week's trace gates on stale evidence and cannot tell -- Lesson 32's state "
            "gap, in the artifact the build decision is made from",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
