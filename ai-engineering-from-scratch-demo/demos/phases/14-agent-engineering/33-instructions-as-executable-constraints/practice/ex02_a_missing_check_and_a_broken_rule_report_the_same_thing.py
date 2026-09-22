"""Exercise 2 — a missing check and a broken rule report the same thing.

    Extend the checker so a rule can carry a severity (`block`, `warn`,
    `info`) and the report aggregates accordingly.

Reading of the exercise: severity is a field and a group-by, which is twenty
lines. What the field exposes is that `score` already has two ways to emit
`passed: False` -- the check ran and said no, or the check was not found --
and once a severity is attached to that verdict, a typo in a `check:` line
becomes a blocking failure.

**ANSWER: a severity field, and a report that aggregates 5 rules into 3
buckets.** Parsing `- severity:` with a default of `block` and grouping the
results gives, on the shipped bad trace, `block` **0/3** passing, `warn`
**0/1** and `info` **0/1**; on the good trace **3/3**, **1/1**, **1/1**. The
parser change is **1** regex and the default is what makes old rules keep
working.

**FINDING: `score` returns `passed: False` for a rule whose check does not
exist.** Renaming one `check:` target to `tests_pas` leaves the rule count at
**5** and the bad trace's failures at **5** -- identical output -- while the
good trace drops from **5** passing to **4**. A typo is indistinguishable
from a violation, and under a `block` severity it fails the build.

**FINDING: the distinction is recoverable and the shipped result type cannot
carry it.** `score` emits **3** keys per rule -- slug, category, passed --
so a fourth value (`error`, `pass`, `fail`) has nowhere to go. Adding it
separates **1** configuration bug from **4** genuine failures on the same
trace, and makes "the build is red" answerable without reading the rules
file.

**FINDING: severity has to default, because the shipped rules carry none.**
All **5** seed rules omit the field, so a parser that requires it drops
**5** of **5** and a parser that defaults to `block` keeps them and makes
every existing rule blocking. Defaulting to `warn` instead would silently
downgrade `no-release-script-edits`, which is the rule most obviously meant
to block.

Structure: `parse_with_severity()` adds one regex; `aggregate()` is the
group-by the report needs.
"""

from __future__ import annotations

import pathlib
import re
import tempfile

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "33-instructions-as-executable-constraints"
SEVERITIES = {"startup/state-file-fresh": "block",
              "forbidden/no-release-script-edits": "block",
              "done/tests-pass": "block",
              "uncertainty/open-question-note": "warn",
              "approval/new-dependency": "info"}
SEVERITY_PATTERN = re.compile(r"-\s*severity:\s*(\S+)")


def with_severities(text):
    """The seed rules, annotated with a severity line under each check line."""
    out, slug = [], None
    for line in text.splitlines():
        out.append(line)
        if line.startswith("## "):
            slug = line[3:].strip()
        elif line.startswith("- check:") and slug in SEVERITIES:
            out.append(f"- severity: {SEVERITIES[slug]}")
    return "\n".join(out) + "\n"


def parse_with(ref, text, default="block"):
    path = pathlib.Path(tempfile.mkdtemp()) / "agent-rules.md"
    path.write_text(text)
    original, ref.RULES_PATH = ref.RULES_PATH, path
    try:
        rules = ref.parse_rules()
    finally:
        ref.RULES_PATH = original
    found = [m.group(1) if (m := SEVERITY_PATTERN.search(b)) else default
             for b in text.split("\n## ")[1:]]
    return rules, dict(zip([r.slug for r in rules], found))


def traces(ref):
    return {
        "bad": ref.TurnTrace(False, ["app.py", "scripts/release.sh"], 0.4,
                             False, 1, ["fastapi"]),
        "good": ref.TurnTrace(True, ["app.py", "test_app.py"], 0.9, False, 0, []),
    }


def aggregate(ref, rules, severities, trace):
    rows = ref.score(rules, ref.RuleChecker(), trace)
    buckets = {}
    for row in rows:
        level = severities[row["slug"]]
        passed, total = buckets.get(level, (0, 0))
        buckets[level] = (passed + row["passed"], total + 1)
    return buckets


def triaged(ref, rules, trace):
    """The third verdict the shipped result type cannot carry."""
    checker, out = ref.RuleChecker(), {"pass": 0, "fail": 0, "error": 0}
    for rule in rules:
        fn = getattr(checker, rule.check, None)
        out["error" if fn is None else ("pass" if fn(trace) else "fail")] += 1
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rules, severities = parse_with(ref, with_severities(ref.SEED_RULES))
    rows = traces(ref)
    typo_text = ref.SEED_RULES.replace("check: tests_pass", "check: tests_pas")
    typo_rules, _ = parse_with(ref, typo_text)
    plain, _ = parse_with(ref, ref.SEED_RULES)
    dropped, _ = parse_with(ref, ref.SEED_RULES, default=None)
    return {
        "rules": len(rules), "buckets": len(set(severities.values())),
        "bad": aggregate(ref, rules, severities, rows["bad"]),
        "good": aggregate(ref, rules, severities, rows["good"]),
        "typo_rules": len(typo_rules),
        "typo_bad_fail": sum(not r["passed"] for r in
                             ref.score(typo_rules, ref.RuleChecker(), rows["bad"])),
        "plain_bad_fail": sum(not r["passed"] for r in
                              ref.score(plain, ref.RuleChecker(), rows["bad"])),
        "typo_good_pass": sum(r["passed"] for r in
                              ref.score(typo_rules, ref.RuleChecker(), rows["good"])),
        "plain_good_pass": sum(r["passed"] for r in
                               ref.score(plain, ref.RuleChecker(), rows["good"])),
        "score_keys": sorted(ref.score(plain, ref.RuleChecker(), rows["bad"])[0]),
        "triaged": triaged(ref, typo_rules, rows["bad"]),
        "seed_has_severity": sum(bool(SEVERITY_PATTERN.search(block))
                                 for block in ref.SEED_RULES.split("\n## ")[1:]),
        "dropped": len(dropped),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a severity field aggregating 5 rules into 3 buckets",
            all([result["rules"] == 5, result["buckets"] == 3,
                 result["bad"] == {"block": (0, 3), "warn": (0, 1), "info": (0, 1)},
                 result["good"] == {"block": (3, 3), "warn": (1, 1),
                                    "info": (1, 1)}]),
            f"parsing a severity line and grouping gives {result['bad']} on the bad "
            f"trace and {result['good']} on the good one, as (passing, total) per "
            f"bucket: {result['rules']} rules, {result['buckets']} buckets",
        ),
        practice.Check(
            "FINDING: score returns passed False for a check that does not exist",
            all([result["typo_rules"] == 5, result["typo_bad_fail"] == 5,
                 result["plain_bad_fail"] == 5,
                 result["typo_good_pass"] == 4, result["plain_good_pass"] == 5]),
            f"a typo'd check target leaves {result['typo_rules']} rules and the bad "
            f"trace failing {result['typo_bad_fail']}, identical to "
            f"{result['plain_bad_fail']} -- while the good trace drops from "
            f"{result['plain_good_pass']} passing to {result['typo_good_pass']}. Under "
            "block, a typo fails the build",
        ),
        practice.Check(
            "FINDING: the shipped result type cannot carry the distinction",
            all([result["score_keys"] == ["category", "passed", "slug"],
                 result["triaged"] == {"pass": 0, "fail": 4, "error": 1}]),
            f"score emits {result['score_keys']} per rule, so a third verdict has nowhere "
            f"to go. Triaging the same trace separates {result['triaged']}: one "
            "configuration bug from four genuine failures",
        ),
        practice.Check(
            "FINDING: severity has to default, because the shipped rules carry none",
            all([result["seed_has_severity"] == 0, result["dropped"] == 5,
                 result["rules"] == 5]),
            f"{result['seed_has_severity']} of the {result['rules']} seed rules carry "
            "the field, so a parser requiring it has nothing to read and one defaulting "
            "to block makes every rule blocking. Defaulting to warn would downgrade "
            "no-release-script-edits, the rule most obviously meant to block",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
