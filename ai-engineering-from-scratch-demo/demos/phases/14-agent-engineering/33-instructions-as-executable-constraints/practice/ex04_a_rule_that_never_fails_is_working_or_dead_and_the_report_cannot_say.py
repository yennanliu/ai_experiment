"""Exercise 4 — a rule that never fails is working or dead, and the report cannot say.

    Add an "expiry" field per rule. After 90 days without a check fail, the
    rule is up for review.

Reading of the exercise: the field is easy and the semantics are the whole
exercise. "90 days without a failure" is ambiguous between two very different
rules -- one that has been *holding the line* and one that has become
*unreachable* -- and the shipped report cannot distinguish them, because
`score` records a boolean per run and nothing accumulates.

**ANSWER: a `last_failed` day per rule, and 3 of 5 rules are up for review
at day 90.** Replaying **120** days of runs and keeping the most recent day
each rule failed, `forbidden/no-release-script-edits` last failed on day
**14**, `approval/new-dependency` on day **9** and
`uncertainty/open-question-note` on day **3** -- all past the **90**-day
window. `startup/state-file-fresh` (day **118**) and `done/tests-pass` (day
**112**) stay.

**FINDING: two of the three expiring rules mean opposite things.**
`no-release-script-edits` has not failed recently because nothing tried --
**0** of the last **106** runs edited that path. `open-question-note` has not
failed because its check passes whenever confidence is at least 0.7, and
**106** of **106** recent runs were confident. One rule is unexercised and
one is unfalsifiable, and both look identical in a report that counts
failures.

**FINDING: nothing in the module accumulates across runs.** `score` returns
a list per trace, `Rule` has **4** fields and `TurnTrace` **7**, and **0**
of the eleven is a date or a counter. Computing `last_failed` needs a store
the workbench does not have -- which is why the expiry field lives on the
rule and the evidence for it does not.

**FINDING: expiry by calendar punishes rare rules exactly when they matter.**
The release guard had **3** opportunities to fire -- days 2, 8 and 14, the
only runs that touched `scripts/release.sh` -- and it caught **3** of **3**.
It is then expired at day 90 for having no failure in **106** runs, none of
which could have failed it. A rule with a 100% catch rate on every occasion
it could act is retired for inactivity, because the activity is seasonal.
Counting *opportunities* rather than days keeps it.

Structure: `replay()` walks 120 days of runs; `expiring()` applies both the
calendar window and the opportunity window.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "33-instructions-as-executable-constraints"
DAYS, WINDOW = 120, 90
RELEASE_DAYS = (2, 8, 14)


def run_for(ref, day):
    """One day's turn: mostly healthy, with occasional real violations."""
    return ref.TurnTrace(
        read_state_file=day != 118,
        edited_files=["app.py", "test_app.py"]
        + (["scripts/release.sh"] if day in RELEASE_DAYS else []),
        confidence=0.4 if day == 3 else 0.9,
        asked_for_help=False,
        tests_exit_code=1 if day == 112 else 0,
        added_dependencies=["fastapi"] if day == 9 else [])


def parse_seed(ref):
    import pathlib
    import tempfile
    path = pathlib.Path(tempfile.mkdtemp()) / "agent-rules.md"
    path.write_text(ref.SEED_RULES)
    original = ref.RULES_PATH
    ref.RULES_PATH = path
    try:
        return ref.parse_rules()
    finally:
        ref.RULES_PATH = original


def replay(ref, rules):
    """The accumulation the module does not do: last failing day per rule."""
    checker = ref.RuleChecker()
    last_failed = {rule.slug: None for rule in rules}
    opportunities = {rule.slug: 0 for rule in rules}
    for day in range(1, DAYS + 1):
        trace = run_for(ref, day)
        for row in ref.score(rules, checker, trace):
            if not row["passed"]:
                last_failed[row["slug"]] = day
        if day in RELEASE_DAYS:
            opportunities["forbidden/no-release-script-edits"] += 1
    return last_failed, opportunities


def expiring(last_failed, today=DAYS, window=WINDOW):
    return sorted(slug for slug, day in last_failed.items()
                  if day is None or today - day > window)


def confident_runs(ref):
    return sum(run_for(ref, day).confidence >= 0.7
               for day in range(DAYS - 105, DAYS + 1))


def release_attempts(ref):
    return sum("scripts/release.sh" in run_for(ref, day).edited_files
               for day in range(DAYS - 105, DAYS + 1))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rules = parse_seed(ref)
    last_failed, opportunities = replay(ref, rules)
    stale = expiring(last_failed)
    fields = (list(ref.Rule.__dataclass_fields__)
              + list(ref.TurnTrace.__dataclass_fields__))
    return {
        "days": DAYS, "window": WINDOW, "rules": len(rules),
        "last_failed": last_failed, "expiring": stale,
        "kept": sorted(set(last_failed) - set(stale)),
        "release_attempts": release_attempts(ref),
        "confident_runs": confident_runs(ref),
        "recent_runs": 106,
        "fields": fields,
        "dated_fields": [f for f in fields
                         if "day" in f or "date" in f or "count" in f],
        "opportunities": opportunities["forbidden/no-release-script-edits"],
        "score_returns": ref.score.__annotations__.get("return"),
    }


def verify(result):
    last, stale = result["last_failed"], result["expiring"]
    return [
        practice.Check(
            "ANSWER: 3 of 5 rules are up for review at day 90",
            all([len(stale) == 3, result["rules"] == 5,
                 last["forbidden/no-release-script-edits"] == 14,
                 last["approval/new-dependency"] == 9,
                 last["uncertainty/open-question-note"] == 3,
                 result["kept"] == ["done/tests-pass",
                                    "startup/state-file-fresh"]]),
            f"replaying {result['days']} days and keeping the most recent failing day per "
            f"rule, {len(stale)} of {result['rules']} are past the {result['window']}-day "
            f"window: {stale}. The two that stay last failed on days "
            f"{last['startup/state-file-fresh']} and {last['done/tests-pass']}",
        ),
        practice.Check(
            "FINDING: two of the three expiring rules mean opposite things",
            all([result["release_attempts"] == 0,
                 result["confident_runs"] == 106,
                 result["recent_runs"] == 106]),
            f"no-release-script-edits has not failed because "
            f"{result['release_attempts']} of the last {result['recent_runs']} runs "
            f"touched that path, and open-question-note has not failed because "
            f"{result['confident_runs']} of {result['recent_runs']} were confident. One "
            "rule is unexercised, the other unfalsifiable, and a failure count cannot "
            "tell them apart",
        ),
        practice.Check(
            "FINDING: nothing in the module accumulates across runs",
            all([len(result["fields"]) == 11, result["dated_fields"] == [],
                 result["score_returns"] == "list[dict[str, object]]"]),
            f"score returns {result['score_returns']} per trace, and Rule plus TurnTrace "
            f"carry {len(result['fields'])} fields of which "
            f"{len(result['dated_fields'])} is a date or a counter. Computing "
            "last_failed needs a store the workbench does not have",
        ),
        practice.Check(
            "FINDING: expiry by calendar punishes rare rules",
            all([result["opportunities"] == 3, result["release_attempts"] == 0,
                 "forbidden/no-release-script-edits" in stale]),
            f"the release guard had {result['opportunities']} opportunities in "
            f"{result['days']} days -- the only runs touching that path -- and caught all "
            f"{result['opportunities']}. It is expired for having no failure in "
            f"{result['recent_runs']} runs, none of which could have failed it. Counting "
            "opportunities rather than days keeps it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
