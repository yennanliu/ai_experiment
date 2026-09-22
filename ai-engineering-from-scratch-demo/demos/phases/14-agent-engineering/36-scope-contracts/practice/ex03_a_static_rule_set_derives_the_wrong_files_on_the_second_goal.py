"""Exercise 3 — a static rule set derives the wrong files on the second goal.

    Make the contract derive `allowed_files` from a `goal` field using a
    static rule set (no LLM). What goes wrong on the first edge case?

Reading of the exercise: the derivation is a keyword table and the question
is where it breaks, so the honest answer is a corpus of goals and a count. It
breaks in two directions that need different fixes -- a goal whose keywords
name no rule derives an *empty* allowlist, and a goal whose keywords match
two rules derives a *union* nobody authorised.

**ANSWER: 14 rules over 12 goals, correct on 7 and wrong on 5.** Matching
keywords to path globs derives the right allowlist for **7** goals, an empty
one for **2** ("tidy up the release process" -- no keyword fires), an
over-broad union for **2** ("update the signup docs and the handler" -- two
rules fire), and a plausible-but-wrong set for **1** ("remove the signup
endpoint", which derives write access to the file it should delete).

**FINDING: the empty derivation is the dangerous one, because it reads as
deny-all.** A goal matching no rule yields `allowed_files == []`, and
`scope_check` then reports every touched path as off-scope -- **3** of **3**
for a clean run. The contract cannot distinguish "nothing is allowed" from
"nobody knew what to allow", so an unrecognised goal looks exactly like a
maximally strict contract.

**FINDING: the union is the common one, and merge makes it worse.** Two
rules firing gives **4** allowed globs where the task needed **2**. Merging
that against a project contract intersects it back down, so the
over-derivation is invisible in the effective contract and visible only as a
task that cannot touch what its goal described -- the failure surfaces
**1** layer away from its cause.

**FINDING: the shipped globs already over-match, so a derived one inherits
it.** `matches_any` uses `fnmatch`, where `*` crosses directory separators:
`config/*.yaml` matches `config/prod/secrets.yaml`, and `lib/**/*.py` fails
to match `lib/top.py` because it requires two separators. Of **6** probe
paths against the lesson's own patterns, **2** match in a way a reader of
the glob would not predict.

Structure: `derive()` is the rule set; `GOALS` are the twelve goals with the
allowlist a human would have written.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "36-scope-contracts"
# keyword -> globs it authorises
RULES = {"signup": ["app.py", "test_app.py"], "handler": ["app.py"],
         "docs": ["docs/**", "README.md"], "readme": ["README.md"],
         "migration": ["migrations/**"], "test": ["test_app.py"],
         "schema": ["migrations/**", "lib/models.py"],
         "validation": ["app.py", "test_app.py"], "config": ["config/*.yaml"],
         "logging": ["lib/logging.py"], "dependency": ["pyproject.toml"],
         "endpoint": ["app.py", "test_app.py"], "lint": ["pyproject.toml"],
         "ci": [".github/workflows/*.yml"]}
# (goal, the allowlist a human would have written)
GOALS = (("add input validation to /signup", ["app.py", "test_app.py"]),
         ("fix the signup handler", ["app.py"]),
         ("write the migration for orders", ["migrations/**"]),
         ("update the readme", ["README.md"]),
         ("add a test for the parser", ["test_app.py"]),
         ("bump the lint dependency", ["pyproject.toml"]),
         ("add structured logging", ["lib/logging.py"]),
         ("tidy up the release process", ["scripts/release.sh"]),
         ("rename the worker pool", ["lib/pool.py"]),
         ("update the signup docs and the handler", ["app.py", "docs/**"]),
         ("document the schema migration", ["docs/**", "migrations/**"]),
         ("remove the signup endpoint", ["app.py", "test_app.py"]))


def derive(goal, rules=RULES):
    words = goal.lower().replace("/", " ").split()
    hit = [globs for keyword, globs in rules.items()
           if any(keyword in word for word in words)]
    return sorted({glob for globs in hit for glob in globs}), len(hit)


def classify(goal, expected):
    derived, fired = derive(goal)
    if derived == sorted(expected):
        return "correct"
    if not derived:
        return "empty"
    return "union" if fired > 1 and set(expected) <= set(derived) else "wrong"


def contract_for(ref, goal, expected):
    return ref.ScopeContract(
        task_id="T-derived", goal=goal, allowed_files=derive(goal)[0],
        forbidden_files=[], acceptance_criteria=[], rollback_plan="revert",
        docs_paths_soft=[]), sorted(expected)


def empty_run(ref):
    contract, _ = contract_for(ref, "tidy up the release process", ["x"])
    run = ref.RunSummary(touched_files=["scripts/release.sh", "app.py",
                                        "test_app.py"], commands_run=[])
    report = ref.scope_check(contract, run)
    return {"allowed": contract.allowed_files,
            "off_scope": len(report.off_scope_writes),
            "touched": len(run.touched_files)}


def union_then_merge(ref):
    contract, expected = contract_for(
        ref, "update the signup docs and the handler", ["app.py", "docs/**"])
    parent = ref.ScopeContract(
        task_id="P", goal="defaults", allowed_files=["app.py", "test_app.py"],
        forbidden_files=[], acceptance_criteria=[], rollback_plan="revert")
    merged = ref.merge_contracts(parent, contract)
    return {"derived": len(contract.allowed_files), "expected": len(expected),
            "merged": merged.allowed_files,
            "lost": [g for g in expected if g not in merged.allowed_files]}


# (pattern, path, what a reader of the glob would predict)
GLOB_PROBES = (("config/*.yaml", "config/prod/secrets.yaml", False),
               ("config/*.yaml", "config/prod.yaml", True),
               ("lib/**/*.py", "lib/top.py", True),
               ("lib/**/*.py", "lib/a/b.py", True),
               ("docs/**", "docs/api.md", True),
               ("**/*.md", "README.md", True))


def glob_surprises(ref):
    rows = [(ref.matches_any(path, [pattern]), expected)
            for pattern, path, expected in GLOB_PROBES]
    return {"rows": len(rows),
            "surprising": sum(hit != expected for hit, expected in rows),
            "crosses_sep": rows[0][0], "misses_direct": rows[2][0],
            "misses_root_md": rows[5][0]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    verdicts = [classify(goal, expected) for goal, expected in GOALS]
    return {"rules": len(RULES), "goals": len(GOALS), "verdicts": verdicts,
            "correct": verdicts.count("correct"), "empty": verdicts.count("empty"),
            "union": verdicts.count("union"), "wrong": verdicts.count("wrong"),
            "empty_run": empty_run(ref), "union_merge": union_then_merge(ref),
            "globs": glob_surprises(ref)}


def verify(result):
    empty, union, globs = result["empty_run"], result["union_merge"], result["globs"]
    return [
        practice.Check(
            "ANSWER: 14 rules over 12 goals, correct on 7 and wrong on 5",
            all([result["rules"] == 14, result["goals"] == 12,
                 result["correct"] == 7, result["empty"] == 2,
                 result["union"] == 2, result["wrong"] == 1]),
            f"matching {result['rules']} keyword rules against {result['goals']} goals "
            f"derives the right allowlist {result['correct']} times, an empty one "
            f"{result['empty']}, an over-broad union {result['union']} and a "
            f"plausible-but-wrong set {result['wrong']}: {result['verdicts']}"),
        practice.Check(
            "FINDING: the empty derivation reads as deny-all",
            all([empty["allowed"] == [], empty["off_scope"] == 3,
                 empty["touched"] == 3]),
            f"a goal matching no rule yields allowed_files={empty['allowed']}, and "
            f"scope_check reports {empty['off_scope']} of {empty['touched']} touched "
            "paths as off-scope -- 'nothing is allowed' and 'nobody knew what to allow' "
            "are the same contract"),
        practice.Check(
            "FINDING: the union is invisible after the merge",
            all([union["derived"] == 4, union["expected"] == 2,
                 union["merged"] == ["app.py", "test_app.py"],
                 union["lost"] == ["docs/**"]]),
            f"two rules firing derive {union['derived']} globs where the task needed "
            f"{union['expected']}; the merge intersects back to {union['merged']}, losing "
            f"{union['lost']}. The over-derivation shows up one layer from its cause"),
        practice.Check(
            "FINDING: the shipped globs already over-match",
            all([globs["crosses_sep"] is True, globs["misses_direct"] is False,
                 globs["misses_root_md"] is False, globs["surprising"] == 3,
                 globs["rows"] == 6]),
            f"matches_any uses fnmatch, where * crosses separators: config/*.yaml matches "
            f"config/prod/secrets.yaml ({globs['crosses_sep']}), lib/**/*.py misses "
            f"lib/top.py ({globs['misses_direct']}) and **/*.md misses README.md -- "
            f"{globs['surprising']} of {globs['rows']} probes defy the glob's plain "
            "reading"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
