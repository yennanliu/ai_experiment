"""Exercise 1 — the turn saving and the blast radius are the same number.

    Run `code/main.py`. How many turns does each scaffold take on the same
    task set? What is the per-action blast radius of each?

Reading of the exercise: two numbers per scaffold invites a table, and a
table of four numbers at one problem size says nothing about which way either
number moves. Both scaffolds accept a starting repo, so the same comparison is
run at **0**, **1**, **2** and **3** remaining bugs and the two columns are
read as functions of that.

**ANSWER: JSON takes 4 turns at blast radius 1; CodeAct takes 2 at blast
radius 3.** Both pass **3** of 3. Across the sweep, JSON turns are exactly
`bugs + 1` -- **1, 2, 3, 4** -- while CodeAct is **2** for any non-zero
number of bugs, and CodeAct's observed blast radius is exactly `bugs` --
**0, 1, 2, 3**.

**FINDING: the trade is one-for-one.** Every turn CodeAct saves is a file
added to the radius of a single action. At 3 bugs it saves **2** turns and
carries **3** files; the two columns are the same quantity read from
different ends, which is a sharper statement of "neither is strictly better"
than the prose gets to.

**FINDING: the two radii are not measured the same way.** `JsonScaffold`'s
is the literal `return 1`; `CodeActScaffold`'s is `worst_touched`, an observed
maximum. At zero bugs CodeAct reports **0** and JSON still reports **1** -- so
the scaffold that touched nothing reports the larger radius, because one
number is an observation and the other is a claim.

**FINDING: the action string is a description, not a cause.** Both `step`
methods call `_apply_fix` and mutate the repo *before* returning, and the only
thing either `run` reads back is whether the action was `done` -- **1** field
in the JSON case and a string comparison in the other. Nothing decodes or
executes the action, so what the comparison measures is how many fixes each
`step` chooses to apply per call.

Structure: `sweep()` runs both scaffolds from a repo with a chosen number of
bugs left in it; everything else is read off that.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "09-coding-agent-landscape"


def repaired(ref):
    """Each file with its fix already applied."""
    return {path: ref.INITIAL_REPO[path].replace(*rule)
            for path, rule in ref.FIXES.items()}


def starting_repo(ref, bugs):
    """The shipped repo with all but `bugs` of its files repaired."""
    fixed, repo = repaired(ref), dict(ref.INITIAL_REPO)
    paths = list(ref.INITIAL_REPO)
    for path in paths[:len(paths) - bugs]:
        repo[path] = fixed[path]
    return repo


def sweep(ref, bugs):
    """(passed, turns, blast) for each scaffold, starting from `bugs` bugs."""
    rows = []
    for scaffold in (ref.JsonScaffold, ref.CodeActScaffold):
        run = scaffold(repo=starting_repo(ref, bugs))
        passed, turns = run.run()
        rows.append((passed, turns, run.blast_radius()))
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = [sweep(ref, bugs) for bugs in range(4)]
    shipped = rows[3]
    json_step, act_step = (inspect.getsource(ref.JsonScaffold.step),
                           inspect.getsource(ref.CodeActScaffold.step))
    return {
        "json": list(shipped[0]),
        "codeact": list(shipped[1]),
        "tests": len(ref.TESTS),
        "json_turns": [row[0][1] for row in rows],
        "codeact_turns": [row[1][1] for row in rows],
        "json_blast": [row[0][2] for row in rows],
        "codeact_blast": [row[1][2] for row in rows],
        "turns_saved": shipped[0][1] - shipped[1][1],
        "radius_added": shipped[1][2] - shipped[0][2],
        "json_radius_literal": "return 1" in inspect.getsource(ref.JsonScaffold.blast_radius),
        "codeact_radius_observed": "worst_touched" in inspect.getsource(
            ref.CodeActScaffold.blast_radius),
        "applies_before_returning": [json_step.index("_apply_fix") < json_step.index("return json"),
                                     act_step.index("_apply_fix") < act_step.index("return \"; \"")],
        "fields_read": inspect.getsource(ref.JsonScaffold.run).count('get("tool")'),
        "returns_in_loop": json_step.count("return json.dumps"),
        "appends_in_loop": act_step.count("snippet_lines.append"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 4 turns at radius 1 against 2 turns at radius 3",
            all([result["json"] == [3, 4, 1], result["codeact"] == [3, 2, 3],
                 result["json_turns"] == [1, 2, 3, 4],
                 result["codeact_turns"] == [1, 2, 2, 2],
                 result["codeact_blast"] == [0, 1, 2, 3]]),
            f"both pass {result['tests']} of {result['tests']}; JSON turns run "
            f"{result['json_turns']} as bugs go 0 to 3 -- bugs + 1 -- while CodeAct runs "
            f"{result['codeact_turns']} and its radius runs {result['codeact_blast']}",
        ),
        practice.Check(
            "FINDING: the trade is one-for-one",
            all([result["turns_saved"] == 2, result["radius_added"] == 2,
                 result["turns_saved"] == result["radius_added"]]),
            f"at three bugs CodeAct saves {result['turns_saved']} turns and adds "
            f"{result['radius_added']} files to a single action -- the two columns are "
            "the same quantity read from different ends",
        ),
        practice.Check(
            "FINDING: the two radii are not measured the same way",
            all([result["json_radius_literal"], result["codeact_radius_observed"],
                 result["json_blast"] == [1, 1, 1, 1],
                 result["codeact_blast"][0] == 0]),
            f"JSON's radius is the literal 1 at every size {result['json_blast']} while "
            f"CodeAct's is an observed maximum -- so at zero bugs the scaffold that "
            f"touched nothing reports {result['codeact_blast'][0]} and the other still "
            "reports 1",
        ),
        practice.Check(
            "FINDING: the action string is a description, not a cause",
            all([result["applies_before_returning"] == [True, True],
                 result["fields_read"] == 1, result["returns_in_loop"] == 2,
                 result["appends_in_loop"] == 1]),
            f"both step methods apply the fix before returning, and the only thing read "
            f"back is whether the action was done -- {result['fields_read']} field. The "
            "difference between the scaffolds is that one returns inside the loop and "
            "the other appends",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
