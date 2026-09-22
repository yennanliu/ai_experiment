"""Exercise 5 — what survives the cut is the two outcomes with units a reader owns.

    Author a one-page summary aimed at a non-engineer. What survives the cut?

Reading of the exercise: a one-page summary is a length budget and an audience
constraint at once. The budget is roughly 350 words; the constraint is that
every number has to carry a unit the reader already owns. That rules most of
the report out before any writing starts.

**ANSWER: 2 of the 5 outcomes survive, and the summary lands in 264 words.**
`tests_actually_run` is a yes/no about whether the thing was checked, and
`files_outside_scope` is a count of files that changed and should not have.
Both are facts a reader can hold an opinion about. The other three --
`acceptance_met`, `handoff_quality`, `reviewer_total` -- need the workbench
explained before they mean anything, so they go in the body as consequences,
not in the table as numbers.

**FINDING: all 5 outcome names are identifiers, not English.** Every one is
snake_case, **3** of them name a workbench surface the reader has never heard
of, and the report prints them verbatim as row labels. Renaming is not
cosmetic: `reviewer_total (/10)` is a scale with no external referent, and a
reader who cannot calibrate a 9 will read it as a grade.

**FINDING: the report's one explanatory paragraph is 50 words and names 3
surfaces.** "runs the acceptance command through the feedback runner, passes
the verification gate, and ships a handoff packet the next session loads" is
the sentence a non-engineer stops reading at. The summary makes the same claim
with **0** surface names.

**FINDING: the number worth printing is the one from a real repository.**
The benchmark's rows are literals, so its 9/10 and its zero off-scope files
are claims. Exercise 2's numbers -- **1** off-scope file and a missing handoff
on a commit that actually shipped -- are checkable, and a summary that cites
the checkable one is the summary that survives a skeptic.

Structure: `SUMMARY` is the deliverable; `readability()` measures it against
the budget and the jargon list.
"""

from __future__ import annotations

import inspect
import re

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "41-workbench-for-real-repos"
BUDGET = 350
SURFACES = ["init script", "scope contract", "state file", "feedback runner",
            "verification gate", "reviewer", "handoff packet"]

SUMMARY = """\
# Does the checklist pay for itself?

We ran the same job twice: once by asking the assistant to do it, once by asking
it to follow a short checklist. Same assistant, same job, different result.

**Two things worth your attention.**

First: did anyone check the work? Without the checklist, the assistant said the
job was done and nothing was run to confirm it. With the checklist, the project's
own tests ran before anyone claimed anything. That is the difference between a
report and a receipt.

Second: how many files changed that should not have? Without the checklist,
three files were edited and two of them were off-limits -- including the script
that publishes releases. With it, the edits stayed where the job said they would.

**What it costs.** The checklist adds about four steps to a job. On a one-line
change that is real overhead and it is fair to skip it. On anything larger the
overhead falls away, because the extra steps are fixed and the work is not.

**What we are not claiming.** The comparison we ship is a worked example, not a
measurement -- its numbers are written in, not observed. The numbers we do stand
behind come from running the checklist over a change that already shipped in this
repository: one file changed outside the agreed list, and no handoff left behind
for whoever picked it up next. Both were invisible before we looked.

**What we would like.** Keep the two cheap steps -- run the tests, check the file
list -- on every job, and treat the rest as optional for small ones.
"""


def readability(text):
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    return {"words": len(words), "lines": len(text.strip().splitlines()),
            "surfaces": sum(name in text.lower() for name in SURFACES),
            "identifiers": len(re.findall(r"\b[a-z]+_[a-z_]+\b", text))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    outcomes = [f for f in ref.TaskOutcome.__dataclass_fields__ if f != "pipeline"]
    report = inspect.getsource(ref.write_report)
    paragraph = report.split('"## Read",')[1]
    prose = " ".join(re.findall(r'"([^"]{20,})"', paragraph))
    mine = readability(SUMMARY)
    return {
        "outcomes": outcomes,
        "survivors": [name for name in outcomes if name in ("tests_actually_run",
                                                            "files_outside_scope")],
        "snake_case": sum("_" in name for name in outcomes),
        "surface_named": sum(any(word in name for word in ("handoff", "reviewer", "acceptance"))
                             for name in outcomes),
        "prose_words": len(re.findall(r"[A-Za-z][A-Za-z'-]*", prose)),
        "prose_surfaces": sum(name in prose.lower() for name in SURFACES),
        **mine, "budget": BUDGET,
        "claimed": (ref.run_workbench().reviewer_total,
                    len(ref.run_workbench().files_outside_scope)),
        "measured": (1, "missing"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 2 of the 5 outcomes survive, in a 264-word page",
            all([len(result["outcomes"]) == 5, len(result["survivors"]) == 2,
                 result["words"] == 264, result["words"] < result["budget"],
                 result["identifiers"] == 0]),
            f"of {len(result['outcomes'])} outcomes the summary keeps "
            f"{result['survivors']} -- a yes/no and a count -- in {result['words']} words "
            f"against a {result['budget']}-word page, with "
            f"{result['identifiers']} snake_case identifiers left in the text",
        ),
        practice.Check(
            "FINDING: all 5 outcome names are identifiers, not English",
            all([result["snake_case"] == 5, result["surface_named"] == 3]),
            f"{result['snake_case']} of {len(result['outcomes'])} outcome names are "
            f"snake_case and {result['surface_named']} name a workbench surface the reader "
            "has never heard of. reviewer_total (/10) is a scale with no external referent",
        ),
        practice.Check(
            "FINDING: the report's explanatory paragraph is 50 words and names 3 surfaces",
            all([result["prose_words"] == 50, result["prose_surfaces"] == 3,
                 result["surfaces"] == 0]),
            f"the report's one paragraph runs {result['prose_words']} words and names "
            f"{result['prose_surfaces']} surfaces; the summary makes the same claim naming "
            f"{result['surfaces']}",
        ),
        practice.Check(
            "FINDING: the number worth printing is the one from a real repository",
            all([result["claimed"] == (9, 0), result["measured"] == (1, "missing")]),
            f"the benchmark's literals say reviewer_total {result['claimed'][0]} and "
            f"{result['claimed'][1]} off-scope files; Exercise 2's checkable numbers on a "
            f"commit that shipped say {result['measured'][0]} off-scope file and a "
            f"{result['measured'][1]} handoff. The summary cites the second",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
