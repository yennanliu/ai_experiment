"""Exercise 3 — read-only is the one thing the contract cannot say.

    Add a read-only research worker whose output is a fact table.

Reading of the exercise: a research worker is defined by what it may *not*
do, and the contract only describes what a unit may write. So the worker is
expressible only as "owns one output file", and the fact table is what makes
it worth the slot: rows that carry receipts, resolved against the code they
describe.

**ANSWER: the research unit owns one output path and returns a 5-row fact
table whose receipts all resolve.** Every row names a `path:line` in the
lesson's own module and the claimed symbol is found on that line -- **5** of
**5**. The plan validates `ready` with **0** conflicts, and the rows are the
kind of fact the lesson asks evidence to be: two of them change the plan
rather than describe it.

**FINDING: `WorkUnit` cannot express "read-only".** Its **5** fields describe
ownership of writes; there is no `reads` field and no flag, so a worker that
must not touch the tree is modelled as a worker that owns `outputs/facts.md`.
The planner will happily let that worker write anywhere -- nothing in it
executes or constrains a worker at all.

**FINDING: state isolation works, but only at the granularity you declare.**
`paths_overlap("outputs", "outputs/facts.md")` is **True**, so a worker that
claims `outputs/` blocks the researcher with **1** conflict, while two workers
declaring distinct files under it pass with **0**. The layer the docs call
state isolation is enforced exactly to the depth of the path strings.

**FINDING: the fact table only pays for itself if it runs first.** Making
`api` and `docs` depend on the researcher gives **3** waves instead of 2 and
**0** conflicts; leaving it unattached leaves it a second sink whose findings
arrive after the code they were meant to inform.

Structure: `research()` gathers the rows; `resolve()` checks each receipt
against the line it cites.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "45-delegate-with-isolation"
OUTPUT = "outputs/facts.md"

# (question answered, claim, receipt, the symbol that must sit on the cited line)
ROWS = [
    ("What does a unit promise?", "WorkUnit carries five fields and no goal",
     "code/main.py:14", "class WorkUnit"),
    ("How is overlap decided?", "Overlap is a PurePosixPath parent test, not a glob match",
     "code/main.py:25", "a in b.parents"),
    ("What happens to a bad dependency?", "waves raises instead of reporting",
     "code/main.py:43", 'raise ValueError("unknown dependency")'),
    ("What does the plan report?", "status is derived from three lists",
     "code/main.py:63", '"status"'),
    ("How big is the shipped split?", "The example ships three units",
     "code/main.py:72", "def example"),
]


def lesson_root(ref):
    return Path(ref.__file__).resolve().parents[1]


def resolve(root, receipt, symbol):
    path, _, number = receipt.rpartition(":")
    lines = (root / path).read_text(encoding="utf-8").splitlines()
    index = int(number) - 1
    return 0 <= index < len(lines) and symbol in lines[index]


def research(ref):
    """The worker's deliverable: one row per question, each with a receipt."""
    root = lesson_root(ref)
    return [{"question": question, "claim": claim, "receipt": receipt,
             "resolves": resolve(root, receipt, symbol)}
            for question, claim, receipt, symbol in ROWS]


def with_research(ref, wired):
    researcher = ref.WorkUnit("research", "reader", (OUTPUT,), (),
                              f"{OUTPUT} lists one receipt per row")
    units = [researcher]
    for unit in ref.example():
        depends = unit.depends_on or (("research",) if wired else ())
        units.append(ref.WorkUnit(unit.id, unit.owner, unit.paths, depends, unit.proof))
    return units


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = research(ref)
    loose = ref.delegation_plan(with_research(ref, wired=False))
    wired = ref.delegation_plan(with_research(ref, wired=True))
    greedy = with_research(ref, wired=False)
    greedy[1] = ref.WorkUnit("api", "worker-api", ("outputs",), (), "python3 -m unittest")
    return {
        "rows": len(rows), "resolved": sum(row["resolves"] for row in rows),
        "questions": len({row["question"] for row in rows}),
        "status": loose["status"], "conflicts": loose["conflicts"],
        "fields": list(ref.WorkUnit.__dataclass_fields__),
        "reads_field": any(name in ref.WorkUnit.__dataclass_fields__
                           for name in ("reads", "read_only", "mode")),
        "runs_anything": sum(word in inspect.getsource(ref)
                             for word in ("subprocess", "os.system")),
        "parent_overlap": ref.paths_overlap("outputs", OUTPUT),
        "sibling_overlap": ref.paths_overlap("outputs/facts.md", "outputs/plan.json"),
        "greedy_conflicts": len(ref.delegation_plan(greedy)["conflicts"]),
        "loose_waves": loose["waves"], "wired_waves": wired["waves"],
        "wired_conflicts": wired["conflicts"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the research unit owns one path and returns 5 rows that all resolve",
            all([result["rows"] == 5, result["resolved"] == 5, result["questions"] == 5,
                 result["status"] == "ready", result["conflicts"] == []]),
            f"the fact table holds {result['rows']} rows answering "
            f"{result['questions']} questions, {result['resolved']} of whose receipts "
            f"resolve to the line they cite, and the plan reads {result['status']!r} with "
            f"{len(result['conflicts'])} conflicts",
        ),
        practice.Check(
            "FINDING: WorkUnit cannot express read-only",
            all([len(result["fields"]) == 5, result["reads_field"] is False,
                 result["runs_anything"] == 0]),
            f"the {len(result['fields'])} fields {result['fields']} describe writes; there "
            "is no reads field and no flag, and the module never runs a worker, so "
            "'read-only' is a promise the artifact cannot carry",
        ),
        practice.Check(
            "FINDING: state isolation works to the depth of the path strings",
            all([result["parent_overlap"] is True, result["sibling_overlap"] is False,
                 result["greedy_conflicts"] == 1]),
            f"a worker claiming outputs/ collides with the researcher "
            f"({result['greedy_conflicts']} conflict) while two distinct files under it do "
            f"not ({result['sibling_overlap']}); the layer the docs call state isolation is "
            "enforced exactly as deep as the strings go",
        ),
        practice.Check(
            "FINDING: the fact table only pays for itself if it runs first",
            all([len(result["loose_waves"]) == 2, len(result["wired_waves"]) == 3,
                 result["wired_waves"][0] == ["research"],
                 result["wired_conflicts"] == []]),
            f"unattached the researcher runs in {result['loose_waves']}; wired as a "
            f"dependency it gives {result['wired_waves']} with "
            f"{len(result['wired_conflicts'])} conflicts, so the findings arrive before the "
            "code they were meant to inform",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
