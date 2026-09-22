"""Exercise 3 — a human unknown leaves the frame reading READY.

    Add a human unknown whose answer would change the public contract.

Reading of the exercise: the lesson names four classes of unknown --
discoverable, decidable, human, deferred -- and says the agent should pause at
human ones "before the choice is buried in code". Adding one is easy; the
question worth answering is whether anything notices.

**ANSWER: the human unknown is which status a duplicate returns to an existing
caller, and adding it changes nothing the tool can see.** A frame carrying
"Does rejecting a duplicate return 409 or 422 to callers already handling
409?" validates with **0** issues and renders `Status: READY`, identical to the
frame without it. The answer changes a public wire contract; the frame's
status does not move.

**FINDING: `validate` has 6 branches and 0 of them mention unknowns.**
`TaskFrame` has **6** fields, `render` prints an `## Unknowns` section, and the
validator never reads the list. A frame with **0** unknowns and one with **4**
are equally READY, which means "pause at human unknowns" is advice the program
cannot hold anyone to.

**FINDING: classification is the missing field, not the missing list.** The
lesson's own example carries "Whether email comparison is case-insensitive" --
a *discoverable* unknown, answerable by reading the store -- in the same list
where a human one would go. Tagging each entry with one of the **4** classes
lets the validator block on **1** of them and pass the rest, which is the
whole of the rule.

**FINDING: the blocking version is one line and flips the status.** Refusing
when any unknown is tagged `human` turns the frame from **0** issues to **1**
and the status from READY to BLOCKED, while the discoverable unknown still
passes. The cost of enforcement here is smaller than the paragraph explaining
why it matters.

Structure: `tagged()` classifies the unknowns; `strict()` is the one-line
validator extension the exercise implies.
"""

from __future__ import annotations

import inspect
import re

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "43-frame-the-task-before-code"
HUMAN = ("Does rejecting a duplicate return 409 or 422 to callers already handling 409?",
         "human")
TAGGED = [
    ("Whether email comparison is case-insensitive", "discoverable"),
    ("Which error envelope key carries the field name", "decidable"),
    HUMAN,
    ("Whether existing duplicate rows are backfilled", "deferred"),
]


def frame(ref, unknowns):
    example = ref.example()
    return ref.TaskFrame(goal=example.goal, allowed_paths=example.allowed_paths,
                         forbidden_paths=example.forbidden_paths,
                         acceptance=example.acceptance, facts=example.facts,
                         unknowns=[text for text, _ in unknowns])


def status(ref, frame_obj, issues=None):
    rendered = ref.render(frame_obj)
    line = [row for row in rendered.splitlines() if row.startswith("Status")][0]
    return line if issues is None else ("Status: BLOCKED" if issues else line)


def strict(ref, frame_obj, unknowns):
    """validate, plus the one rule the lesson states and the code omits."""
    issues = list(ref.validate(frame_obj))
    issues += [f"human unknown blocks design: {text}"
               for text, kind in unknowns if kind == "human"]
    return issues


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    without = frame(ref, [row for row in TAGGED if row[1] != "human"])
    with_human = frame(ref, TAGGED)
    validator = inspect.getsource(ref.validate)
    discoverable = [row for row in TAGGED if row[1] == "discoverable"]
    return {
        "issues_without": ref.validate(without), "issues_with": ref.validate(with_human),
        "status_without": status(ref, without), "status_with": status(ref, with_human),
        "unknowns_without": len(without.unknowns), "unknowns_with": len(with_human.unknowns),
        "frame_fields": len(ref.TaskFrame.__dataclass_fields__),
        "branches": len(re.findall(r"issues\.append", validator)),
        "mentions_unknowns": "unknown" in validator.lower(),
        "renders_section": "## Unknowns" in ref.render(with_human),
        "classes": sorted({kind for _, kind in TAGGED}),
        "example_unknowns": ref.example().unknowns,
        "strict_issues": strict(ref, with_human, TAGGED),
        "strict_status": status(ref, with_human, strict(ref, with_human, TAGGED)),
        "strict_discoverable": strict(ref, frame(ref, discoverable), discoverable),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: adding the human unknown changes nothing the tool can see",
            all([result["issues_with"] == [], result["issues_without"] == [],
                 result["status_with"] == "Status: READY",
                 result["status_with"] == result["status_without"],
                 result["unknowns_with"] == result["unknowns_without"] + 1]),
            f"the frame goes from {result['unknowns_without']} unknowns to "
            f"{result['unknowns_with']} and validate returns {result['issues_with']} either "
            f"way, rendering {result['status_with']!r} -- the answer changes a public wire "
            "contract and the status does not move",
        ),
        practice.Check(
            "FINDING: validate has 6 branches and none mention unknowns",
            all([result["branches"] == 6, result["mentions_unknowns"] is False,
                 result["frame_fields"] == 6, result["renders_section"] is True]),
            f"TaskFrame has {result['frame_fields']} fields and render prints an "
            f"## Unknowns section, but {result['branches']} validator branches read the "
            "list zero times, so 'pause at human unknowns' is advice the program cannot "
            "enforce",
        ),
        practice.Check(
            "FINDING: classification is the missing field, not the missing list",
            all([len(result["classes"]) == 4,
                 result["example_unknowns"] == ["Whether email comparison is case-insensitive"]]),
            f"the four classes are {result['classes']} and the lesson's own example carries "
            f"{result['example_unknowns']} -- a discoverable unknown sitting in the same "
            "untyped list a human one would go into",
        ),
        practice.Check(
            "FINDING: the blocking version is one line and flips the status",
            all([len(result["strict_issues"]) == 1,
                 result["strict_status"] == "Status: BLOCKED",
                 result["strict_discoverable"] == []]),
            f"refusing on any unknown tagged human yields {len(result['strict_issues'])} "
            f"issue and {result['strict_status']!r}, while a frame carrying only the "
            f"discoverable unknown still returns {result['strict_discoverable']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
