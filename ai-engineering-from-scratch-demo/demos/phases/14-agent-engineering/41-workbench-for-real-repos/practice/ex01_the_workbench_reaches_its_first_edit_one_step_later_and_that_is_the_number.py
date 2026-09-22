"""Exercise 1 — the workbench reaches its first edit one step later, and that is the number.

    Add a sixth outcome: time-to-first-meaningful-edit. How do you measure it
    cleanly?

Reading of the exercise: "cleanly" rules out the wall clock. Both pipelines
here are scripted, so a timer measures the machine; on a real run it measures
the model's sampling speed and the network. The clean unit is the pipeline's
own step list, which the lesson publishes, and the clean definition of
"meaningful" is an edit to a file inside the scope contract.

**ANSWER: prompt-only reaches its first edit at step 3 of 4, the workbench at
step 4 of 8.** Counting the doc's own numbered pipelines: one extra step of
setup before the first keystroke, **75.0%** of the prompt-only pipeline spent
before the edit against **50.0%** of the workbench's. The number that matters
is not the delay -- it is what follows: prompt-only has **1** step after the
edit, the workbench has **4**, and all four of them are checks.

**FINDING: the measurement has to exclude edits outside the contract.**
Prompt-only's touched list is `app.py`, `README.md` and
`sample_app/scripts/release.sh`, and **2** of those **3** are outside
`ALLOWED`. An unqualified "first edit" would score a pipeline faster for
writing to the release script first, so "meaningful" must mean *in scope*, or
the metric rewards the failure mode the whole workbench exists to stop.

**FINDING: there is no clock in the module to measure with.** **0** of the
**5** functions import `time`, and both pipelines return hardcoded
`TaskOutcome` literals, so a duration field would record **0.0** for both. A
sixth outcome has to be derivable from the artifacts, not from the run.

**FINDING: `FORBIDDEN` is declared and never read.** The set names
`sample_app/scripts/release.sh`, and `files_outside_scope` is computed as
`p not in ALLOWED`, so the forbidden write lands in the same bucket as the
README. **1** of the **2** off-scope writes is a block-severity event in
Lesson 38's vocabulary and the benchmark counts it as one of two.

Structure: `pipelines()` parses the two step lists out of the lesson's own
docs; `first_edit()` finds the first step that writes inside the contract.
"""

from __future__ import annotations

import inspect
import re

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "41-workbench-for-real-repos"
EDIT = re.compile(r"\bEdit\b")


def pipelines():
    """The two numbered step lists, read from the lesson's own documentation."""
    doc = parity.doc_text(PHASE, LESSON)
    out = {}
    for name in ("Prompt-only:", "Workbench-guided:"):
        block = doc[doc.index(name):]
        steps = re.findall(r"^\d+\. (.+)$", block, re.M)
        out[name.rstrip(":")] = steps[:4] if name.startswith("Prompt") else steps[:8]
    return out


def first_edit(steps):
    """1-based index of the first step that edits, and what follows it."""
    index = next(i for i, step in enumerate(steps, 1) if EDIT.search(step))
    return {"at": index, "of": len(steps), "share": round(index / len(steps), 3),
            "after": len(steps) - index}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    steps = pipelines()
    prompt_only = first_edit(steps["Prompt-only"])
    workbench = first_edit(steps["Workbench-guided"])
    touched = re.findall(r'"([^"]+)"', inspect.getsource(ref.run_prompt_only).split("touched = ")[1]
                         .split("]")[0])
    module = inspect.getsource(ref)
    return {
        "prompt_only": prompt_only, "workbench": workbench,
        "extra_steps": workbench["at"] - prompt_only["at"],
        "touched": touched,
        "off_scope": [p for p in touched if p not in ref.ALLOWED],
        "forbidden": sorted(ref.FORBIDDEN),
        "forbidden_read": module.count("FORBIDDEN") > 1,
        "imports_time": "import time" in module,
        "functions": len([v for v in vars(ref).values() if inspect.isfunction(v)
                          and v.__module__ == ref.__name__]),
        "hardcoded": module.count("TaskOutcome(") == 2,
    }


def verify(result):
    prompt, bench = result["prompt_only"], result["workbench"]
    return [
        practice.Check(
            "ANSWER: prompt-only edits at step 3 of 4, the workbench at step 4 of 8",
            all([prompt["at"] == 3, prompt["of"] == 4, bench["at"] == 4, bench["of"] == 8,
                 result["extra_steps"] == 1, prompt["after"] == 1, bench["after"] == 4]),
            f"the doc's own pipelines put the first edit at step {prompt['at']}/{prompt['of']} "
            f"({prompt['share']:.1%} of the run) against {bench['at']}/{bench['of']} "
            f"({bench['share']:.1%}) -- {result['extra_steps']} extra step of setup. After "
            f"the edit prompt-only has {prompt['after']} step and the workbench has "
            f"{bench['after']}, all of them checks",
        ),
        practice.Check(
            "FINDING: the measurement has to exclude edits outside the contract",
            all([len(result["touched"]) == 3, len(result["off_scope"]) == 2,
                 "sample_app/scripts/release.sh" in result["off_scope"]]),
            f"prompt-only touches {result['touched']} and {len(result['off_scope'])} of "
            f"{len(result['touched'])} are outside ALLOWED, so an unqualified 'first edit' "
            "would score a pipeline faster for writing to the release script first",
        ),
        practice.Check(
            "FINDING: there is no clock in the module to measure with",
            all([result["imports_time"] is False, result["functions"] == 5,
                 result["hardcoded"] is True]),
            f"{result['functions']} functions, {'no' if not result['imports_time'] else 'a'} "
            "time import, and both pipelines return hardcoded TaskOutcome literals -- a "
            "duration field would record 0.0 for both. The sixth outcome has to come from "
            "the artifacts",
        ),
        practice.Check(
            "FINDING: FORBIDDEN is declared and never read",
            all([result["forbidden"] == ["sample_app/scripts/release.sh"],
                 result["forbidden_read"] is False]),
            f"FORBIDDEN names {result['forbidden']} and nothing reads it: "
            "files_outside_scope is `p not in ALLOWED`, so a block-severity forbidden write "
            "is counted as one of two off-scope files alongside the README",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
