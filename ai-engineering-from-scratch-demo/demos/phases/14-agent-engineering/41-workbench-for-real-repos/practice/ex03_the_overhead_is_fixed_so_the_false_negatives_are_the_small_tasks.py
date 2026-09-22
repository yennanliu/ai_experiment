"""Exercise 3 — the overhead is fixed, so the false negatives are the small tasks.

    Add a "false negative" pass: tasks where prompt-only would have been
    faster and the workbench overhead is real cost. Defend keeping the
    workbench anyway.

Reading of the exercise: "real cost" wants a number. The workbench's cost is
fixed -- **7** of its **8** steps are not edits -- so its overhead as a share
of the run depends only on how many edits the task needs. That makes the false
negatives identifiable in advance rather than in hindsight.

**ANSWER: the ratio runs from 2.00x on a one-edit task to 1.27x on a
twelve-edit one, and all 5 enumerated false negatives need 1 edit or fewer.**
A formatter run, a one-line lint fix, a version bump, a typo in a docstring
and a single-fact lookup: prompt-only finishes each in **4** steps or fewer
where the workbench spends **8**. The overhead is real and it is worst exactly
where the task is smallest.

**FINDING: the saving is conditional on a fact only the skipped steps
establish.** **0** of prompt-only's **4** steps inspect the diff, so "it was
only one line" is not something that pipeline can know -- it is what the scope
check, step **6** of the workbench, would have told you. Every false negative
is a bet that the task was what it looked like before anyone measured.

**FINDING: on cost alone, "always run it" does not survive the arithmetic.**
The workbench costs **4** extra steps on a one-edit task; a prompt-only miss
costs a redo plus a guided run, **12** steps, so it pays off above a
**33.3%** failure rate. The figures the lesson itself cites straddle that
line -- Vercel's agent at 80% success, WebAgent's baseline at 40-50% -- so the
defence has to be sharper than an average.

**FINDING: the defensible version keeps 2 of the 8 steps and 2 of the 5
outcomes.** The acceptance run and the verification gate are **25.0%** of the
pipeline and they keep `files_outside_scope` and `acceptance_met` measurable --
the first of which Exercise 2 found slipping on a real task. The fast path
gives up `handoff_quality` and `reviewer_total` on purpose, which is a
statement a reviewer can argue with, unlike "we skipped the workbench".

Structure: `TASKS` enumerates the false negatives with their edit counts;
`cost()` turns edits into step counts for both pipelines.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "41-workbench-for-real-repos"
PROMPT_FIXED, BENCH_FIXED = 3, 7  # non-edit steps in each published pipeline

# (task, edits it needs, why prompt-only is defensible)
TASKS = [
    ("run the formatter", 1, "the formatter decides the diff, not the agent"),
    ("one-line lint fix", 1, "the linter named the file, the line and the rule"),
    ("bump a pinned version", 1, "the change is a literal the human supplied"),
    ("fix a typo in a docstring", 1, "no behaviour to verify"),
    ("answer where a symbol is defined", 0, "no edit at all"),
]
REAL_TASKS = [("add signup validation", 2), ("rename across three files", 3),
              ("refactor a twelve-file module", 12)]


def cost(edits):
    prompt, bench = PROMPT_FIXED + edits, BENCH_FIXED + edits
    return {"prompt": prompt, "bench": bench, "ratio": round(bench / prompt, 2),
            "overhead_share": round(BENCH_FIXED / bench, 3)}


def breakeven(edits=1, redo=None):
    """Extra steps now against the cost of being wrong later."""
    extra = cost(edits)["bench"] - cost(edits)["prompt"]
    redo = redo if redo is not None else cost(edits)["prompt"] + cost(edits)["bench"]
    return extra, redo, round(extra / redo, 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    small, large = cost(1), cost(12)
    extra, redo, rate = breakeven()
    kept = ["Run acceptance command via feedback runner", "Run verification gate"]
    return {
        "small": small, "large": large,
        "false_negatives": len(TASKS),
        "max_edits": max(edits for _, edits, _ in TASKS),
        "real_edits": [edits for _, edits in REAL_TASKS],
        "prompt_inspects_diff": 0, "prompt_steps": PROMPT_FIXED + 1,
        "scope_step": 6,
        "extra": extra, "redo": redo, "breakeven": rate,
        "kept": len(kept), "pipeline_steps": BENCH_FIXED + 1,
        "kept_share": round(len(kept) / (BENCH_FIXED + 1), 3),
        "outcomes": len(ref.TaskOutcome.__dataclass_fields__) - 1,
        "retained": ["files_outside_scope", "acceptance_met"],
        "given_up": ["handoff_quality", "reviewer_total"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the ratio runs 2.00x to 1.27x and every false negative needs one edit",
            all([result["small"]["ratio"] == 2.0, result["large"]["ratio"] == 1.27,
                 result["small"]["bench"] == 8, result["small"]["prompt"] == 4,
                 result["false_negatives"] == 5, result["max_edits"] == 1]),
            f"a one-edit task costs {result['small']['prompt']} steps prompt-only against "
            f"{result['small']['bench']} guided ({result['small']['ratio']}x, "
            f"{result['small']['overhead_share']:.1%} of it fixed overhead), falling to "
            f"{result['large']['ratio']}x at twelve edits; all "
            f"{result['false_negatives']} enumerated false negatives need "
            f"{result['max_edits']} edit or fewer",
        ),
        practice.Check(
            "FINDING: the saving is conditional on a fact only the skipped steps establish",
            all([result["prompt_inspects_diff"] == 0, result["prompt_steps"] == 4,
                 result["scope_step"] == 6]),
            f"{result['prompt_inspects_diff']} of prompt-only's {result['prompt_steps']} "
            f"steps inspect the diff, so 'it was only one line' is what step "
            f"{result['scope_step']} of the workbench would have told you. Every false "
            "negative is a bet the task was what it looked like",
        ),
        practice.Check(
            "FINDING: on cost alone, 'always run it' does not survive the arithmetic",
            all([result["extra"] == 4, result["redo"] == 12, result["breakeven"] == 0.333]),
            f"{result['extra']} extra steps on a one-edit task against a "
            f"{result['redo']}-step redo pays off above a {result['breakeven']:.1%} failure "
            "rate, and the rates the lesson cites -- 80% success at Vercel, 40-50% for "
            "WebAgent -- straddle it. The defence has to be sharper than an average",
        ),
        practice.Check(
            "FINDING: the defensible version keeps 2 of the 8 steps and 2 of the 5 outcomes",
            all([result["kept"] == 2, result["pipeline_steps"] == 8,
                 result["kept_share"] == 0.25, result["outcomes"] == 5,
                 len(result["retained"]) == 2, len(result["given_up"]) == 2]),
            f"the acceptance run and the verification gate are {result['kept']} of "
            f"{result['pipeline_steps']} steps ({result['kept_share']:.1%}) and they keep "
            f"{result['retained']} measurable -- the first of which Exercise 2 found slipping "
            f"on a real task. The fast path gives up {result['given_up']} by design",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
