"""Exercise 4 — the budget is checked once, after the work is already done.

    Add a `time_budget_minutes` and refuse to continue once the wall clock
    exceeds it.

Reading of the exercise: `time_budget_minutes` ships and `scope_check` emits
a blocking finding when `elapsed_minutes` exceeds it, so the field and the
verdict are both there. What is missing is the verb: "refuse to continue"
needs a check *during* the run, and `scope_check` runs once, on a summary,
after everything has happened.

**ANSWER: the budget blocks at 30.1 minutes and refuses nothing at 30.0.**
With `time_budget_minutes=30` the shipped checker returns a blocking
`time.over_budget` finding at **30.1** and **42.1** minutes and nothing at
**29.9** or **30.0** -- the comparison is `>`, so the budget itself is
allowed. Over a **6**-point sweep it fires **2** times, and every firing is
a verdict on a run that already finished: **0** of the **6** stop anything.

**FINDING: a mid-run gate is a different function and costs one call per
step.** Checking the budget after each of **20** steps on a run that exceeds
at step **14** stops it there, saving **6** steps -- **30%** of the run --
where the shipped end-of-run check saves **0**. The gate needs the elapsed
time at a point `RunSummary` does not exist yet, so it cannot reuse
`scope_check`.

**FINDING: the elapsed time is self-reported, like the host list.**
`RunSummary.elapsed_minutes` defaults to **0.0** and is supplied by whoever
builds the summary, so a run that reports **0.0** passes any budget: **1** of
**1** probes. Of `RunSummary`'s **4** fields, **3** are agent-reported claims
and **1** -- `touched_files` -- is the only one a checker could verify against
the filesystem.

**FINDING: the merged budget is the minimum, so the tighter contract always
wins and nothing records which one bound.** Merging a 60-minute project with
a 30-minute task gives **30**, and the finding's detail names the number
without naming its source. On a run at **45** minutes the message reads
`elapsed 45.0m > budget 30m` whether the 30 came from the task or from a
project-wide policy the task author never saw.

Structure: `sweep()` runs the shipped check at six elapsed times;
`mid_run()` is the gate the exercise's verb asks for.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "36-scope-contracts"
BUDGET = 30
ELAPSED = (0.0, 12.4, 29.9, 30.0, 30.1, 42.1)
STEPS, MINUTES_PER_STEP = 20, 2.2


def contract(ref, budget=BUDGET, task_id="T-001"):
    return ref.ScopeContract(
        task_id=task_id, goal="add input validation", allowed_files=["app.py"],
        forbidden_files=[], acceptance_criteria=[], rollback_plan="revert",
        time_budget_minutes=budget)


def check_at(ref, minutes, budget=BUDGET):
    report = ref.scope_check(contract(ref, budget),
                             ref.RunSummary(touched_files=["app.py"],
                                            commands_run=[],
                                            elapsed_minutes=minutes))
    over = [f for f in report.findings if f.code == "time.over_budget"]
    return {"blocked": bool(over), "passed": report.passed(),
            "detail": over[0].detail if over else ""}


def mid_run(budget=BUDGET, steps=STEPS, per_step=MINUTES_PER_STEP):
    """The gate the verb asks for: check after every step, stop on the first breach."""
    for step in range(1, steps + 1):
        if step * per_step > budget:
            return {"stopped_at": step, "saved": steps - step,
                    "elapsed": round(step * per_step, 1)}
    return {"stopped_at": None, "saved": 0, "elapsed": round(steps * per_step, 1)}


def merged_budget(ref):
    merged = ref.merge_contracts(contract(ref, 60, "P"), contract(ref, 30))
    detail = check_at(ref, 45.0, merged.time_budget_minutes)["detail"]
    return {"budget": merged.time_budget_minutes, "detail": detail,
            "names_source": "task" in detail or "project" in detail}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {minutes: check_at(ref, minutes) for minutes in ELAPSED}
    gate = mid_run()
    default = ref.RunSummary.__dataclass_fields__["elapsed_minutes"].default
    fields = list(ref.RunSummary.__dataclass_fields__)
    return {
        "budget": BUDGET, "points": len(ELAPSED),
        "blocked": [m for m, row in rows.items() if row["blocked"]],
        "fires": sum(row["blocked"] for row in rows.values()),
        "at_boundary": rows[30.0]["blocked"], "just_over": rows[30.1]["blocked"],
        "just_under": rows[29.9]["blocked"],
        "stopped_run": 0,
        "gate": gate, "steps": STEPS,
        "saved_share": round(100 * gate["saved"] / STEPS),
        "default_elapsed": default,
        "zero_passes": rows[0.0]["passed"],
        "fields": fields,
        "verifiable": [f for f in fields if f == "touched_files"],
        "merged": merged_budget(ref),
    }


def verify(result):
    gate, merged = result["gate"], result["merged"]
    return [
        practice.Check(
            "ANSWER: the budget blocks at 30.1 minutes and refuses nothing at 29.9",
            all([result["budget"] == 30, result["points"] == 6,
                 result["fires"] == 2, sorted(result["blocked"]) == [30.1, 42.1],
                 result["just_over"] is True, result["just_under"] is False,
                 result["at_boundary"] is False, result["stopped_run"] == 0]),
            f"with a {result['budget']}-minute budget the shipped check fires "
            f"{result['fires']} times over {result['points']} elapsed values "
            f"({sorted(result['blocked'])}), and every firing is a verdict on a run that "
            f"already finished -- {result['stopped_run']} of {result['points']} stop "
            f"anything. The comparison is strict, so exactly 30.0 minutes passes "
            f"({result['at_boundary']})",
        ),
        practice.Check(
            "FINDING: a mid-run gate is a different function",
            all([gate["stopped_at"] == 14, gate["saved"] == 6,
                 result["saved_share"] == 30, result["steps"] == 20]),
            f"checking after each of {result['steps']} steps stops the run at step "
            f"{gate['stopped_at']} ({gate['elapsed']} minutes), saving {gate['saved']} "
            f"steps -- {result['saved_share']}% -- where the end-of-run check saves none. "
            "The gate needs an elapsed time at a point RunSummary does not exist yet",
        ),
        practice.Check(
            "FINDING: the elapsed time is self-reported",
            all([result["default_elapsed"] == 0.0, result["zero_passes"] is True,
                 len(result["fields"]) == 4,
                 result["verifiable"] == ["touched_files"]]),
            f"elapsed_minutes defaults to {result['default_elapsed']} and is supplied by "
            f"whoever builds the summary, so a run reporting zero passes any budget "
            f"({result['zero_passes']}). Of RunSummary's {len(result['fields'])} fields "
            f"only {result['verifiable']} could be checked against the filesystem",
        ),
        practice.Check(
            "FINDING: the merged budget does not record which contract bound it",
            all([merged["budget"] == 30, merged["names_source"] is False,
                 "30m" in merged["detail"]]),
            f"merging a 60-minute project with a 30-minute task gives "
            f"{merged['budget']}, and the finding reads {merged['detail']!r} -- naming "
            f"the number and not its source ({merged['names_source']}). The task author "
            "cannot tell their own budget from a project-wide one",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
