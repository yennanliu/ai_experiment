"""Exercise 1 — N=2 does produce more work-tokens, and N=8 produces exactly as much progress as N=4.

    Run `code/main.py` with the default settings. Confirm the N=2 Hogwild!
    configuration produces more work-tokens than the N=1 baseline in the same
    wall time.

Reading of the exercise: "the same wall time" is the same `step_budget`, since
`run_hogwild` advances one step per iteration for every worker at once, so the
confirmation is one call per worker count. The sweep is then extended to N=4 and
N=8, because the exercise asks for a comparison and two points cannot show
whether it continues.

**ANSWER: confirmed, and it stops at 2.**

    N   tokens   work tokens   unique progress   progress / step
    1      200          193               193          0.96
    2      400          385               385          1.93
    4      800          761               400          2.00
    8     1600         1511               400          2.00

Work tokens scale linearly with N -- they are just "tokens that were not noise".
Unique progress is **identical at N=4 and N=8**, at 400 in both cases.

**MECHANISM: there are two work categories, so at most two tokens per step can
be unique.** `run_hogwild` credits progress once per distinct category in
`("A", "B")` per step; the rest of that step's tokens are tagged `redundant`.
With 8 workers, 6 of every 8 tokens are redundant by construction, whatever they
contain.

**FINDING: the metric the exercise names is the one that cannot saturate.**
"Work-tokens" counts every non-noise token, so it grows with N forever and
confirms the exercise's claim at any N. `unique_progress`, which is what the
simulator computes redundancy for, saturates at `2 * step_budget`. The exercise
asks for the measurement that always passes.

**FINDING: the useful fraction falls from 100% to 26%.** At N=1, 193 of 193 work
tokens are unique; at N=8, 400 of 1511. The simulator is reporting exactly the
phenomenon Hogwild! exists to avoid, and the headline metric does not see it.

Structure: `arm` runs the lesson's own `run_hogwild` at one worker count;
`useful` is the share of work tokens that were not redundant.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "22-async-hogwild-inference"
STEPS, TARGET, WEIGHT = 200, 100, 1.0
WORKERS = (1, 2, 4, 8)
CATEGORIES = ("A", "B")


def arm(ref, workers, weight=WEIGHT):
    row = ref.run_hogwild(workers, STEPS, TARGET, weight)
    return dict(row, useful=row["unique_progress"] / row["work_tokens"])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {n: arm(ref, n) for n in WORKERS}
    return {
        "rows": rows,
        "work_growth": {n: rows[n]["work_tokens"] / rows[1]["work_tokens"] for n in WORKERS},
        "progress_growth": {n: rows[n]["unique_progress"] / rows[1]["unique_progress"]
                            for n in WORKERS},
        "ceiling": len(CATEGORIES) * STEPS,
        "categories": CATEGORIES,
    }


def column(rows, field, fmt):
    return ", ".join(f"N={n} {format(row[field], fmt)}" for n, row in rows.items())


def verify(result):
    rows = result["rows"]
    one, two, four, eight = (rows[n] for n in WORKERS)
    return [
        practice.Check(
            "ANSWER: confirmed at N=2, and unique progress stops growing after N=4",
            two["work_tokens"] > one["work_tokens"]
            and four["unique_progress"] == eight["unique_progress"],
            f"in the same {STEPS}-step budget the workers emit " + column(rows, "tokens_emitted", "d")
            + " tokens, of which " + column(rows, "work_tokens", "d")
            + " are work tokens -- so N=2 produces "
            f"{two['work_tokens'] / one['work_tokens']:.2f}x the N=1 baseline, as the exercise "
            f"asks. Unique progress is " + column(rows, "unique_progress", "d")
            + ": identical at N=4 and N=8",
        ),
        practice.Check(
            "MECHANISM: two work categories means at most two unique tokens per step",
            eight["unique_progress"] == result["ceiling"]
            and eight["progress_per_step"] == len(CATEGORIES),
            f"run_hogwild credits progress once per distinct category in {list(CATEGORIES)} per "
            f"step and tags the rest redundant, so progress_per_step is bounded by "
            f"{len(CATEGORIES)} and the budget by {result['ceiling']}. At N=8 it reaches exactly "
            f"{eight['progress_per_step']:.2f} per step -- 6 of every 8 tokens are redundant by "
            "construction, whatever they contain",
        ),
        practice.Check(
            "FINDING: the metric the exercise names is the one that cannot saturate",
            result["work_growth"][8] > 7 and result["progress_growth"][8] < 2.5,
            "work tokens grow " + column(result["rows"], "work_tokens", "d")
            + " -- a factor of " + ", ".join(f"{v:.2f}" for v in result["work_growth"].values())
            + " -- while unique progress grows "
            + ", ".join(f"{v:.2f}" for v in result["progress_growth"].values())
            + "x. 'Work-tokens' counts every non-noise token and confirms the exercise's claim at "
            "any N; unique_progress is what the simulator computes redundancy for, and the "
            "exercise asks for the other one",
        ),
        practice.Check(
            "FINDING: the useful fraction falls from 100% to 26%",
            one["useful"] == 1.0 and eight["useful"] < 0.3,
            "the share of work tokens that were not redundant is " + column(rows, "useful", ".0%")
            + f". At N=1 all {one['work_tokens']} are unique; at N=8, "
            f"{eight['unique_progress']} of {eight['work_tokens']}. The simulator is reporting "
            "exactly the phenomenon Hogwild! exists to avoid, and the headline metric does not "
            "see it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
