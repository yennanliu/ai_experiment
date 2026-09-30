"""Exercise 4 -- skipping CI saves $0 because the lesson's CI is free; the saving has to come from capping the plan.

    Build a "dry-run" mode that opens a draft PR without running CI, so reviewers can examine the plan cheap.

Reading of the exercise: dry-run is a second path through the lesson's own
dispatcher parts: `permit`, `InstallationToken.mint`, `run_agent`, then
`open_pr` with `run_verify` skipped, and the PR marked draft. Two variants
are measured against the full path (`dispatch`) on 2,000 paired issues. Each
issue gets its own seed, so both paths draw the same difficulty and the same
agent turns. "skip-CI" runs the agent loop to its end. "plan-only" also caps
the loop at 3 turns, enough to show the plan and the first edits. Drafts go
to the ledger as spend but not as PRs, and the lesson's
`record(opened_pr=True)` is tried as well to see what it does to the day's
PR cap.

**ANSWER: `dry_run` below; the plan-only variant is the cheap one.** It
costs $1.37 a draft against $4.27 for a full run, 32%. Skip-CI costs
$4.27, the same as the full run: no saving.

**FINDING: in the lesson, CI costs nothing.** `run_verify` adds $0 and 0
minutes, so every dollar of a run is agent turns. Skipping CI changes the
outcome instead. Both variants open a draft for all 2,000 issues, which
includes 79 (4.0%) whose full run went red in CI and 245 (12.3%) whose full
run hit the turn cap without converging.

**FINDING: the "no PR without CI" rule is not in `open_pr`.** Called on a run
with `ci_green=False`, `open_pr` opens the PR. The rule lives only in
`dispatch`'s if-chain. Recorded the lesson's way (`opened_pr=True`), 5
dry-runs spend acme/widget's daily PR cap, and the next real issue is
refused.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "16-github-issue-to-pr-agent"
PLAN_TURNS = 3


def dry_run(ref, task, ledger, rng, plan_turns=None, count_as_pr=False):
    """Draft PR without CI: permit -> mint -> agent loop -> open_pr (draft)."""
    difficulty = rng.uniform(0.3, 0.92)
    ok, reason = ledger.permit(task.repo, 2.0 + difficulty * 8.0)
    run = ref.SandboxRun(task)
    if not ok:
        run.failure, run.state = f"dispatcher: {reason}", ref.SState.FAILED
        return run
    token = ref.InstallationToken.mint(task.repo)
    ref.run_agent(run, difficulty, rng, **({"turn_cap": plan_turns} if plan_turns else {}))
    run.state, run.failure = ref.SState.PR, None  # a draft carries the plan, converged or not
    ref.open_pr(run, token)
    run.trace.append("draft: CI not run")
    ledger.record(task.repo, run.dollars, count_as_pr)
    return run


def paired(ref, n=2000):
    full, skip, plan = [], [], []
    for seed in range(n):
        task = ref.Task(seed, "acme/widget", 800 + seed, "labeled issue")
        full.append(ref.dispatch(task, ref.BudgetLedger(), random.Random(seed)))
        skip.append(dry_run(ref, task, ref.BudgetLedger(), random.Random(seed)))
        plan.append(dry_run(ref, task, ref.BudgetLedger(), random.Random(seed), PLAN_TURNS))
    return stats(full, skip, plan)


def stats(full, skip, plan):
    def mean(runs):
        return round(sum(r.dollars for r in runs) / len(runs), 2)

    return {
        "cost": {"full": mean(full), "skip_ci": mean(skip), "plan_only": mean(plan)},
        "minutes_equal": sum(f.wall_min == s.wall_min for f, s in zip(full, skip)),
        "drafts": (sum(r.pr_opened for r in skip), sum(r.pr_opened for r in plan)),
        "red_ci_drafts": sum(f.failure in ("flaky_test", "coverage_regression") for f in full),
        "turn_cap_full": sum(f.failure == "turn_cap" for f in full),
    }


def ci_cost(ref):
    run = ref.SandboxRun(ref.Task(0, "acme/widget", 1, "x"))
    before = (run.dollars, run.wall_min)
    ref.run_verify(run, 0.9, random.Random(0))
    return run.dollars - before[0], run.wall_min - before[1]


def gates(ref):
    red = ref.SandboxRun(ref.Task(0, "acme/widget", 1, "x"), state=ref.SState.AGENT, ci_green=False)
    ref.open_pr(red, ref.InstallationToken.mint("acme/widget"))
    ledger = ref.BudgetLedger()
    for i in range(5):
        dry_run(ref, ref.Task(i, "acme/widget", i, "dry"), ledger, random.Random(i), PLAN_TURNS, True)
    real = ref.dispatch(ref.Task(9, "acme/widget", 9, "real"), ledger, random.Random(9))
    return {"red_ci_opened": red.pr_opened, "after_5_dry_runs": real.failure}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {**paired(ref), "ci_cost": ci_cost(ref), **gates(ref)}


def verify(result):
    r = result
    c = r["cost"]
    return [
        practice.Check(
            "ANSWER: a plan-only draft costs 32% of a full run; skip-CI alone costs the same",
            (c, round(c["plan_only"] / c["full"], 2)) == ({"full": 4.27, "skip_ci": 4.27, "plan_only": 1.37}, 0.32),
            f"mean $ per issue {c}",
        ),
        practice.Check(
            "FINDING: the lesson's CI is free, so skipping it only lets red and stuck runs through",
            (r["ci_cost"], r["minutes_equal"], r["drafts"], r["red_ci_drafts"], r["turn_cap_full"])
            == ((0.0, 0.0), 2000, (2000, 2000), 79, 245),
            f"run_verify adds ($, min) {r['ci_cost']}; same minutes on {r['minutes_equal']}/2000; drafts "
            f"(skip, plan) {r['drafts']}; full runs red in CI {r['red_ci_drafts']}, at turn cap {r['turn_cap_full']}",
        ),
        practice.Check(
            "FINDING: open_pr opens a red-CI PR, and 5 counted dry-runs spend the day's PR cap",
            (r["red_ci_opened"], r["after_5_dry_runs"])
            == (True, "dispatcher: daily PR cap (5) for acme/widget reached"),
            f"open_pr with ci_green=False -> pr_opened {r['red_ci_opened']}; next real issue: "
            f"{r['after_5_dry_runs']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
