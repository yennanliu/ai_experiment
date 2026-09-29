"""Exercise 5 -- p50 TTFGB is 8.0 minutes over green repos and 26.0 over all 50, and 6% of passes finish past the 30-minute cap.

    Measure time-to-first-green-build (TTFGB) as a UX metric. Target: p50 under 10 minutes.

Reading of the exercise: TTFGB is read off the lesson's own clock,
`Attempt.wall_min`, at the moment `migrate` returns a pass; the simulation
has no separate build step, so the first green build and the pass are the
same event. A repo that never goes green has an infinite TTFGB. The p50 is
taken two ways, over the green repos only and over all 50 repos, on
`main()`'s seed 19 and then over 1,000 seeds.

**ANSWER: the target holds only if the repos that never go green are left
out.** On seed 19, 30 of 50 repos go green with p50 8.0 minutes, under
the 10-minute target; over all 50 the p50 is 26.0 minutes. Over 1,000
seeds the green-only p50 is under 10 minutes in 747 runs and the all-repo
p50 in 3.

**FINDING: 6.0% of passes go green after the 30-minute budget.**
`agent_loop` checks the budget before a turn and then adds up to 4.7
minutes, so a turn started at minute 29 still passes. Seed 19 has one at
33.9 minutes; over 1,000 seeds, 1,691 of 28,304 passes are past the cap.

**FINDING: the $8 and 20-turn caps never bind.** Over the 1,000 seeds no
repo takes more than 11 turns or spends more than $7.4725. A turn costs at
least 2.9 minutes, so the 30-minute clock stops every loop by turn 11.

**FINDING: the agent path does not time the recipe pass or the first
build.** `agent_loop` starts its clock at zero, while a recipe-only pass is
charged 3-7 minutes. All 3,221 one-turn agent passes over the 1,000 seeds
are faster than the median recipe-only pass (4.99 minutes), so needing the
agent makes a repo go green sooner.

Structure: `ttfgb()` maps an attempt to minutes or infinity; `shipped()`
is seed 19; `sweep()` counts the 1,000 seeds.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "09-code-migration-agent"
SHIPPED_SEED, SEEDS, TARGET = 19, 1000, 10.0


def run(ref, seed):
    rng = random.Random(seed)
    return [ref.migrate(repo, rng) for repo in ref.synth_bench(rng)]


def ttfgb(attempt):
    """Minutes to the first green build; a repo that never goes green never gets there."""
    return attempt.wall_min if attempt.status == "pass" else math.inf


def p50(values):
    return round(statistics.median(values), 2)


def shipped(ref):
    results = run(ref, SHIPPED_SEED)
    times = [ttfgb(a) for a in results]
    green = [t for t in times if t < math.inf]
    return {"green": len(green), "p50_green": p50(green), "p50_all": p50(times),
            "over_cap": [round(t, 1) for t in green if t > ref.BUDGET_MIN]}


def seed_row(ref, seed):
    results = run(ref, seed)
    times = [ttfgb(a) for a in results]
    green = [t for t in times if t < math.inf]
    by_turns = lambda n: [a.wall_min for a in results if a.status == "pass" and a.agent_turns == n]
    return {
        "green_ok": p50(green) < TARGET, "all_ok": p50(times) < TARGET, "passes": len(green),
        "late_passes": sum(t > ref.BUDGET_MIN for t in green),
        "max_turns": max(a.agent_turns for a in results), "max_cost": max(a.cost_usd for a in results),
        "one_turn": by_turns(1), "straight": by_turns(0),
    }


def pooled(rows, key):
    return [t for r in rows for t in r[key]]


def sweep(ref):
    rows = [seed_row(ref, seed) for seed in range(SEEDS)]
    one_turn, mid = pooled(rows, "one_turn"), statistics.median(pooled(rows, "straight"))
    out = {k: sum(r[k] for r in rows) for k in ("green_ok", "all_ok", "passes", "late_passes")}
    return {**out, "max_turns": max(r["max_turns"] for r in rows),
            "max_cost": round(max(r["max_cost"] for r in rows), 4),
            "one_turn": len(one_turn), "straight_p50": round(mid, 2), "one_turn_faster": sum(t < mid for t in one_turn)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {**shipped(ref), **sweep(ref)}


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: p50 is under 10 minutes only if the never-green repos are left out",
            (r["green"], r["p50_green"], r["p50_all"], r["green_ok"], r["all_ok"]) == (30, 8.0, 26.0, 747, 3),
            f"seed 19: {r['green']} green, p50 {r['p50_green']} min green-only, {r['p50_all']} min all 50; "
            f"under target in {r['green_ok']} / {r['all_ok']} of {SEEDS} seeds",
        ),
        practice.Check(
            "FINDING: 6.0% of passes go green after the 30-minute budget",
            (r["over_cap"], r["late_passes"], r["passes"]) == ([33.9], 1691, 28304),
            f"seed 19 late passes {r['over_cap']}; {r['late_passes']} of {r['passes']} over {SEEDS} seeds",
        ),
        practice.Check(
            "FINDING: the $8 and 20-turn caps never bind",
            (r["max_turns"], r["max_cost"]) == (11, 7.4725),
            f"most turns {r['max_turns']} of 20, most spend ${r['max_cost']} of $8",
        ),
        practice.Check(
            "FINDING: the agent path does not time the recipe pass or the first build",
            (r["one_turn"], r["one_turn_faster"], r["straight_p50"]) == (3221, 3221, 4.99),
            f"{r['one_turn_faster']}/{r['one_turn']} one-turn agent passes beat the recipe-only "
            f"median {r['straight_p50']} min",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
