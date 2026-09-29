"""Exercise 3 -- a user who floods 20 issues costs less than anyone else, and the lesson's ledger never starts a new day.

    Implement a budget dashboard: per-repo per-day cost, per-user cost. Alert on anomaly.

Reading of the exercise: the dashboard is built over the lesson's own
dispatcher and `BudgetLedger`, driven through a seeded 7-day week: 4-8
issues a day from four users across the lesson's three repos, plus one
anomaly -- on day 5 a fifth user, eve, labels 20 issues on acme/service. The
lesson's `Task` has no user field, so the dashboard keeps the user beside
each dispatch. Tiles are the two the exercise names, per-repo per-day cost
and per-user cost. Two alerts run on them: a cost alert (modified z-score
over the 21 repo-day cells above 3.5, the Iglewicz-Hoaglin cutoff) and a
pressure alert (any repo-day with dispatcher denials). The week is run twice:
one `BudgetLedger` for the week, as the lesson's `main` holds it, and a
fresh ledger per day.

**ANSWER: the dashboard, with a fresh ledger each day.** The tiles show 40
PRs over the week. The pressure alert fires once, on acme/service day 5,
where eve's flood drew 18 of the week's 18 denials. The cost alert flags
acme/service day 5 ($17.00) and two ordinary days: acme/widget day 1
($15.08) and acme/library day 7 ($13.07), each a handful of hard issues.

**FINDING: per-user cost is the wrong tile to catch a flood.** Eve filed 20
issues and cost $10.87, less than every other user ($14.04-39.05). The daily PR
cap turned her away at dispatch, and denials cost $0. Only the denial count
sees her.

**FINDING: the lesson's ledger has no day in it.** `spent_today` and
`prs_today` are keyed by repo alone. Run as one ledger for the week, the PR
caps are spent by day 4: days 5-7 open 0 PRs (15 PRs in the week against
40), and 44 of the 59 issues are denied.

**FINDING: the $20 and 30-minute caps cannot fire, and the reservation
reserves nothing.** At the default 20 turns a run costs at most $13.28 and
takes 29.04 minutes. `permit` checks spend so far plus $20 but records no
hold, so 8 issues admitted before any finishes all pass. Run at difficulty
0.92, together they spend $65.07 against the $50 daily cap.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "16-github-issue-to-pr-agent"
REPOS = ["acme/widget", "acme/service", "acme/library"]
USERS = ["ana", "ben", "chen", "dee"]


def traffic():
    rng = random.Random(16)
    events = []
    for day in range(1, 8):
        events += [(day, rng.choice(REPOS), rng.choice(USERS)) for _ in range(rng.randint(4, 8))]
        if day == 5:
            events += [(5, "acme/service", "eve")] * 20
    return events


def week(ref, per_day):
    rng, ledger, day_of, rows = random.Random(9), None, None, []
    for i, (day, repo, user) in enumerate(traffic()):
        if ledger is None or (per_day and day != day_of):
            ledger, day_of = ref.BudgetLedger(), day
        run = ref.dispatch(ref.Task(i, repo, 800 + i, "labeled issue"), ledger, rng)
        denied = (run.failure or "").startswith("dispatcher")
        rows.append({"day": day, "repo": repo, "user": user, "cost": run.dollars,
                     "pr": run.pr_opened, "denied": denied})
    return rows


def tiles(rows):
    cell = {(r, d): 0.0 for r in REPOS for d in range(1, 8)}
    denials, user_cost = dict.fromkeys(cell, 0), {}
    for x in rows:
        cell[x["repo"], x["day"]] += x["cost"]
        denials[x["repo"], x["day"]] += x["denied"]
        user_cost[x["user"]] = user_cost.get(x["user"], 0.0) + x["cost"]
    return cell, denials, user_cost


def cost_alerts(cell):
    """Modified z-score (Iglewicz-Hoaglin) over every repo-day cell, cutoff 3.5."""
    med = statistics.median(cell.values())
    mad = max(statistics.median(abs(v - med) for v in cell.values()), 0.01)
    return sorted((k, round(v, 2)) for k, v in cell.items() if 0.6745 * (v - med) / mad > 3.5)


def dashboard(rows):
    cell, denials, user_cost = tiles(rows)
    return {
        "cost_alerts": cost_alerts(cell),
        "pressure_alerts": sorted((k, n) for k, n in denials.items() if n),
        "user_cost": {u: round(c, 2) for u, c in sorted(user_cost.items())},
        "prs_by_day": [sum(x["pr"] for x in rows if x["day"] == d) for d in range(1, 8)],
        "denied": sum(x["denied"] for x in rows), "issues": len(rows),
    }


def burst(ref, n=8):
    """n issues admitted before any finishes: permit() holds nothing back."""
    ledger = ref.BudgetLedger()
    admitted = [ledger.permit("acme/widget", 9.36)[0] for _ in range(n)]
    for seed in range(n):
        run = ref.SandboxRun(ref.Task(seed, "acme/widget", seed, "burst"))
        ref.run_agent(run, 0.92, random.Random(seed))
        ledger.record("acme/widget", run.dollars, False)
    return sum(admitted), round(ledger.spent_today["acme/widget"], 2)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    max_run = ref.SandboxRun(ref.Task(0, "x", 0, "worst case"))
    ref.run_agent(max_run, 0.92, random.Random(0), turn_cap=20, dollar_cap=1e9, minute_cap=1e9)
    return {"daily": dashboard(week(ref, True)), "weekly": dashboard(week(ref, False)),
            "burst": burst(ref), "worst": (round(max_run.dollars, 2), round(max_run.wall_min, 2))}


def verify(result):
    d, w = result["daily"], result["weekly"]
    uc = d["user_cost"]
    return [
        practice.Check(
            "ANSWER: per-day tiles, 40 PRs; pressure alert on service day 5, cost alert on it plus 2 false",
            (sum(d["prs_by_day"]), d["pressure_alerts"], d["cost_alerts"])
            == (40, [(("acme/service", 5), 18)],
               [(("acme/library", 7), 13.07), (("acme/service", 5), 17.0), (("acme/widget", 1), 15.08)]),
            f"PRs by day {d['prs_by_day']}; pressure {d['pressure_alerts']}; cost {d['cost_alerts']}",
        ),
        practice.Check(
            "FINDING: the flooding user costs less than every other user",
            uc["eve"] == 10.87 and uc["eve"] < min(v for u, v in uc.items() if u != "eve"),
            f"per-user cost {uc}",
        ),
        practice.Check(
            "FINDING: one ledger for the week spends every PR cap by day 4",
            (w["prs_by_day"], w["denied"], w["issues"]) == ([6, 4, 3, 2, 0, 0, 0], 44, 59),
            f"PRs by day {w['prs_by_day']}; {w['denied']}/{w['issues']} issues denied",
        ),
        practice.Check(
            "FINDING: the $20/30-min caps are unreachable and 8 unreserved admits spend $65 on a $50 cap",
            (result["worst"], result["burst"]) == ((13.28, 29.04), (8, 65.07)),
            f"worst run ($, min) {result['worst']}; burst (admitted, spent) {result['burst']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
