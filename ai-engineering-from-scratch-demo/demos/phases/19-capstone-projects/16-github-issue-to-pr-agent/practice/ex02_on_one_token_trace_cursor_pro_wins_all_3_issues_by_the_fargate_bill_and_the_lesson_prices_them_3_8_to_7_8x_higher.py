"""Exercise 2 -- on one token trace Cursor Pro wins all 3 issues by the Fargate bill, Teams loses all 3, and the lesson prices them 3.8-7.8x higher.

    Compare cost vs Cursor Background Agents on three shared issues. Report which tools win where.

Reading of the exercise: no Cursor run can happen offline, so the comparison
is held on the one axis both products expose in the same unit: dollars for the
same agent trace on the same model. The three shared issues are the lesson's
own difficulty range -- easy 0.30, medium 0.61 and hard 0.92, the ends and
middle of `dispatch`'s `uniform(0.3, 0.92)` -- each run 2,000 times through
the lesson's `run_agent` and `run_verify`. Every run's turn count becomes a
token trace, with the same stated profile for both products: 30,000 tokens of
prompt, tools and repo map written to the cache on turn 1, then each turn
re-reads the cached context, writes 4,000 new tokens and emits 800. Prices
(read 2026-09-29): Claude Opus 4.7 at $5 input, $6.25 cache write, $0.50
cache read and $25 output per million tokens
(platform.claude.com/docs/en/about-claude/pricing). Cursor lists the same
rates and says "Cloud Agents are charged at API pricing for the selected
model" (cursor.com/docs/cloud-agent). Teams plans add a "Cursor Token Rate of
$0.25 per million tokens" on third-party models
(cursor.com/docs/models-and-pricing), which the page does not scope, so it is
applied to every token and, as a lower bound, to non-cached ones only. The
self-hosted worker adds Fargate at 2 vCPU and 4 GB, at $0.000011244 per
vCPU-second and $0.000001235 per GB-second with a 1-minute minimum
(aws.amazon.com/fargate/pricing), for the lesson's wall-clock minutes.

**ANSWER: the platform barely matters; the model tokens are 97-98% of the
self-hosted bill.** Dollars per resolved issue:

| issue | resolved | lesson `dollars` | tokens | + Fargate | Cursor Pro | Cursor Teams |
|---|---:|---:|---:|---:|---:|---:|
| easy 0.30 | 94.3% | 1.64 | 0.432 | 0.440 | 0.432 | 0.479 |
| medium 0.61 | 90.3% | 3.95 | 0.680 | 0.696 | 0.680 | 0.777 |
| hard 0.92 | 60.5% | 14.00 | 1.800 | 1.850 | 1.800 | 2.127 |

Cursor Pro wins all three, by the Fargate bill (0.8 to 5 cents), because its
VM is not billed. Teams loses all three: its token rate adds 11-18% to the
model spend, against 1.9-2.8% for Fargate. If cached reads are exempt,
Teams still loses easy, ties medium and wins hard (3.6 cents against 5.0).

**FINDING: the lesson's `dollars` is not a token price.** It charges a flat
0.25 + 0.45 x difficulty per turn, so it prices the three issues 3.8x, 5.8x and
7.8x above the token bill. It would not move if the model changed.

**FINDING: the doc's 30-turn cap would buy the hard issue 1.4 points.**
`run_agent` defaults to 20 turns, where the doc says 30, and 743 of the hard
issue's 2,000 runs stop there. At 30 turns the 30-minute cap trips instead, at
turn 21 (1.452 minutes a turn), on 714 runs, so it resolves 61.9% against
60.5%.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "16-github-issue-to-pr-agent"
ISSUES = {"easy": 0.30, "medium": 0.61, "hard": 0.92}
BASE, STEP, OUT = 30_000, 4_000, 800  # tokens: cached prefix, new per turn, output per turn
OPUS = {"read": 0.50, "write": 6.25, "out": 25.0}  # $ per million tokens
TOKEN_RATE = 0.25
FARGATE_PER_MIN = 60 * (2 * 0.000011244 + 4 * 0.000001235)


def trace(turns):
    """(cache reads, cache writes, output) for a run of `turns` turns."""
    reads = sum(BASE + STEP * (k - 2) for k in range(2, turns + 1))
    return reads, BASE + STEP * (turns - 1), OUT * turns


def bill(run):
    reads, writes, out = trace(run.turns)
    model = (reads * OPUS["read"] + writes * OPUS["write"] + out * OPUS["out"]) / 1e6
    return {"lesson": run.dollars, "tokens": model,
            "self_hosted": model + max(1.0, run.wall_min) * FARGATE_PER_MIN,
            "cursor_pro": model, "cursor_teams": model + (reads + writes + out) * TOKEN_RATE / 1e6,
            "teams_no_reads": model + (writes + out) * TOKEN_RATE / 1e6}


def issue(ref, d, turn_cap=20, n=2000):
    totals, resolved, fails = {}, 0, {}
    for seed in range(n):
        rng = random.Random(seed)
        run = ref.SandboxRun(ref.Task(seed, "acme/widget", 842, "shared issue"))
        ref.run_agent(run, d, rng, turn_cap=turn_cap)
        if run.state == ref.SState.VERIFY:
            ref.run_verify(run, d, rng)
        resolved += run.state == ref.SState.PR
        fails[run.failure] = fails.get(run.failure, 0) + 1
        for k, v in bill(run).items():
            totals[k] = totals.get(k, 0.0) + v
    per = {k: round(v / resolved, 3) for k, v in totals.items()}
    return {"resolved": resolved / n, "fails": fails, **per}


def winner(row, pro_or_teams):
    return "cursor" if row[pro_or_teams] < row["self_hosted"] else (
        "tie" if abs(row[pro_or_teams] - row["self_hosted"]) < 0.001 else "self_hosted")


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {name: issue(ref, d) for name, d in ISSUES.items()}
    return {
        "rows": rows,
        "wins": {n: [winner(r, k) for k in ("cursor_pro", "cursor_teams", "teams_no_reads")]
                 for n, r in rows.items()},
        "token_share": [round(r["tokens"] / r["self_hosted"], 3) for r in rows.values()],
        "lesson_over": [round(r["lesson"] / r["tokens"], 1) for r in rows.values()],
        "hard_fails": {c: issue(ref, ISSUES["hard"], turn_cap=c)["fails"] for c in (20, 30)},
        "hard_at_30": issue(ref, ISSUES["hard"], turn_cap=30)["resolved"],
        "default_cap": ref.run_agent.__defaults__[0],
    }


def verify(result):
    r = result
    rows = r["rows"]
    table = {n: (x["resolved"], x["lesson"], x["tokens"], x["self_hosted"], x["cursor_teams"])
             for n, x in rows.items()}
    return [
        practice.Check(
            "ANSWER: Cursor Pro wins all 3 by the Fargate bill, Teams loses all 3; tokens are 97-98%",
            table == {"easy": (0.943, 1.64, 0.432, 0.44, 0.479), "medium": (0.9025, 3.951, 0.68, 0.696, 0.777),
                      "hard": (0.605, 14.005, 1.8, 1.85, 2.127)}
            and r["wins"] == {"easy": ["cursor", "self_hosted", "self_hosted"],
                              "medium": ["cursor", "self_hosted", "tie"],
                              "hard": ["cursor", "self_hosted", "cursor"]}
            and r["token_share"] == [0.982, 0.977, 0.973],
            f"(resolved, lesson, tokens, self-hosted, Teams) per issue {table}; winners (Pro, Teams, "
            f"Teams without cache reads) {r['wins']}; token share of self-hosted {r['token_share']}",
        ),
        practice.Check(
            "FINDING: the lesson's flat per-turn dollars price the issues 3.8-7.8x above the token bill",
            r["lesson_over"] == [3.8, 5.8, 7.8],
            f"lesson dollars / token bill per resolved issue: {r['lesson_over']}",
        ),
        practice.Check(
            "FINDING: at the doc's 30 turns the 30-minute cap trips at turn 21, buying 1.4 points",
            (r["default_cap"], r["hard_at_30"], r["hard_fails"]) == (20, 0.619, {
                20: {None: 1210, "turn_cap": 743, "flaky_test": 47},
                30: {None: 1238, "minute_cap": 714, "flaky_test": 48}}),
            f"turn_cap default {r['default_cap']}; hard issue resolves {r['hard_at_30']} at 30 turns; "
            f"outcomes at 20 and 30 turns {r['hard_fails']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
