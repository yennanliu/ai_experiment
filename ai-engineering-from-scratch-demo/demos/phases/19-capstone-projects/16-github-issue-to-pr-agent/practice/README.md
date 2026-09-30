<!-- generated:start -->
# 19-capstone-projects / 16-github-issue-to-pr-agent

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/16-github-issue-to-pr-agent/) · upstream spec
`phases/19-capstone-projects/16-github-issue-to-pr-agent/docs/en.md`

```bash
uv run demo practice run 16-github-issue-to-pr-agent --ex 1
uv run demo explain 16-github-issue-to-pr-agent --ex 1
uv run pytest demos/phases/19-capstone-projects/16-github-issue-to-pr-agent
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a "fix flaky test" mode: the label `@agent stabilize-flake TestX` runs the test 50 times… | code | T0 | `ex01_50_green_runs_accept_a_timeout_bump_that_still_fails_1_run_in_360_and_the_label_never_reaches_the_agent.py` |
| 2 | Compare cost vs Cursor Background Agents on three shared issues. Report which tools win where. | code | T0 | `ex02_on_one_token_trace_cursor_pro_wins_all_3_issues_by_the_fargate_bill_and_the_lesson_prices_them_3_8_to_7_8x_higher.py` |
| 3 | Implement a budget dashboard: per-repo per-day cost, per-user cost. Alert on anomaly. | code | T0 | `ex03_a_user_who_floods_20_issues_costs_less_than_anyone_else_and_the_lessons_ledger_never_starts_a_new_day.py` |
| 4 | Build a "dry-run" mode that opens a draft PR without running CI, so reviewers can examine the… | code | T0 | `ex04_skipping_ci_saves_0_dollars_because_the_lessons_ci_is_free_and_a_3_turn_plan_costs_32pct_of_a_run.py` |
| 5 | Add a retention policy: PR branches older than 7 days without merge get deleted automatically. | code | T0 | `ex05_4_of_the_7_day_policys_5_deletions_close_an_open_pr_and_the_lessons_token_would_delete_main.py` |
<!-- generated:end -->

## Answers

Every exercise imports and runs the lesson's `code/main.py`: a seeded
dispatcher with a `BudgetLedger`, an `InstallationToken`, and an agent loop
whose turns cost a flat 0.25 + 0.45 x difficulty dollars. Three exercises
also read the TypeScript webhook receiver in `code/ts/src/`. No model,
sandbox or GitHub is called, so each "live" part is a seeded stand-in, and
each file's docstring says which one. External facts were read on 2026-09-29
from Anthropic's pricing page, Cursor's Cloud Agents and Models & Pricing
docs, AWS Fargate pricing, and GitHub's webhook and branch docs.

### 1 — 50 green runs accept a timeout bump that still fails 1 run in 360, and the label never reaches the agent

**The mode proposes "bump the timeout 80 -> 160 ms", a 1-line diff.** The
stand-in `TestX` waits 80 ms for a job with lognormal latency. It fails 5 of
the first 50 runs. The patches are tried smallest diff first. A rerun
decorator is refused because it only hides the failure, and the timeout
bump then goes 50/50 green.

**50 green runs cannot prove a test is stable.**

| patch | true fail rate | passes the 50-run gate |
|---|---:|---:|
| none | 8.28% | — |
| timeout 80 -> 160 ms | 0.28% (1 in 360) | 864 / 1,000 episodes |
| wait on the job's done event | 0 | never reached |

50/50 green only shows the fail rate is below 5.8% (at 95% confidence).
That is above the lesson's own 5% CI flake rate. The lesson also cannot
start this mode. Its router dispatches only `issues.opened`, and adding a
label arrives as `issues.labeled`, which gets "skipped". `run_verify` flakes
5.4% of the time on any issue: it never reads its `difficulty` argument.

### 2 — on one token trace Cursor Pro wins all 3 issues by the Fargate bill, and the lesson prices them 3.8-7.8x higher

**On the same model and the same trace, the platforms differ by cents.** The
model tokens are 97-98% of the self-hosted bill. The three shared issues are
the ends and middle of the lesson's difficulty range, run 2,000 times each.
Every run's turns are priced as Claude Opus 4.7 tokens under one stated
profile. Dollars per resolved issue:

| issue | resolved | lesson `dollars` | tokens | + Fargate | Cursor Pro | Cursor Teams |
|---|---:|---:|---:|---:|---:|---:|
| easy 0.30 | 94.3% | 1.64 | 0.432 | 0.440 | 0.432 | 0.479 |
| medium 0.61 | 90.3% | 3.95 | 0.680 | 0.696 | 0.680 | 0.777 |
| hard 0.92 | 60.5% | 14.00 | 1.800 | 1.850 | 1.800 | 2.127 |

Cursor Pro wins everywhere, because its VM is not billed. Cursor Teams loses
everywhere: its $0.25-per-million token rate adds 11-18%, against 1.9-2.8% for
Fargate. If cached reads were exempt from that rate (the page does not say),
Teams would lose easy, tie medium and win hard. The lesson's own `dollars`
field is not a token price. It prices the three issues 3.8x, 5.8x and 7.8x
above the token bill. The hard issue's misses come from caps. `run_agent`
stops at 20 turns, not the doc's 30. At 30 turns, the 30-minute cap trips at
turn 21, so resolution only rises from 60.5% to 61.9%.

### 3 — a user who floods 20 issues costs less than anyone else, and the lesson's ledger never starts a new day

**The dashboard shows per-repo per-day cost and per-user cost, with two
alerts.** One is a cost alert (modified z-score above 3.5). The other is a
pressure alert that fires on dispatcher denials. The test is a seeded week
in which a fifth user, eve, files 20 issues on acme/service on day 5.

- **Fresh ledger each day:** 40 PRs over the week. The pressure alert fires
  once, on acme/service day 5 (18 denials). The cost alert flags that day
  ($17.00) and two ordinary days ($15.08 and $13.07).
- **Per-user cost:** eve filed 20 issues and cost $10.87, less than any other
  user ($14.04-39.05). The PR cap turned her away at dispatch, and a denial
  costs nothing. Only the denial count sees her.
- **One ledger for the week:** this is how the lesson's `main` holds it.
  `spent_today` has no day in its key, so PRs by day go 6, 4, 3, 2, 0, 0, 0,
  and 44 of 59 issues are denied.

Two budget guards are hollow. First, at 20 turns a run costs at most $13.28
and takes 29.04 minutes, so the $20 and 30-minute caps can never fire.
Second, `permit` reserves nothing, so 8 issues admitted before any finishes
all pass and spend $65.07 against a $50 cap.

### 4 — skipping CI saves $0 because the lesson's CI is free, and a 3-turn plan costs 32% of a run

**`dry_run` opens a draft PR without CI. The cheap version also caps the
agent at 3 turns.** Mean cost over 2,000 paired issues:

| mode | $ per issue |
|---|---:|
| full run (`dispatch`) | 4.27 |
| dry-run, skip CI only | 4.27 |
| dry-run, plan only (3 turns) | 1.37 |

`run_verify` adds $0 and 0 minutes, so skipping CI saves nothing. What it
changes is what gets opened. Both dry-run variants open a draft for all
2,000 issues, including 79 whose full run went red in CI and 245 that hit
the turn cap. The "no PR without green CI" rule is not in `open_pr`: it
opens a PR with `ci_green=False`, and the rule lives only in `dispatch`'s
if-chain. Recorded the lesson's way (`opened_pr=True`), 5 dry-runs use up the
day's PR cap, and the next real issue is refused.

### 5 — 4 of the 7-day policy's 5 deletions close an open PR, and the lesson's token would let it delete main

**`retention()` deletes 5 of 40 seeded `agent/issue-<n>` branches.** A branch
is deleted when it is unmerged and its last push is more than 7.0 days old;
a branch at exactly 7.0 days is kept. GitHub's own setting only deletes
branches after a merge, so the unmerged case needs its own job.

| variant | branches deleted | open PRs closed |
|---|---:|---:|
| as written, last-push clock | 5 | 4 |
| skip branches with an open PR | 1 | 0 |
| as written, creation clock | 17 | — |

Per GitHub's docs, deleting a branch closes its open PRs, so the rule as
written closes 4 of them unmerged. The lesson's token does not protect
branches. `can` refuses only the exact string `force_push` and actions that
start with `write:main`. It allows `force-push`, `delete:refs/heads/main`,
`write:refs/heads/main` and `write:.github/workflows`, and it never reads its
own `permissions` map. Without the `agent/` prefix, the rule also deletes 3
stale human branches, and `can` approves all 3.
