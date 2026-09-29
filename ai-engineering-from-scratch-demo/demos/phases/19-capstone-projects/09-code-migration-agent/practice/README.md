<!-- generated:start -->
# 19-capstone-projects / 09-code-migration-agent

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/09-code-migration-agent/) · upstream spec
`phases/19-capstone-projects/09-code-migration-agent/docs/en.md`

```bash
uv run demo practice run 09-code-migration-agent --ex 1
uv run demo explain 09-code-migration-agent --ex 1
uv run pytest demos/phases/19-capstone-projects/09-code-migration-agent
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run the migrate pipeline with OpenRewrite only (no agent). Compare pass rate to the full pipe… | code | T0 | `ex01_recipes_alone_pass_11_of_50_repos_the_agent_alone_19_more_and_turning_it_off_rescues_4.py` |
| 2 | Implement a "lint-clean" check: after migration, run a style linter (spotless for Java, ruff… | code | T0 | `ex02_a_lint_gate_blocks_12_of_16_coverage_preserved_prs_and_every_new_error_comes_from_two_recipes.py` |
| 3 | Add a "minimal-diff" optimizer: after the agent's branch passes tests, trim unnecessary chang… | code | T0 | `ex03_trimming_cuts_the_agents_diff_34_to_60pct_and_every_cut_past_the_noise_reverts_an_untested_fix.py` |
| 4 | Extend to a third migration: Node 18 to Node 22. Reuse the sandbox wrapping; swap the recipe… | code | T0 | `ex04_the_codemod_clears_346_of_401_node_22_hazards_and_no_recipes_at_all_changes_0_of_50_outcomes.py` |
| 5 | Measure time-to-first-green-build (TTFGB) as a UX metric. Target: p50 under 10 minutes. | code | T0 | `ex05_p50_ttfgb_is_8_minutes_over_green_repos_and_26_over_all_50_and_6pct_of_passes_overrun_the_cap.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`: a 50-repo simulation in
which `run_recipes` returns a rewrite count, `migrate` passes a repo
straight after the recipes when one draw lands under
`0.55 * (1 - hardness)`, and otherwise `agent_loop` rolls a per-turn pass
chance under a 30-minute, $8, 20-turn budget. No source code exists in the
simulation, so exercises 2-4 pair its repos with small generated source
files. `main()` (seed 19) passes 30 of 50. The TypeScript dashboard under
`code/ts/` is not run.

### 1 — recipes alone pass 11 of 50 repos, the agent alone 19 more, and turning it off rescues 4

**OpenRewrite only passes 11 of 50 (22%); the full pipeline passes 30
(60%).** The agent alone is the difference on 19 repos, after 1 to 8
turns. These repos are not a harder class. Whether a repo passes on
recipes alone is one coin flip weighted by hardness, capped at 52.25% for
the easiest repo, so the agent-alone cases are the repos whose flip
missed.

| | recipe-only | agent-alone |
|---|---:|---:|
| seed 19 repos | 11 | 19 |
| hardness range | 0.220-0.735 | 0.219-0.823 |
| pass rate over 1,000 seeds | 19.45% | +37.16 points (56.61% full) |

**The naive A/B is broken by the shared rng.** Rerunning seed 19 with the
agent switched off passes 15 repos, and 4 of them failed with the agent
on. The agent loop consumes draws, so the rerun compares different coin
flips, not the same repo with and without an agent.

**The doc's "70-80% of migrations" is 19.45% in the code**, and the
failure class is drawn at random: 1,186 of the 3,013 `custom_annotation`
repos over 1,000 seeds are Python, the same 40% share Python has in the
bench.

### 2 — a lint gate blocks 12 of 16 coverage-preserved PRs, and every new error comes from two recipes

**12 of the 16 passes with no coverage loss regress on style (75%).** Each
simulated repo is paired with a generated Python 2 module, migrated by
regex recipes, and linted before and after with two of ruff's default
rules, F401 and E711 (a stdlib stand-in, because ruff cannot parse Python
2). The gate blocks 23 of the 30 green PRs. All 50 migrated files compile.
The rate depends on the fixture's snippet mix; the cause does not.

**All new errors come from two recipes.** `<>` to `!=` produces all 26
E711 errors (`result != None`). Rewriting `string.join` and `string.upper`
into str methods leaves `import string` unused (14 F401). Adding a
`<> None` to `is not None` rule still leaves 14 blocked PRs, all F401.

**The lesson's coverage gate lets losses through.** 14 of the 30 passes
end below base coverage and still count as passed; the code fails a repo
only below base minus 2.0 points. The doc says "dropped more than 2%".
Over 1,000 seeds, 55 passes dropped more than 2% of base and passed, and
only 8 repos were ever filed as `coverage_regression`.

### 3 — trimming cuts the agent's diff 34-60%, and every cut past the noise reverts an untested fix

**The diff shrinks from 214 changed lines to 142 (-33.6%) when lines are
reverted one at a time, and to 85 (-60.3%) when every subset of one
function's changed lines is tried.** The fixture is the 19 repos the agent
passed on seed 19. Each has five generated functions, four of them tested
(the lesson's 80% base coverage), holding 95 needed fixes and 119 noise
lines. The one-line pass keeps 57 noise lines: a renamed local spans three
lines, and reverting any one of them alone breaks the function.

**Both passes revert fixes the tests never run.** 10 needed fixes in the
untested function (5 `has_key`, 3 `iteritems`, 2 `xrange`) are reverted,
and in 10 of 19 repos that function then crashes, while the tested
functions still pass.

**On Python 3.14 the result depends on the interpreter.** PEP 758 lets the
Python 2 `except ValueError, err:` compile (as `except (ValueError, err):`),
so the trimmer reverts 7 more fixes and 17 of 19 repos break. The line is
a SyntaxError on 3.11-3.13 (measured on 3.12).

| | one line at a time | per function |
|---|---:|---:|
| changed lines kept (3.12 / 3.14) | 142 / 135 | 85 / 78 |
| noise lines kept | 57 | 0 |
| needed fixes reverted (3.12 / 3.14) | 10 / 17 | 10 / 17 |

### 4 — the codemod clears 346 of 401 Node 22 hazards, and no recipes at all changes 0 of 50 outcomes

**Five regex rules clear 346 of the 401 hazards (86.3%) in 50 generated
Node 18 files.** The rules cover import assertions (`assert { type:
'json' }` is a SyntaxError on Node 22; checked on node v22.23.2), `util.isArray`
and `util.isDate` (DEP0044/DEP0047, runtime-deprecated in v22.0.0),
`new Buffer()` (DEP0005) and `require('punycode')` (DEP0040), per the
[Node.js deprecations page](https://nodejs.org/api/deprecations.html)
(read 2026-09-29). The 55 left are `new Buffer(payload)`: from the text
alone it could mean `Buffer.from` or `Buffer.alloc`, so it goes to the
agent. 19 of the 50 files come out clean.

**The swap changes nothing.** The codemod replaces `run_recipes` inside the
lesson's `migrate`. With the codemod, the lesson's recipes or a layer that
rewrites nothing, all 50 repos get the same status, failure class, turns
and cost (30 pass). `migrate` stores the rewrite count and never reads it.
With no recipes, 9 of the 11 repos that pass without an agent turn still
contain an import assertion Node 22 cannot parse.

### 5 — p50 TTFGB is 8 minutes over green repos and 26 over all 50, and 6% of passes overrun the cap

**The 10-minute target holds only if the repos that never go green are
left out.** TTFGB is `wall_min` when `migrate` returns a pass (the
simulation has no separate build step), and infinite for a repo that
never passes.

| | seed 19 | seeds under 10 min (of 1,000) |
|---|---:|---:|
| p50 over green repos (30) | 8.0 min | 747 |
| p50 over all 50 | 26.0 min | 3 |

**The budget leaks and two of its three caps never bind.** `agent_loop`
checks the clock before a turn, so 1,691 of 28,304 passes (6.0%) go green
after minute 30 (33.9 minutes on seed 19). No repo takes more than 11
turns or spends more than $7.4725, so the $8 and 20-turn caps never fire.
The agent path also starts its clock at zero: all 3,221 one-turn agent
passes are faster than the median recipe-only pass (4.99 minutes).
