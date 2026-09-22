<!-- generated:start -->
# 14-agent-engineering / 38-verification-gates

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/38-verification-gates/) · upstream spec
`phases/14-agent-engineering/38-verification-gates/docs/en.md`

```bash
uv run demo practice run 38-verification-gates --ex 1
uv run demo explain 38-verification-gates --ex 1
uv run pytest demos/phases/14-agent-engineering/38-verification-gates
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a `coverage_floor` check: the test command must produce a coverage report with at least 8… | code | T0 | `ex01_a_missing_coverage_report_is_a_warning_and_passes_the_gate.py` |
| 2 | Support a `--strict` mode that promotes every `warn` to `block`. Document the cases where str… | code | T0 | `ex02_strict_promotes_everything_including_the_finding_that_means_no_data.py` |
| 3 | Make the gate produce a Markdown summary in addition to JSON. Defend which fields belong in t… | code | T0 | `ex03_the_json_has_a_field_the_summary_must_not_repeat.py` |
| 4 | Add a `time_since_last_human_touch` check: any file edited within 60 seconds of a human keyst… | code | T0 | `ex04_the_exemption_needs_a_timestamp_the_scope_report_does_not_carry.py` |
| 5 | Run the gate on a real agent diff from your product. How many findings are real and how many… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — a missing coverage report is a warning, and warnings pass the gate

The floor lives in three places at once, and none of them is the artifact.
`COVERAGE_FLOOR_DEFAULT = 0.80` is a module constant, `verify(..., coverage_floor=)`
is a parameter, and `coverage_report` supplies the two numbers the check compares.
A task at 62% blocks at the default; passing `--floor 0.60` clears
`coverage.below_floor` and the task **still blocks**, because
`coverage.regression` compares `previous` against `current` and 0.80 → 0.62 is an
18-point drop no floor argument touches. The report that passes under a 60% floor
is the one that omits `previous` entirely — which is to say the floor and the
regression check read the same artifact and answer to different owners.

The hole is one level up. `coverage_report` defaults to `None`, and a `None` report
yields one `coverage.missing` finding at `warn` severity, so `passed` stays `True`.
The floor is unenforceable in exactly the case where it matters most: the run where
the test command crashed before writing `coverage.json`. An agent that deletes the
failing test and an agent whose coverage tool fell over produce the same verdict.

Two smaller asymmetries: the regression comparison guards both branches with
`math.isclose`, so a 0.85 → 0.84 drop lands in `coverage.minor_regression` (warn)
rather than on the boundary; the floor comparison is a bare `current < floor`, so
`0.7999999999999999` blocks and `0.80` passes with no tolerance at all. And
`VerdictReport` records `coverage` but not the floor that was applied, so a reader
cannot tell a green report at 80% from a green report at 60%.

### 2 — `--strict` promotes everything, including the finding that means "no data"

Of the 8 codes the gate emits, 3 are born `warn`: `coverage.minor_regression`,
`coverage.missing` and `scope.off_scope`. The other 5 — `acceptance.missing`,
`acceptance.failed`, `feedback.null_exit`, `scope.forbidden`, `rule.failed` — are
already blocking, so strict cannot touch them. Run the lesson's own three demo
tasks both ways and the verdicts go `[True, False, False]` to
`[False, False, False]`: one task flips, and it is the only one strict *can* flip,
the one that was passing with a warning attached.

**Where strict is the right default:** release branches, post-incident triage, and
any diff that will be merged without a human reading it. The argument is not that
warnings become important on a release branch; it is that nobody is reading them
there, and an unread warning is an unenforced rule.

**Where it is wrong** is visible in a single code. `coverage.missing` fires when
`coverage_report` is `None`, with the detail `"no coverage_report.json; cannot
enforce floor"`. Under strict, a task whose coverage tool crashed blocks on that
finding alone — the gate promoting its own ignorance to a verdict. A task that
failed the floor and a task that produced no data should not exit the same way,
and exercise 1 is why: absence of data is the agent's most convenient outcome.

Two structural notes. Promotion is all-or-nothing: `verify` rewrites every warn in
one comprehension, so of the 8 subsets of the three warn codes a boolean flag
reaches 2. A team that wants "off-scope blocks, missing coverage warns" cannot say
so — per-code severity, the shape Lesson 33 gave rules, costs one dict. And
`VerdictReport` carries `strict` but not what strict changed: promoted findings
arrive as `severity="block"` with no memory of having been warnings, so a reader
cannot separate a genuine block from a promoted one without re-running the gate.

### 3 — the JSON has a field the summary must not repeat

The summary carries 4 of `VerdictReport`'s 6 fields — `passed`, `task_id`,
`head_commit`, and the blocking findings — and renders a failing task in 10 lines
against the JSON's 42, while still naming all 5 of its blocking findings. That is
the whole defence: a human reads a summary to decide whether to look at the JSON,
so it needs the verdict, the thing the verdict is about, the commit it was computed
over, and every finding that caused it. Nothing else.

Three specifics. `passed` is computed *after* strict promotion, so the summary has
to render `(strict)` explicitly; a reader who re-runs the gate non-strict and sees
green would otherwise conclude the summary lied. The task's 6 findings span 6 codes
and 2 severities in source order, so grouping by severity — not truncating — is
what puts the 5 blocks above the 1 warning. And `coverage` belongs in the summary
because it is two numbers answering a decision question, while the feedback log
does not: it is 35 captured lines per command, it lives on `Artifacts` rather than
`VerdictReport`, and reproducing it in the summary would make the summary the
artifact instead of the pointer to it.

### 4 — the exemption needs a timestamp the scope report does not carry

Give the check the data it wants and it works: five off-scope paths edited 5, 45,
61, 600 and 3600 seconds after the last human keystroke leave 2 exempt
(`README.md`, `docs/api.md`) and 3 flagged. The shipped gate exempts **0** of them,
and not because the rule is wrong — because `scope_report["off_scope_writes"]` is a
list of strings. There are no timestamps to compare, so the gate emits one
`scope.off_scope` finding listing all five paths at once.

Where the timestamps go is forced by the types. `Artifacts` has 7 fields and
`scope_report` is typed `dict[str, object]`, so the per-path mtimes arrive inside
it — and both call sites in `_scope_findings` change with it, because a list of
strings and a list of `{path, mtime}` records do not read the same way.
`Finding` carries `code`, `severity` and `detail` with nowhere to record
"considered and excused", so an exempted path either disappears from the report or
needs a fourth field; disappearing is the wrong half of that choice.

One design note the exercise invites: the window has to be one-sided. Exempting
files touched within 60 seconds *after* the last keystroke exempts 2 of 5;
a symmetric ±60s window exempts 4, including a file the agent touched a full
minute before any human was at the keyboard. The case the rule exists to cover is a
human editing a file the agent is already in — which is a human timestamp *followed
by* an agent write, not the reverse.

### 5 — running the gate on a real agent diff

I ran it on one: commit `8a91709` of this repository, the agent-authored diff that
added the Lesson 37 practice solutions. Contract: `allowed_files` =
`demos/phases/14-agent-engineering/37-runtime-feedback-loops/practice/**`,
`forbidden_files` = `scripts/**` and `harness/**`. Acceptance commands: the lesson's
`pytest` run and `scripts/audit_practice.py`, both executed for real and their exit
codes fed in as feedback records.

**8 files changed, 2 findings, 0 of them real.**

- `scope.off_scope: ['README.md']` — the repository's top-level README, regenerated
  by `scripts/coverage.py` as a mechanical byproduct of finishing a lesson. It is
  outside the contract's globs and it is *supposed* to be. This is noise produced by
  a contract that cannot express "files this repo's own tooling regenerates."
- `coverage.missing` — there is no `coverage_report.json` in this repo at all, so
  the gate reports it every single run, on every task, forever.

Both are warnings, so the verdict is `passed: True` and the noise costs nothing
today. Under `--strict` the same commit fails on both, and that is the honest test
of the signal-to-noise ratio: a gate whose strict mode is unusable on a clean diff
is a gate that will be run non-strict, which means its warnings are decoration.

**Where it needs to grow.** Four things, in the order the measurements found them.

1. **A `generated_files` list in the contract**, distinct from `allowed_files`.
   Paths a build step owns are neither in scope nor a violation; they are outside
   the agent's intent. Today the only way to silence the README finding is to widen
   the globs, which also grants the agent write access to it.
2. **`coverage.missing` must know whether coverage was ever configured.** A project
   with no coverage tool and a project whose coverage tool crashed are different
   facts and the gate reports them identically. The floor's owner — per exercise 1,
   the contract — should also say whether a floor applies at all.
3. **Command comparison is string equality on a shape the upstream lesson does not
   produce.** `_acceptance_findings` does `str(rec.get("command"))` and compares it
   against `acceptance_commands`. A Lesson 37 feedback record carries argv as a
   list, so `["pytest", "-q"]` stringifies to `"['pytest', '-q']"` and never matches
   `"pytest -q"` — the gate reports `acceptance.missing: never ran: pytest -q` for a
   command that ran and exited 0. Two lessons that are supposed to compose hand each
   other a shape mismatch, and the failure mode is a false block.
4. **The override log is written and never read.** `record_override` appends a
   signed entry to `overrides.jsonl` and `verify_signature` validates it, but
   `verify()` never opens `OVERRIDES_PATH` — a signed, valid override changes no
   verdict. The doc's **Refuse without exception** rule is exactly right that a
   block can only be lifted by a human with a recorded reason and user id; the gate
   currently implements the refusal and not the exception, so the recorded override
   is an audit trail for a decision the gate never acted on. Wiring it in is where
   the hard part starts: an override has to be scoped to a `head_commit`, or it
   silently lifts the same finding on every future commit.
