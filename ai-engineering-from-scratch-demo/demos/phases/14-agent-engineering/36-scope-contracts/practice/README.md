<!-- generated:start -->
# 14-agent-engineering / 36-scope-contracts

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/36-scope-contracts/) · upstream spec
`phases/14-agent-engineering/36-scope-contracts/docs/en.md`

```bash
uv run demo practice run 36-scope-contracts --ex 1
uv run demo explain 36-scope-contracts --ex 1
uv run pytest demos/phases/14-agent-engineering/36-scope-contracts
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a `network_egress` field listing allowed external hosts. Refuse runs that touch other hosts. | code | T0 | `ex01_the_host_list_is_self_reported_so_an_empty_one_always_passes.py` |
| 2 | Extend the checker to fail soft on `docs/` and hard on `scripts/`. Justify the asymmetry. | explain | T0 | prose, below |
| 3 | Make the contract derive `allowed_files` from a `goal` field using a static rule set (no LLM)… | code | T0 | `ex03_a_static_rule_set_derives_the_wrong_files_on_the_second_goal.py` |
| 4 | Add a `time_budget_minutes` and refuse to continue once the wall clock exceeds it. | code | T0 | `ex04_the_budget_is_checked_once_after_the_work_is_already_done.py` |
| 5 | Run two contracts against the same diff. What is the right merge semantics when both apply? | code | T0 | `ex05_intersecting_allowed_files_can_empty_the_allowlist_silently.py` |
<!-- generated:end -->

## Answers

### 1 — the host list is self-reported, so an empty one always passes

`network_egress` ships on `ScopeContract`, `network_hosts` ships on `RunSummary`,
and `scope_check` already emits a blocking `network.unallowed_host` finding. So
the field exists, and against an allowlist of `['api.anthropic.com']` a run
reporting `['api.anthropic.com', 'evil.example']` produces one blocking finding
naming the bad host.

The same run reporting `[]` passes. The check is guarded by
`and run.network_hosts`, so over four runs — compliant, violating, empty, and a
violating run that simply omits the host — the checker refuses one. That makes
the field a *declaration* check, not an egress control: it verifies that the
hosts an agent admits to are on the list, and an agent that reports nothing is
indistinguishable from an agent that called nothing. Real enforcement needs the
host list to come from a proxy or a sandbox's network log, not from the summary
the run wrote about itself.

Two smaller things worth knowing before writing contracts against it.

`None` and `[]` are different policies and `None` is the default. Omitting the
field permits every host; setting it to `[]` permits none. A task author deleting
the line to "simplify the contract" gets the permissive reading, which is the
wrong direction for a default on a security field.

Hosts are compared with `not in`, so matching is exact. All four near-misses are
blocked — `API.anthropic.com` (case), `api.anthropic.com.evil.test` (suffix
attack), `api.anthropic.com:443` (port) — which is the right direction, and
`eu.api.anthropic.com` is blocked too. Exact matching fails closed and needs an
explicit suffix rule before a team can express "this vendor's endpoints".

### 2 — why docs fail soft and scripts fail hard

**Rollback is part of scope** carries the argument: "listing how to roll back
forces the contract author to think about what could go wrong. A contract you
cannot roll back from is a contract that should not be approved." The asymmetry
between `docs/**` and `scripts/**` is a rollback asymmetry, and the checker's
current shape does not express it.

**What the code actually does today.** `scope_check` classifies a touched path in
order: forbidden → allowed → `docs_paths_soft` → off-scope. A soft hit produces a
`Finding(severity="info")`, and `passed()` is
`not over_budget and not any(f.severity == "block")`. `over_budget` counts *warn*
findings only. So an `info` finding contributes to nothing: a run that writes a
hundred markdown files passes, identically to a run that writes none. "Fail soft"
is implemented as "do not fail", which is a different thing and the distinction
matters when someone later asks why nobody noticed the docs drift.

**Why the asymmetry is nonetheless right.** Three reasons, in order of weight.

*Rollback cost.* An unwanted `docs/**` edit is reverted by `git checkout` and
costs nothing else — the artifact is the file. An unwanted `scripts/**` edit may
already have *run*: a release script, a migration runner, a deploy hook. Reverting
the file does not revert its effects. The contract's `rollback_plan` field is
answerable for one and not the other, which is exactly the test the doc proposes.

*Blast radius outside the repo.* Docs are read by humans, who apply judgement.
Scripts are read by CI, which does not. A wrong sentence in `docs/api.md` is
caught by the next reader; a wrong line in `scripts/release.sh` is caught by
production.

*Base rate and signal value.* Agents touch docs constantly and usually
harmlessly — exercise 3 measured a keyword rule set deriving `docs/**` for 2 of
12 goals that did not ask for it. If a docs write blocked, the gate would fire so
often it would be disabled, and the lesson's `violation_budget` exists precisely
to keep noisy-but-cheap violations from halting a run. Scripts are touched rarely,
so a hard failure there has a high prior of being real.

**What "fail soft" should mean, concretely.** Not `info`. Three changes make the
asymmetry real:

1. Soft violations become `warn`, so they count against `violation_budget` — one
   stray docs edit is free, five is a pattern and stops the run. That reuses the
   budget mechanism the lesson already built instead of adding a fourth severity.
2. Soft violations are recorded in the archive even when the run passes, so
   "docs drift" is a trend someone can query rather than a line nobody read.
3. `scripts/**` moves into `forbidden_files` rather than relying on absence from
   `allowed_files`. Today an unlisted `scripts/foo.sh` lands in `off_scope` at
   `warn`, and it is only the explicit `scripts/release.sh` entry that blocks —
   so the hard half of the asymmetry currently covers one literal path, not the
   directory. Exercise 3's glob finding applies here: `scripts/**` under `fnmatch`
   does match nested paths, but `scripts/*` would too, for the wrong reason.

**The caveat.** This asymmetry is a property of *this* repo's layout. A
documentation site where `docs/**` is the deployed artifact inverts it entirely —
there, docs are production and `scripts/**` may be developer-local. The rule to
carry is not "docs soft, scripts hard" but "soft where rollback is a revert, hard
where rollback is an incident", which is a question the contract author has to
answer per repo and which `rollback_plan` is the place to answer it.

### 3 — the rule set is right 7 times in 12, and wrong in three different ways

A 14-keyword table mapping goal words to path globs derives the correct allowlist
for 7 of 12 goals. The 5 failures are not one bug:

- **empty (2)** — "tidy up the release process" and "rename the worker pool" fire
  no rule at all
- **union (2)** — "update the signup docs and the handler" fires two rules and
  derives 4 globs where 2 were wanted
- **wrong (1)** — "document the schema migration" derives `migrations/**` and
  `lib/models.py` alongside `docs/**`, i.e. write access to the thing being
  documented

**The empty derivation is the dangerous one**, and it is the answer to "what goes
wrong on the first edge case". A goal matching no rule yields `allowed_files ==
[]`, and `scope_check` then reports every touched path as off-scope — 3 of 3 for
an otherwise clean run. The contract cannot distinguish "nothing is allowed" from
"nobody knew what to allow", so an unrecognised goal is indistinguishable from a
maximally strict contract. A derivation that cannot fail loudly should refuse to
produce a contract at all; the two-line fix is to raise rather than return `[]`.

**The union is the common one, and the merge hides it.** Two rules firing derives
4 globs; merging against a project contract intersects it back down to
`['app.py', 'test_app.py']`, losing `docs/**`. So the over-derivation never
appears in the effective contract — it appears as a task that cannot touch what
its own goal described, one layer away from the cause. That is the worst place for
a defect to surface.

And the derived globs inherit a problem the shipped ones already have.
`matches_any` uses `fnmatch`, where `*` crosses directory separators: 3 of 6
probes behave against the glob's plain reading — `config/*.yaml` matches
`config/prod/secrets.yaml`, `lib/**/*.py` *misses* `lib/top.py` (it requires two
separators), and `**/*.md` misses `README.md`. The lesson's "globs, not raw paths"
advice is right and needs `pathlib.PurePath.full_match` or `glob`-style semantics
to mean what a reader expects.

### 4 — the budget is checked once, after the work is already done

`time_budget_minutes` ships and `scope_check` emits a blocking
`time.over_budget` finding. Over a six-point sweep it fires twice — at 30.1 and
42.1 minutes — and not at 29.9 or 30.0, because the comparison is strict and the
budget itself is allowed.

What is missing is the verb. "Refuse to *continue*" needs a check during the run,
and `scope_check` runs once, on a `RunSummary`, after everything has happened. All
six firings are verdicts on runs that already finished: zero of them stop
anything. Checking after each of 20 steps stops a breaching run at step 14,
saving 6 steps — 30% of the run — and that gate cannot reuse `scope_check`,
because it needs an elapsed time at a point where `RunSummary` does not exist yet.
It is a different function with a different input, which is why "add a field" does
not finish this exercise.

Two structural notes. `elapsed_minutes` defaults to `0.0` and is supplied by
whoever builds the summary, so a run reporting zero passes any budget — the same
self-reporting problem exercise 1 found in `network_hosts`. Of `RunSummary`'s
four fields, only `touched_files` could be checked against the filesystem; the
other three are claims. And the merged budget takes the minimum without recording
its source, so a run at 45 minutes reads `elapsed 45.0m > budget 30m` whether the
30 came from the task or from a project-wide policy the task author never saw.

### 5 — least privilege on five fields, and the intersection is the one that bites

`merge_contracts` states its semantics in its own docstring: intersect allowed,
union forbidden, narrowest budgets, accumulate approvals. Merging the lesson's
project-wide and task contracts gives `['app.py', 'test_app.py']`, 3 forbidden
patterns, a 30-minute budget, violation budget 0 and egress
`['api.anthropic.com']`. The clean run passes; the creep run yields 5 findings, 4
blocking. **Least privilege is the right default**, and it is right on four of the
five fields.

The exception is `allowed_files`. An intersection can be empty, and an empty
allowlist is indistinguishable from a contract that forbids everything. Merging a
task that needs `docs/api.md` with a project that never listed it gives a 2-entry
allowlist, not 3 — the path is dropped with no finding, no warning and no field
recording the drop. The run then reports that path as an off-scope write, so the
*contract's* gap surfaces as the *agent's* violation. The fix is one field: a
`dropped_by_merge` list on the effective contract, so the difference between "the
task exceeded its scope" and "the merge removed what the task was granted" is
visible without diffing two JSON files.

Two more observations about the semantics.

`network_egress` uses `None` for "no enforcement" and `[]` for "deny all", so one
sentinel carries two meanings. Folding `(None, allow, deny)` gives `[]` either
way, and two `None` parents leave enforcement off entirely — correct in each case,
and it means the fold's result depends on whether any contract in the chain
opted in at all. A three-state field (`unset` / `deny` / `allow[...]`) removes the
ambiguity; a bare `None` invites a merge that quietly disables a control.

`violation_budget` takes the minimum, so a permissive parent cannot grant slack.
The project allows one off-scope write and the task zero, so the merge is zero —
which is least privilege and means a project-wide budget is unreachable from any
task that does not restate it. Of the merged fields, four can only tighten and
one, `acceptance_criteria`, only grows. That is the correct asymmetry for a
safety contract, and it is worth stating explicitly because it makes project-level
*allowances* pointless: only project-level *restrictions* survive a merge.
