<!-- generated:start -->
# 14-agent-engineering / 35-initialization-scripts

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/35-initialization-scripts/) · upstream spec
`phases/14-agent-engineering/35-initialization-scripts/docs/en.md`

```bash
uv run demo practice run 35-initialization-scripts --ex 1
uv run demo explain 35-initialization-scripts --ex 1
uv run pytest demos/phases/14-agent-engineering/35-initialization-scripts
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a probe that diffs the current commit against the last-known-good commit and refuses to s… | code | T0 | `ex01_the_probe_ships_and_a_fresh_lock_skips_it_entirely.py` |
| 2 | Wire the script to write a `prereqs.lock` file and refuse to start if the lock is older than… | code | T0 | `ex02_the_ttl_is_one_day_the_exercise_says_seven.py` |
| 3 | Add a `--fix` flag that auto-installs missing dev dependencies but never modifies runtime dep… | code | T0 | `ex03_the_script_cannot_tell_a_dev_dependency_from_a_runtime_one.py` |
| 4 | Move probes from hardcoded functions to a YAML registry. Defend the trade-off. | explain | T0 | prose, below |
| 5 | Add a timing budget per probe. A probe that runs longer than three seconds is a workbench smell. | code | T0 | `ex05_the_budget_downgrades_a_pass_to_a_warn_and_warn_never_blocks.py` |
<!-- generated:end -->

## Answers

### 1 — the probe ships, and a fresh lock skips it entirely

`probe_lkg_diff` is already in `run_probes` with `LKG_FILE_DIFF_BUDGET = 50`, so
the work is finding out when it runs and what it counts. Against a real
temporary git repository it returns `pass` at 50 changed files and `fail` at 51,
naming the baseline's short sha — exactly the boundary the constant puts it at.

Three things make it weaker than it reads.

**A fresh `prereqs.lock` skips it.** `main` calls `lock_is_fresh()` and returns 0
before `run_probes` is ever reached, so a workbench with a lock written an hour
ago launches on a tree 500 files from its baseline without computing the diff.
The cache the lesson compares to a Docker layer sits *in front of* the safety
check rather than behind it. That is the wrong order for anything that refuses to
start: cache the expensive probes, never the gate.

**The budget counts files, not change.** 51 one-line edits fail; one file with
5000 changed lines passes. `git diff --name-only` discards the magnitude, so the
probe measures how *spread out* the drift is. That is a defensible proxy — a
change touching many files is harder to review — but it is not what "more than 50
files changed" implies to someone reading the exercise, and a `--numstat` sum is
the same call with one more flag.

**The diff runs in the wrong directory.** `cwd=HERE` points at the lesson's own
`code/` folder, so the comparison is against whatever repository the workbench's
*source* lives in, not the project the agent is editing. Where those differ — which
is the normal case once the init script is vendored into a real repo — the number
has no relationship to the agent's blast radius.

### 2 — the TTL is one day; the exercise says seven

The lock ships, `write_lock` ships, `lock_is_fresh` ships. `LOCK_TTL_SECONDS` is
86400 where the exercise asks for 604800, so locks aged 25, 72 and 167 hours flip
from stale to fresh when the constant is corrected. That is one line.

The second half of the exercise's sentence is a different change. "Refuse to
start if the lock is older than seven days" is not what the shipped stale path
does: it falls through to `run_probes()` and can still exit 0. Re-probing on a
stale lock is arguably the better behaviour, and it is not the behaviour the
exercise describes — worth deciding deliberately rather than inheriting.

Three findings about the lock itself.

**The fingerprint is blind to the repository.** `_deps_fingerprint` hashes four
module literals — `REQUIRED_DEPS`, `REQUIRED_TEST_COMMAND`, `REQUIRED_ENV_VARS`,
`REQUIRED_PYTHON`. Editing a source file leaves it byte-identical; editing the
init script's own constants moves it. So the lock is invalidated by changing the
checker and by nothing that happens in the project, which is backwards from what
a cache key usually wants.

**A corrupt lock is silently treated as absent.** Three malformed shapes — not
JSON, not a dict, an uncoercible `written_at` — all return `False` with zero
diagnostics. The direction is safe, and a lock that has been wrong for a month is
indistinguishable from a first run, so nobody finds out.

**A cache hit leaves the report stale.** `main` checks the lock before it writes
`REPORT_PATH`, and `write_lock` runs only after every probe passes. So a hit exits
0 *without writing a report*, and `init_report.json` keeps describing a run up to
24 hours old while the console says the workbench is fine. Anything reading the
report as current is reading a cached verdict without knowing it.

### 3 — the script cannot tell a dev dependency from a runtime one

The flag is easy; the classification is the exercise. `REQUIRED_DEPS` is a flat
list of two module names with no dev/runtime marker, so a `--fix` written against
the shipped data has exactly one behaviour available — install everything — and
the "never without approval" half is unimplementable until the registry carries
the distinction. The split has to be added to the *data* before it can be honoured
in the code.

With the split in place, a fixture of five missing dependencies gives: three
installed, two refused with `approval required`, exit 1. Without `--fix`: zero
installed, exit 1, which is the shipped behaviour. Note that `--fix` still exits
non-zero — a repair that leaves runtime dependencies unapproved has not fixed the
workbench, and returning 0 would be the tempting mistake.

Two consequences worth designing for.

**`--fix` breaks the lock's meaning and the fingerprint cannot see it.**
`_deps_fingerprint` hashes the *declared* lists rather than what is installed, so
a fix that installs three packages leaves the hash unchanged and the lock written
before the repair is still "fresh" afterwards. A repair that changes the
environment has to invalidate the cache explicitly, because the content hash is
computed from the wrong content.

**The evidence the fix needs is formatted into a sentence.**
`probe_dependencies` computes `missing` and returns a `Probe` whose `detail` is
`f"missing: {missing}"`. `Probe` has four fields and none holds a list, so `--fix`
either re-derives the set with `find_spec` or parses one English sentence back
into one list. One structured field removes the round trip. (`_timed` also does
not use `functools.wraps`, so the probes lose their names — `probe_dependencies`
introspects as `_wrap`, which makes any registry keyed on `__name__` quietly
wrong.)

### 4 — a YAML registry, and what it costs

**Fail loud, fail fast, fail in one place** is the section the trade-off has to be
argued against: "a probe failure means halt and surface to the human… the whole
point of init is to refuse to start when the workbench is broken."

**What moving to YAML buys.** Three things, in descending order of value.

*Probes become data, so they can be diffed and reviewed like rules.* Lesson 33
made this argument for instructions and it holds here: a probe added in a YAML
file shows up in a PR as three lines a reviewer can read, where a probe added as a
decorated function shows up as a function a reviewer has to run. The registry also
makes the probe *set* visible — exercise 5's finding that only 1 of the 6 shipped
probes can ever return `fail` is obvious in a table and invisible in a module.

*Per-probe configuration stops being a module constant.* Today
`PROBE_BUDGET_SECONDS`, `LKG_FILE_DIFF_BUDGET` and `STATE_FRESHNESS_SECONDS` are
globals shared by every probe, so the 3-second budget applies equally to a string
comparison and a subprocess. A registry gives each entry its own budget and
severity, which is the fix exercise 5 actually needs.

*Different repos can carry different probe sets without forking the script.* This
is the reason it usually gets proposed, and it is the weakest of the three — one
`probes.yaml` per repo and one shared runner is a real gain only once there is
more than one repo.

**What it costs, and why the doc's heading is the right place to argue it.**

*"Fail in one place" gets harder, not easier.* Today a probe failure is a Python
exception or a `Probe(status="fail")` in one file, and the stack trace points at
the code. With a registry there are two failure surfaces: the probe failed, or the
*registry entry* was wrong — a typo'd check name, a missing key, a budget that
will not parse. Lesson 33 measured exactly this: `score` reports `passed=False`
both when a check fails and when the check does not exist, and a malformed rule is
dropped silently so the run it would have blocked goes green. A YAML probe
registry inherits both bugs unless the loader validates against a schema and
distinguishes `error` from `fail` — which is the work, and it is more work than
the registry itself.

*YAML is a dependency and a parser.* The init script is currently stdlib-only,
which matters because it runs *before* the dependency probe. A registry parsed
with PyYAML cannot check whether PyYAML is installed. Either the parser is vendored
(this repo's own `harness/yamlite.py` is the stdlib-subset answer) or the
bootstrap has a hole in exactly the place the script exists to cover.

*Probes are code, and most of them want to stay code.* `probe_lkg_diff` is 25
lines of subprocess handling, error branches and sha validation. In a registry it
becomes `check: lkg_diff` plus parameters — the logic stays in Python and the
registry holds a name and a number. So the honest version of the migration is
*configuration* in YAML and *implementations* in Python, which is a smaller change
than "move probes to YAML" sounds and is the only version that does not make
failure harder to localise.

**The verdict.** Move the registry, not the probes: a YAML file listing probe
names, severities, budgets and parameters, with implementations staying as Python
functions looked up by name — and a loader that validates the file against a
schema and reports an unknown probe name as `error`, distinctly from `fail`. That
keeps "fail in one place" (the runner), gains the reviewability and per-probe
configuration, and avoids the bootstrap hole. Skip it entirely while there is one
repo and six probes; the registry pays for itself at the point where probe sets
diverge between projects.

### 5 — the budget downgrades a pass to a warn, and `warn` never blocks

`_timed` ships, `PROBE_BUDGET_SECONDS` is 3.0, and the decorator already rewrites
a slow `pass` into a `warn` with the duration appended. So the budget exists, and
what it does is nothing: `main` computes `ok = all(p.status != "fail")`, so `warn`
is indistinguishable from `pass` at the exit code. Probes at 250ms, 4000ms and
8000ms come back `pass`, `warn`, `warn` and the run's verdict is `True` in all
three cases. A smell is reported and never acted on, which is the thing the
lesson's "fail loud" section says not to do.

Three sharper problems underneath.

**The budget is measured after the probe returns.** `_timed` calls `probe_fn()`
and *then* compares the elapsed time, so a probe that hangs for an hour is
reported after an hour. A budget that cannot interrupt is a metric. Exactly one of
the six shipped probes carries a real bound — the `timeout=2.0` on
`probe_lkg_diff`'s subprocess — and 2.0s is *below* the 3.0s budget, so the only
boundable probe can never trip it. The two numbers were chosen independently and
one makes the other unreachable.

**The downgrade only applies to a passing probe.** The condition is
`status == "pass"`, so a probe that is slow *and* failing keeps `fail` and loses
the timing note entirely. A slow failure and a fast failure read identically,
which is the pair a workbench most wants to distinguish — one is a broken check,
the other is a broken environment.

**Only 1 of the 6 probes can return `fail` at all.** `REQUIRED_DEPS` is
`['json', 'dataclasses']`, both stdlib, so the dependency probe cannot fail;
`REQUIRED_ENV_VARS` is empty, so the env probe cannot fail; the runtime probe
needs Python below 3.10; `probe_test_command` looks for `python3`, which resolved
in order to start the script; and `probe_state_freshness` returns `warn` by
construction. The gate's entire real coverage is the LKG diff — which exercise 1
showed is skipped whenever the lock is fresh. Fixing the timing budget is worth
less than making the other five probes capable of failing.
