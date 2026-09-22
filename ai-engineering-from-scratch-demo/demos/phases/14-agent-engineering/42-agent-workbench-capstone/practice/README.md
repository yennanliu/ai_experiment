<!-- generated:start -->
# 14-agent-engineering / 42-agent-workbench-capstone

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/42-agent-workbench-capstone/) · upstream spec
`phases/14-agent-engineering/42-agent-workbench-capstone/docs/en.md`

```bash
uv run demo practice run 42-agent-workbench-capstone --ex 1
uv run demo explain 42-agent-workbench-capstone --ex 1
uv run pytest demos/phases/14-agent-engineering/42-agent-workbench-capstone
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Decide which optional fifth doc deserves promotion into the canonical pack. Defend the cut. | code | T0 | `ex01_the_schema_that_ships_without_a_doc_is_the_one_the_policy_leans_on.py` |
| 2 | Rewrite the installer as Python with a `--dry-run` flag. Compare ergonomics against bash. | code | T0 | `ex02_an_unknown_flag_is_a_real_install_and_the_guard_is_one_file.py` |
| 3 | Add a `bin/uninstall.sh` that safely removes the pack and refuses if state files have non-tri… | code | T0 | `ex03_non_trivial_means_the_repo_changed_it_and_the_lock_records_no_filenames.py` |
| 4 | Add a `lint_pack.py` that fails when the pack drifts from `VERSION`. Wire it into CI for the… | code | T0 | `ex04_the_version_is_a_constant_in_the_generator_so_drift_is_the_default.py` |
| 5 | Author the migration runbook from a hand-rolled workbench to this pack. What is the order of… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — the schema that ships without a doc is the one the policy leans on

The cut is decidable rather than tasteful: a doc earns promotion when something in
the pack depends on a human writing a file the pack never explains.

**The fifth doc is a scope-contract authoring guide.**
`scope_contract.schema.json` ships with 7 properties, 6 of them required, and none
of the four docs mentions it by name — where `agent_state` is at least named in
`docs/agent-rules.md`. The gate reads a scope report, the reliability policy leans
on the scope check as its answer to failure mode 2, and nothing tells a human how
to write the contract both depend on.

The measurements behind that, in order of how much they bothered me:

**2 of the 5 failure modes the policy claims to absorb have a script behind them.**
`run_with_feedback.py` covers cascading errors; `verify_agent.py` covers
hallucinated action. Scope creep, context loss and tool misuse name a scope
checker, a state writer and a reviewer that the pack's 4 scripts do not include.
The policy is a promise the directory does not keep.

**The gate reads three paths the pack never writes.** `verify_agent.py` loads
`outputs/scope/closed/<task>.json`, `outputs/scope/closed/<task>.report.json` and
`outputs/rule_report.json`, each through a loader that returns a default when the
file is missing. A fresh install therefore verifies 0 acceptance commands against
0 rules and reports `passed: true` — a green gate that has checked nothing.

**`AGENTS.md` opens with two files the installer never creates.** It names 6 paths
to read before acting. The installer lays down the 4 docs and leaves
`agent_state.json` and `task_board.json` absent, which `init_agent.py` then reports
as the warn-severity "no state file yet". A day-one agent reads a router pointing
at two files that do not exist.

A scope-contract guide does not fix all of that — a scope checker script would fix
more — but it is the smallest addition that makes the rest legible, and it is the
one the exercise asks for. Second place: a "clean state" checklist from Lesson 40,
which is real but has no schema depending on it.

### 2 — an unknown flag is a real install, and the guard is one file

The comparison is not about syntax. `--dry-run` has to answer "what exactly would
change", which means the installer needs the file list before it writes anything.
Whether that is natural is what separates the two.

**`bash bin/install.sh --dry-run` performs a real install.** `FORCE="${1:-}"`
compares argument one against `--force` and ignores every other value. On a clean
target the dry run writes 13 files and prints "pack installed at version 1.0.0".
Any flag the script does not know is a silent yes. The Python rewrite plans the
same 13 paths, prints them, and writes 0 — not because Python is better at strings,
but because the plan exists as a list before it exists as an effect.

Three more things the bash version cannot do without becoming the Python one:

**The guard is one file.** The refusal checks `$TARGET/AGENTS.md` and nothing else.
Delete that single file from a fully installed repo and the installer exits 0,
overwriting 12 others without `--force`. A planned install compares the whole
planned set against disk and reports the 12 it would overwrite.

**15 files ship, 13 install.** `README.md`, `VERSION` and `bin/install.sh` stay
behind, and `.workbench-version` is written in their place. Nothing in the target
carries the pack's own README, so the version lock is the only trace of where the
files came from.

**Re-running never removes what the pack dropped.** `cp -r` unions; it does not
sync. Install a pack carrying an extra doc, then re-install the pack without it,
and the target holds 14 files where a fresh install lays down 13. The orphan is
invisible to every subsequent install, and it is indistinguishable from a file the
team added on purpose.

Ergonomics verdict: bash wins on "no interpreter assumptions" and loses on
everything that needs a decision before a side effect. Since the pack already
depends on Python 3 for all four of its scripts, the bash installer's one advantage
does not apply here.

### 3 — non-trivial means the repo changed it, and the lock records no filenames

"Non-trivial" has to be decidable from the repository, not from intent. Three
signals are available in a git checkout: the path is dirty in the working tree, its
history carries commits beyond the install, or its bytes no longer match what the
pack laid down. The third subsumes most of the first two and needs no history walk.

Installed into a real temp git repo, committed, then used for a session: the
uninstall removes **11 of the 12** files the pack wrote and refuses on the
customised `docs/agent-rules.md`. `agent_state.json` never enters the conversation,
because the pack never wrote it — which is the cleanest form of the answer. *What
counts as non-trivial* is the wrong question to ask about state; the right one is
*what did the pack install*, and everything else is the user's by default.

**`.workbench-version` records a version and zero filenames.** So an uninstaller
has to re-derive the list from a pack source it may no longer have, and if the pack
moved on, the derived list is the new pack's rather than the installed one's. A
12-line manifest written at install time removes the guesswork entirely, and it is
the same artifact `lint_pack.py` wants in Exercise 4.

**Without hashes, a customised pack file looks untouched.** Removing by name
deletes an edited `docs/agent-rules.md` silently. Comparing bytes against the pack
catches it. The rules doc is the file a team is most likely to edit — sharpening
rules to the team's history is what the lesson's own "Ship It" section recommends —
which makes it the worst possible file to delete by name.

### 4 — the version is a constant in the generator, so drift is the default

"Drifts from `VERSION`" needs a number behind it. The workable definition is a
content digest: hash every shipped file, record the digest beside the version, fail
when the digest moves and the version does not.

`PACK_VERSION = "1.0.0"` is a module constant and `VERSION` is written straight from
it, so nothing in the pipeline connects a content change to a version bump. Add one
line to `docs/agent-rules.md` and the digest over the 15 shipped files changes while
`VERSION` reads `1.0.0` before and after. `lint_pack.py` is the only thing in the
pack that can notice, which is exactly why the exercise wires it into CI.

Three consequences worth building into the lint:

**The version travels into targets; the digest does not.** `install.sh` copies
`VERSION` into `.workbench-version` and nothing else, so two installs from packs
with different contents and the same version leave byte-identical locks. The lint
must run in the pack's own repo — a target repo has no way to tell them apart. (It
also means the lock should carry the digest, which is a one-line change.)

**Doc-only is the majority case.** 7 of the 15 files are docs or schemas against 4
scripts, so most edits are patch bumps under the lesson's rule. A digest alone
reports *that* the pack moved, never whether the move needed a major. The useful
lint is per-directory: a change under `schemas/` or `scripts/` demands a major or
minor; a change under `docs/` accepts a patch.

**Hash the generator, not just the outputs.** Two runs produce byte-identical
packs, so an output digest is stable — but the outputs are regenerated from string
constants in `main.py`, so a commit that edits a script body and the version
together is invisible to an output-only digest taken afterwards. Folding the
generator source into the hash is what makes the check total.

### 5 — the migration runbook from a hand-rolled workbench

The order below is chosen so that every step before the last is additive. The
agent keeps working throughout; nothing it depends on is deleted until the pack has
demonstrably replaced it.

**0. Inventory and hash what you have.** List the hand-rolled files and record a
digest for each. This is not ceremony: Exercise 2 showed the installer overwrites
`docs/`, `schemas/` and `scripts/` wholesale as soon as `AGENTS.md` is absent, so
the inventory is what lets you tell "the pack replaced this" from "the pack ate
this". Exercise 3's uninstaller needs the same list, from the other direction.

**1. Install onto a branch, never the working tree.** The installer writes 13 files
and refuses only on `AGENTS.md`. On a branch the whole change is one diff a
reviewer can read, and the fallback is `git checkout main` rather than a restore
from memory.

**2. Reconcile the rules doc first, and only the rules doc.** It is the file your
team already has an opinion about, and the one Exercise 3 found most likely to
diverge. Merge your existing rules into `docs/agent-rules.md` in the pack's format
— category, `check:`, prose — before touching anything else, because every
downstream surface reads rule slugs.

**3. Migrate state to the schema before running `init_agent.py`.**
`agent_state.schema.json` requires `schema_version`, `active_task_id`,
`touched_files` and `next_action`. A hand-rolled state file almost certainly has
three of the four under different names. Do this while the old workbench is still
running, so a bad mapping costs a re-edit rather than a stalled session.

**4. Fill the board and set acceptance commands.** Until `task_board.json` exists
with real `acceptance` entries, the gate verifies nothing and reports
`passed: true` (Exercise 1). A green gate that has checked nothing is worse than no
gate, because it is the one people quote.

**5. Run the gate on the last task you already closed.** You know the answer, so a
disagreement is a mapping bug rather than a code problem. This is the step that
converts "installed" into "working", and it is the last reversible one.

**6. Now delete the hand-rolled scripts.** One commit, referencing the inventory
from step 0, after the pack has produced a verdict you recognise.

**Downtime is confined to step 3**, and only for the minutes the state file is
being rewritten. Everything else runs alongside the old workbench, which is the
whole reason for this ordering: a migration that starts by deleting is a migration
that has to finish in one sitting.

Two version notes, because the pack's **Versioning** rule is what makes a
repeat migration cheap. First, the target records which pack version it was
installed against in `.workbench-version`, and the state file should record the
same number — the migration is not done until the two agree, and a major bump later
means a state migration rather than a re-copy. Second, per Exercise 4 the lock
should carry the content digest beside the version, or "we are on 1.0.0" stays true
across packs that share nothing but a number.
