<!-- generated:start -->
# 13-tools-and-protocols / 27-skill-evals-packaging-and-portability

Solutions to all 8 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability/) · upstream spec
`phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability/docs/en.md`

```bash
uv run demo practice run 27-skill-evals-packaging-and-portability --ex 1
uv run demo explain 27-skill-evals-packaging-and-portability --ex 1
uv run pytest demos/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Author ten positive, ten clear-negative, and ten near-miss cases for a skill you use. Split t… | code | T0 | `ex01_the_clear_negatives_are_free_and_they_are_a_third_of_the_score.py` |
| 2 | Run a five-run baseline and treatment comparison. Report every per-task regression even if th… | code | T0 | `ex02_the_shipped_router_discards_the_run_index.py` |
| 3 | Add a rubric dimension that requires human judgment. Calibrate it on five examples before usi… | code | T0 | `ex03_a_three_point_rubric_reaches_the_gate_as_one_boolean.py` |
| 4 | Add one host capability and define supported, adapted, degraded, and unsupported outcomes. | code | T0 | `ex04_adapter_required_is_two_outcomes_and_the_gate_counts_neither.py` |
| 5 | Modify an installed reference after manifest creation. Prove the package verification fails b… | code | T0 | `ex05_the_digest_covers_the_bytes_and_nothing_about_the_file.py` |
| 6 | Create a skill whose body passes lint but whose script violates its artifact contract. Identi… | code | T0 | `ex06_lint_reads_the_prose_and_the_artifact_layer_reads_a_fixture.py` |
| 7 | Add an upgrade eval that compares invocation policy and required capabilities between two pac… | code | T0 | `ex07_nothing_in_the_module_knows_what_a_version_is.py` |
| 8 | Publish a compatibility report that names tested host versions, dates, fallbacks, and unverif… | code | T0 | `ex08_the_badge_ban_is_a_phrase_ban.py` |
<!-- generated:end -->

## Answers

All eight are T0 and stdlib, and all eight ship code.

`run_release_gate` has eleven checks across five layers, and the recurring
finding is that **the layers are graded on what someone typed into them**.
The artifact layer scores a string the caller supplies. The script and safety
layers are `EvidenceCheck` booleans with an evidence field nothing reads. The
five-run baseline calls a router that discards the run index. Each layer is
correctly built and each is one substitution away from grading a fixture
instead of a package.

The parts that do hold are the ones that read bytes: `lint_package` walks the
tree, `build_manifest` hashes every file, `verify_manifest` compares sets,
and the gate refuses to verify an install by verifying the source. Those are
also the only checks in the lesson that cannot be satisfied by writing
something down.

### 1 — the clear negatives are free, and they are a third of the score

**ANSWER: 30 cases in three frozen groups, scored apart.** **10/10**,
**10/10**, **6/10** — precision **0.7143**, recall **1.0**, accuracy
**0.8667**. Pooled it reads like a strong router; split, it is a router with
one failure mode.

**FINDING: the clear negatives are free.** Dropping them takes accuracy from
**0.8667** to **0.80** and removes **0** false positives. A third of the set
moves the number and cannot move the design.

**FINDING: one false-positive counter for two different failures.**
`TriggerCase` has **3** fields and no group, so the split survives only as an
id convention.

**FINDING: tuning after looking is what the ordering rule forbids.**
Threshold 3 fixes all **4** near misses and breaks **2** positives — recall
**1.0 → 0.8** for precision **0.7143 → 1.0**.

### 2 — the shipped router discards the run index

**ANSWER: the average improves and three cases regress.** Mean **0.82 →
0.92**; `near-trigger` **-0.4**, `near-package` **-0.2**, `pos-manifest`
**-0.2**.

**FINDING: the shipped router discards the run index.** `del run_index` on
line one, so five runs are five copies of one and the only rates it can
produce are **0.0** and **1.0**. The flakiness had to be injected to be
studied.

**FINDING: five runs gives six possible rates.** The instrument's resolution
is **0.2**, and **2** of the **3** regressions are exactly one run wide.

**FINDING: the pooled metrics hide what the per-case rates show.** Precision
and recall *both* rise while three cases get worse.

### 3 — a three-point rubric reaches the gate as one boolean

**ANSWER: score 0–2, calibrate on five, gate only after the boundary rule.**
Agreement **4/5**, and the disagreement is the boundary case — the only kind
a gate ever decides. One written sentence takes it to **5/5**.

**FINDING: a three-point rubric reaches the gate as one boolean.** The
threshold moves out of the gate and into whoever fills in `passed`.

**FINDING: the evidence string is required and never read.** Evidence `"x"`
validates and passes; only all-whitespace is rejected.

**FINDING: the gate is `all(passed)`, so there is no room for a mean.**
Rubrics averaging **1.0** and **1.6** arrive as **2** and **3** booleans.

### 4 — adapter-required is two outcomes, and the gate counts neither

**ANSWER: one capability, four outcomes, and the rule is whether the gap can
be emulated.** Companion files can be inlined; script execution and tool
enforcement cannot.

**FINDING: `adapter-required` is two outcomes, and the new gap reads as
`native`.** A capability the matrix does not ask about cannot be missing —
adding the field is what makes the gap exist.

**FINDING: capabilities are hard-coded fields and extensions are data.**
Adding an extension is a string; adding a capability edits two dataclasses
and the matrix function.

**FINDING: the gate counts native hosts, so a degraded host is invisible.**
`ReleaseThresholds` has **4** fields and none counts one.

### 5 — the digest covers the bytes and nothing about the file

**ANSWER: the installed tree fails and the source passes, from one
manifest.** Three tamper shapes, three distinct fields: `mismatched`,
`unexpected`, `missing`.

**FINDING: the digest covers the bytes and nothing about the file.** Drop a
script's executable bit and verification returns `passed=True`. Mode,
ownership and timestamps are outside what a manifest can say.

**FINDING: the manifest cannot list itself, and the module makes that
explicit.** `build_manifest` skips the reserved path and `verify_manifest`
rejects a manifest that mentions it — solved by refusing to pretend.

**FINDING: the gate verifies two trees and refuses when they are one.**
Caught by path identity, not by content.

### 6 — lint reads the prose and the artifact layer reads a fixture

**ANSWER: the artifact-contract layer blocks it, and only with the real
output.** Lint is clean; `with_skill_artifact` and `artifact_improvement`
are the **2** failing checks of **11**.

**FINDING: lint reads the prose and never the code.** The only thing it
reads inside a script is a credential regex — a script contradicting the
`## Output contract` section above it is invisible.

**FINDING: the artifact layer grades a string the caller supplies.**
`artifact_mode` defaults to `"fixture"` and nothing runs the script.

**FINDING: `artifact_improvement` is a second question, not a second
opinion.** It earns its place in the other direction: a baseline that
already passes fails it alone.

### 7 — nothing in the module knows what a version is

**ANSWER: five deltas — three widening, one narrowing, one neutral.**
Widening is the review gate; narrowing is the migration note.

**FINDING: nothing in the module knows what a version is.** **0** of **10**
dataclasses carries one, so the upgrade eval is a second harness that runs
the gate twice.

**FINDING: narrowing is a break the portability matrix reports as an
improvement.** Dropping an extension moves a host to `native` while the
skill has silently stopped asking for it.

**FINDING: the manifest cannot say what changed.** A rename plus an edit and
three genuine edits look the same.

### 8 — the badge ban is a phrase ban

**ANSWER: a report passing a contract that requires four sections and bans
five badge phrases.** Versions, dates, fallbacks and unverified behaviours
all present; `portable` appears **0** times.

**FINDING: the badge ban is a phrase ban.** The lesson's own
`("guaranteed portable",)` misses `fully portable`, `portable` and
`portability-verified`. Banning a claim means enumerating its spellings.

**FINDING: three of the four required facts are not in the matrix.**
`HostCapabilities` carries a bare `name` — **0** version or date fields.

**FINDING: "not verified" is the complement of a set nothing enumerates.**
`missing` lists what is known absent; nothing lists what was never tested,
which makes it the section most likely to be quietly dropped.
