<!-- generated:start -->
# 13-tools-and-protocols / 24-skill-discovery-and-progressive-disclosure

Solutions to all 6 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/24-skill-discovery-and-progressive-disclosure/) · upstream spec
`phases/13-tools-and-protocols/24-skill-discovery-and-progressive-disclosure/docs/en.md`

```bash
uv run demo practice run 24-skill-discovery-and-progressive-disclosure --ex 1
uv run demo explain 24-skill-discovery-and-progressive-disclosure --ex 1
uv run pytest demos/phases/13-tools-and-protocols/24-skill-discovery-and-progressive-disclosure
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a plugin scope and place it between user and built-in precedence. Prove the collision res… | code | T0 | `ex01_the_scope_name_is_the_identity_so_two_plugins_are_fatal.py` |
| 2 | Change the collision policy from highest precedence to qualified names. Preserve both entries… | code | T0 | `ex02_a_qualified_name_is_not_a_name_the_loader_can_find.py` |
| 3 | Add a byte-size limit to `load_reference`. Test a file exactly at the limit and one byte abov… | code | T0 | `ex03_the_file_is_read_in_full_before_the_limit_is_consulted.py` |
| 4 | Create two descriptions that sound nearly identical. Rewrite them so the trigger boundaries d… | code | T0 | `ex04_the_clause_that_disambiguates_is_the_one_the_budget_cuts.py` |
| 5 | Add a manifest containing hashes for every reference and script. Detect a modified resource b… | code | T0 | `ex05_a_manifest_that_does_not_cover_itself_covers_nothing.py` |
| 6 | Instrument the demo to report Level 1, Level 2, and Level 3 byte counts separately. | code | T0 | `ex06_level_one_scales_with_the_catalog_and_the_others_with_the_task.py` |
<!-- generated:end -->

## Answers

All six are T0 and stdlib, and all six ship code.

Five of the six change one policy and then look for what else moved. That is
the shape of this lesson: discovery is a pipeline of small decisions —
precedence, collision policy, size limits, description text, path
containment, budget — and each one is consumed somewhere the exercise does
not name. A qualified name breaks a loader 90 lines away. A byte limit
exposes that the read already happened. A description rewrite is undone by
the catalog budget that publishes it.

The recurring finding is the **unit mismatch**: characters against bytes,
per-call against per-run, what the author validates against what the model
reads, what the diagnostics record against what the catalog publishes. Every
one of those pairs is spelled the same way and measures something different.

### 1 — the scope name is the identity, so two plugins are fatal

**ANSWER: plugin sits third, and dropping the winning scope each time walks
the order.** `project → user → plugin → builtin`, with **1** collision
recorded over **4** candidates. Dropping the winner repeatedly pins plugin
from both sides; one collision would only have pinned it from one.

**FINDING: adding a scope is two edits, and forgetting the second is a hard
error.** `resolve_collisions` knows scopes only through the caller's
precedence tuple, so an unlisted scope raises `CollisionError` instead of
sinking to the bottom. Fail-closed — and it means the order cannot be
inferred from the roots.

**FINDING: the scope *name* is the identity.** Two `Scope("plugin", …)` over
different roots tie, and equal precedence is fatal. A host with two plugin
sources must name them apart *before* discovery.

**FINDING: the shadowing survives in diagnostics and is invisible to the
model.** `CatalogEntry` has **4** fields and no `selected` flag — unlike the
lesson's own JSON example — and `model_dict()` publishes entries only.

### 2 — a qualified name is not a name the loader can find

**ANSWER: every candidate survives as `scope::name`, and the catalog grows.**
**3** entries and **0** collisions against **2** and **1**; **403**
characters against **254**, **59%** more for one duplicate.

**FINDING: a qualified name is not a name the loader can find.**
`_candidate_for` matches name *and* directory, so the qualified entry raises
`KeyError` and Levels 2 and 3 stop working. Four lines in the catalog builder
break a function that never mentions scopes.

**FINDING: a qualified name is not a portable skill name either.**
`NAME_PATTERN` rejects `project::evidence-report`. The qualifier belongs in a
second field — which is what `scope` already is.

**FINDING: the ambiguity is not removed, it is moved.** Zero collisions is
the worse signal: the diagnostics go quiet and the model gets two names
differing by a prefix nobody explained to it.

### 3 — the file is read in full before the limit is consulted

**ANSWER: exactly at the limit passes, one byte over raises, and `stat()`
decides.** **1024** bytes loads; **1025** raises having returned **0**. Both
pass the shipped character check, because `max_chars` was never the same
question.

**FINDING: the file is read in full before the limit is consulted.** A
**524288**-byte file is decoded into memory before being refused. `st_size`
answers before anything opens — the only reason a byte limit is worth adding
rather than tightening the character one.

**FINDING: characters and bytes are different limits, and the module uses
both.** `FRONTMATTER_LIMIT` counts encoded bytes; `load_reference` counts
characters. **1024** CJK characters are **3072** bytes.

**FINDING: the character limit is not a bound on memory.**
`max_chars=12_000` admits a 48KB read, already allocated by the time the
check runs. A byte limit bounds the read; a character limit bounds the
result.

### 4 — the clause that disambiguates is the one the budget cuts

**ANSWER: the rewrite removes every tie, by adding words.** Trigger overlap
**3 → 0**, ties **6 → 0**, and the descriptions get **84** characters
*longer*.

**FINDING: disambiguation is conditions, not brevity.** Both originals say
what the skill produces and neither says when it applies. The **5** terms
still shared after the rewrite are the nouns both skills genuinely own —
deleting those separates nothing.

**FINDING: the clause that disambiguates is the one the budget cuts.** Both
rewritten sentences share their first **59** characters, so at a
**60**-character budget the catalog publishes two identical entries and all
six queries tie again. The rewrite is intact on disk and gone from the
model's view.

**FINDING: the author is validated against 1024 and the model reads 240.**
**784** characters vanish behind an ellipsis and nothing reports it.

### 5 — a manifest that does not cover itself covers nothing

**ANSWER: a modified reference is refused and an untouched one loads.**
**3** resources covered; **46** characters returned clean, **0** returned
after an edit, against the shipped loader's unconditional **41**.

**FINDING: the content is in the process before the verdict exists.**
Hashing consumes the whole file. The gate protects the model's context, not
the host's memory — a size check can precede the read and an integrity check
cannot.

**FINDING: a manifest that does not cover itself covers nothing.** Edit the
resource *and* its manifest entry and it verifies clean. The digest has to
live where the package cannot write it, or the manifest only catches
accidents.

**FINDING: a per-path lookup cannot see an added file, and the body is
uncovered.** A new `references/extra.md` passes every listed hash;
only a set comparison reports it. And `SKILL.md` is out of scope by
construction — Level 3 sealed, Level 2 open.

### 6 — Level 1 scales with the catalog and the others with the task

**ANSWER: three counts, in bytes, reported apart and per resource.** **459**
/ **28** / **80** across **2** resources at 3 skills installed; Level 1 is
**81%** of the run.

**FINDING: Level 1 scales with the catalog and the others with the task.**
At **50** skills, Levels 2 and 3 are byte-identical and Level 1 reaches
**99%**. Shortening descriptions moves the first number and only the first —
two budgets, not two halves of one.

**FINDING: the per-call cap is not a budget.** 12000 characters per file,
nothing accumulated: **2** references cost **80** bytes against **46** for
one and no code path notices. A Level 3 budget has to live in the caller.

**FINDING: the demo reports characters, and the smaller of the two catalog
numbers.** `catalog_chars` against `report_chars`, both carrying the
absolute install path — so Level 1 moves when the tree does, **684** bytes
here against **459** with the path folded. In the other direction **240**
CJK characters are **720** bytes.
