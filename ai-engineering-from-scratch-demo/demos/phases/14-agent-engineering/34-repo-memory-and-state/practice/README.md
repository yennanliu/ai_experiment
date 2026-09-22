<!-- generated:start -->
# 14-agent-engineering / 34-repo-memory-and-state

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/34-repo-memory-and-state/) · upstream spec
`phases/14-agent-engineering/34-repo-memory-and-state/docs/en.md`

```bash
uv run demo practice run 34-repo-memory-and-state --ex 1
uv run demo explain 34-repo-memory-and-state --ex 1
uv run pytest demos/phases/14-agent-engineering/34-repo-memory-and-state
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a `last_human_touch` timestamp. Refuse any agent write within five seconds of a human edit. | code | T0 | `ex01_the_field_is_refused_before_the_rule_can_read_it.py` |
| 2 | Extend the validator to support `oneOf` so a task can be either a build task or a review task… | code | T0 | `ex02_one_of_is_a_union_and_the_unexpected_field_rule_is_an_intersection.py` |
| 3 | Add a `schema_version` field and write the migration from v1 to v2 (rename `blockers` to `ris… | code | T0 | `ex03_unexpected_fields_fires_before_the_version_is_ever_read.py` |
| 4 | Move the storage backend from a local file to SQLite. Keep the `StateManager` API identical. | code | T0 | `ex04_sqlite_keeps_the_api_and_changes_what_a_write_guarantees.py` |
| 5 | Run two agents against the same state file with a 50 ms write race. What goes wrong and how d… | code | T0 | `ex05_the_rename_prevents_a_torn_file_and_not_a_lost_update.py` |
<!-- generated:end -->

## Answers

### 1 — the field is refused before the rule that reads it can run

The rule is three lines and it cannot run, because `commit` validates before it
writes and `validate` rejects any key not in `properties`. Adding
`last_human_touch` to a state document without touching the schema raises
`unexpected fields ['last_human_touch']`. So the schema change comes first, and
then the 5-second window refuses agent writes at +500ms, +2s and +4.9s and allows
+5s, +7s, +30s, +90s and +3600s — 3 refused of 8 probes.

Three things the exercise's framing does not mention.

The validator cannot express the field's natural type. `_check_type` handles
`object`, `array`, `string`, `integer` and `null` — five of JSON Schema's seven —
so `{"type": "number"}` fails on every value and a float timestamp raises
`expected integer, got float`. Two missing `if` branches decide that the field
must be integer milliseconds. That is a fine choice; it should not be made by an
omission.

Storing the timestamp *in the document the agent overwrites* makes it a
self-clearing guard. `commit` writes the whole state, so an agent that read at
+1s and commits at +6s writes back the `last_human_touch` it read. If a human
edited in between, the stale copy silently reinstates the old timestamp and the
guard allows the write. A sidecar the agent never writes refuses the same race.
The rule's data has to live somewhere the guarded party does not own.

And the guard needs a clock the manager does not have: `StateManager` takes two
constructor arguments, `commit` takes one, and `commit` calls `stat` zero times —
so the comparison time arrives from the caller, and a caller that supplies its own
`now` can supply any `now`. Reading the human's mtime from the filesystem is one
`stat` and makes the refusal non-advisory.

The window itself is arbitrary: 1s refuses 1 probe and 60s refuses 6. Nothing in
the workbench knows how long a human's editor takes to flush, so the number is a
guess until someone measures it.

### 2 — `oneOf` is a union and the unexpected-field rule is an intersection

Adding a `oneOf` branch is fifteen lines: validate against each alternative, count
the matches, refuse unless exactly one. Over six probes — two valid build tasks,
two valid review tasks, one missing a branch-required field, one satisfying both
required sets — it accepts four and refuses two.

The first obstacle is structural. `validate`'s unexpected-fields check compares
the document's keys against *this* schema's `properties`, and a `oneOf` schema has
none of its own. Running the shipped validator against it refuses a perfectly good
build task on `unexpected fields ['acceptance', 'goal', 'id', 'owner', 'status']`
— all five keys. `oneOf` cannot be added as a peer clause beside `type` and
`enum`; the strictness check has to move inside the branch, which restructures the
function rather than extending it.

The second is more interesting and inverts the exercise's premise. The document
carrying both `acceptance` and `reviewer` matches **zero** branches, not two,
because each branch rejects the other's keys as unexpected. So `oneOf` and `anyOf`
give identical answers on every probe — the choice between them, which is the
whole modelling decision the exercise is pointing at, is unobservable. Relax the
unexpected rule and the same document matches two branches, at which point `oneOf`
refuses and `anyOf` accepts. The strictness has quietly turned a union into a
partition, and that is worth knowing before designing around either keyword.

Last: a failed `oneOf` has one reason per branch, and `SchemaError` carries one
string. A naive port reports the last branch's failure and hides the first, which
makes a union schema undebuggable exactly when it matters.

### 3 — `unexpected fields` fires before the version is ever read

`schema_version` ships, so the work is the migration — and the migration cannot
start, because `load` validates before it returns. The refusal is not the version
pin `{"enum": [1]}`, which is what you would expect: it is the unexpected-fields
rule, which runs over the whole key set before `validate` recurses into any
property. Reading a migrated document with the v1 schema raises
`unexpected fields ['risks']`; reading the original with the v2 schema raises
`unexpected fields ['blockers']`. Two cross-version loads, two failures, and
neither mentions the version field that was added to make versioning work.

So version detection has to happen outside the validator. `load` does
`json.loads`, `validate`, `return` — there is no seam. A three-line
`migrate_on_load` that parses, branches on `schema_version`, applies the rename
and *then* validates reads the v1 file and returns a valid v2 document. The
lesson's claim that "the manager refuses to load a file from a version it cannot
migrate" is true; the shipped manager also refuses one it can.

The strict rule is what makes migrations mandatory rather than optional. Because
any unknown key is refused, a v1 reader cannot read a v2 document *at all* —
refused, not degraded. Relaxing that one rule would let old readers ignore `risks`
and keep working, at the cost of the guarantee the lesson wants. One rule decides
whether rollout order matters, which is the kind of consequence worth being
deliberate about.

And a migration is not reversible for free. The rename round-trips byte-identically
here only because `risks` did not exist in v1; a migration that *adds* a field
loses it on the way back, and the document carries one version integer and no
record of which steps ran.

### 4 — SQLite keeps the API and changes what a write guarantees

`SqliteState` exposes `load` and `commit` with the same signatures, validates with
the same `validate`, and round-trips 20 states through a real `sqlite3` connection
with 20 exact matches. The caller in `main` needs no changes. That part is the
easy half and the exercise is right that it should be easy.

What changes underneath is the point. Replaying exercise 5's schedule — both
agents read, both edit, both commit — loses agent A's `touched_files` entry 20 of
20 times on the file backend. The same schedule with a `WHERE revision = ?` guard
refuses the second commit 20 of 20 times, because the row moved. That guard is one
SQL clause and it is not expressible in a rename: `os.replace` makes a write
all-or-nothing and says nothing about what the writer was replacing.

History comes free in one backend and is absent in the other. `atomic_write`
replaces the file, so one version is on disk at any instant; appending each commit
as a row keeps all 41 written during this run, and "what did the agent think at
step 12" becomes a `SELECT`. The file backend can only get this by writing a
second file that nothing validates — which is how audit logs drift from the state
they claim to describe.

The catch is that the API survives and the *failure modes* do not. The shipped
manager raises `FileNotFoundError` on a missing file; the SQLite version returns
`None`. A caller treating the exception as "first run" — which `main` effectively
does by committing before it loads — silently stops working. One of the two error
paths is outside the API the exercise says to keep, and "keep the API identical"
has to include the exceptions or the swap is not a swap.

### 5 — the rename prevents a torn file and not a lost update

Modelling the 50 ms as an interleaving — both agents read, both edit, both commit
— makes the two failure modes separable, which a wall-clock race would not.

**What the rename saves you from:** every read during the race parses and
validates, 20 of 20, and zero are torn. Replacing `atomic_write` with a two-chunk
`write_text` and reading between the chunks produces 8 `JSONDecodeError`s in 20
attempts against `atomic_write`'s 0. That is the whole of what `os.replace` buys,
and it is a *durability* property: the file is never half-written, so a crashed
process cannot leave a state file that is worse than no file at all.

**What it does not save you from:** in 20 of 20 races the second committer's state
overwrites the first's, and agent A's `touched_files` entry is gone from the final
file every time. Atomicity and isolation are different properties; the rename
provides one. The lesson's "a half-written one is worse than no file at all" is
correct and is not the failure a concurrent agent produces.

Nothing in the manager can detect the loss. `StateManager` exposes two methods and
`load` returns the parsed document with no version, mtime or handle, so agent B
has no way to know the file moved under it. A read token — the state's own
revision, checked at commit — turns 20 silent overwrites into 20 detected
conflicts for one extra field, and exercise 4 shows the SQL version of the same
guard.

One durability gap worth fixing while you are in there: `atomic_write` fsyncs the
file descriptor and then renames, which survives a process crash. It does not
fsync the *parent directory*, so a machine losing power between the rename and the
directory flush can come back with neither name pointing at the new data. One
extra `os.fsync` on the directory closes it, and the shipped function has zero.
