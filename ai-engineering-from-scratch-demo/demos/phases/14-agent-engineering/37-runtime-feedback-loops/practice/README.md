<!-- generated:start -->
# 14-agent-engineering / 37-runtime-feedback-loops

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/37-runtime-feedback-loops/) · upstream spec
`phases/14-agent-engineering/37-runtime-feedback-loops/docs/en.md`

```bash
uv run demo practice run 37-runtime-feedback-loops --ex 1
uv run demo explain 37-runtime-feedback-loops --ex 1
uv run pytest demos/phases/14-agent-engineering/37-runtime-feedback-loops
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a `cwd` field per record so the same command run from different directories is distinguis… | code | T0 | `ex01_the_runner_never_passes_a_cwd_so_there_is_none_to_record.py` |
| 2 | Add a `redaction` step that strips lines matching `^Bearer ` or `password=`. Test on a fixtur… | code | T0 | `ex02_truncating_before_redacting_drops_the_begin_and_keeps_the_key.py` |
| 3 | Cap total `feedback_record.jsonl` size at 1 MB by rotating to `.1`, `.2` files. Defend the ro… | code | T0 | `ex03_the_one_megabyte_cap_is_really_six_and_checked_before_the_write.py` |
| 4 | Add a `parent_command_id` so retry chains are visible: which command produced the input that… | code | T0 | `ex04_the_chain_is_a_linked_list_and_rotation_can_cut_it.py` |
| 5 | Pipe the JSONL into a tiny TUI that highlights the latest non-zero exit. Eight key features t… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — the runner never passes a `cwd`, so there is nothing to record

Adding the field is one line, and it would record a constant.
`run_with_feedback` calls `subprocess.run(command, capture_output=True,
text=True, timeout=timeout_s)` with no `cwd=`, so every command inherits the
caller's working directory. "The same command run from different directories" is
not a state this runner can produce: running `os.getcwd()` from two temporary
directories gives one distinct output. Forwarding `cwd` gives two, with `command`
byte-identical and only the new field differing — which is exactly the
discrimination the exercise asks for.

Why it matters: `FeedbackRecord` has twelve fields and none is a path, so two
records for `['pytest', '-x']` in different packages are distinguishable only by
`command_id`, which is random. Grouping a day's records by `command` — the obvious
way to find a flaky test — merges directories that were never the same test.

Two notes for the implementation. The runner takes four arguments and forwards
exactly one to `subprocess.run`; `agent_note`, `timeout_s` and
`parent_command_id` are bookkeeping. So `cwd`, `env` and `stdin` are all
unreachable from the call site and each is the same one-line change in the same
place — worth doing together. And the field must carry the value *passed to*
`subprocess.run`, not `os.getcwd()` sampled at record construction: on a runner
that does forward `cwd`, the two disagree on every run.

### 2 — truncating before redacting drops the BEGIN and keeps the key

Redaction ships, with five patterns covering rather more than the two the
exercise names, and all five replace their canonical shapes. The fixture is where
the work is, and it exposes an ordering bug that `_process_capture` documents in
its own docstring: *"Truncate first, then redact."*

That is safe for a single-line secret — a token on a dropped line is simply gone —
and unsafe for a multi-line one. A PEM block whose `-----BEGIN-----` line falls
in the truncated middle keeps **9 lines of key material and the `-----END-----`
marker** in the tail, unredacted, because the anchor the pattern needs was deleted
before the pattern ran. Redacting the untruncated text first and tailing second
keeps zero. One pass over a larger string, and the failure disappears.

Two more things the fixture shows.

The redaction counter undercounts by whatever was truncated away. Output with 200
Bearer tokens, 165 of them in the dropped middle, records
`redactions={'stdout': 35}`. So the number in the record means "secrets in the
kept excerpt", not "secrets the command emitted" — and a reviewer reading 35 has
no way to learn the real figure. If the count is meant as a leak signal it has to
be computed before truncation even if the redaction is not.

And the two patterns the exercise actually names have the loosest anchors.
`bearer\s+` requires whitespace, so `Bearer:ya29.AbCdEf` is untouched; the
assignment pattern requires `[:=]`, so `password is hunter2` is untouched. Both
survive as plaintext in a record reporting zero redactions. Pattern-based
redaction is a floor, not a guarantee, which is the same conclusion Lesson 21
reached about injection markers and Lesson 24 about PII — and the right response
is the same: do not treat "0 redactions" as evidence of anything.

### 3 — the one-megabyte cap is really six, and it is checked before the write

`maybe_rotate` ships with `ROTATE_BYTES` at 1 MB and `MAX_ROTATIONS` at 5, so the
mechanism exists and the *total* is not what the exercise asked for. Appending
until rotation cycles leaves `feedback_record.jsonl` plus five numbered siblings —
5 MB on this run, ceiling 6 MB, against an exercise that says cap the total at
1 MB. The rotation ordering is correct (oldest dropped, `.1` newest) and
`load_all` reads every file, so lineage survives it.

Two properties that read as choices and are not.

The check runs *before* the append. `maybe_rotate` is called, returns early
because the file is under the cap, and then a record is written — so a file at
1048575 bytes grows past `ROTATE_BYTES` and is only rotated on the *next* call.
The cap is a floor: the file is rotated once it is already too big, never before.
For a 1 MB cap and ~1 KB records that overshoot is invisible; for a cap sized to
a memory budget it is the difference between fitting and not.

And truncation, not rotation, is what actually bounds a record.
`deterministic_tail` keeps 5 head and 30 tail lines, so a capture is 35 lines
maximum and a single record cannot approach a megabyte. `ROTATE_BYTES` only
decides how many records fit. Remove the truncation — which someone will propose
the first time a reviewer wants full output — and the rotation policy stops
bounding anything.

**Defending the policy.** Size-based rotation is right for the stated goal: the
lesson wants loader memory bounded, and bytes are what memory is measured in. It
is wrong for the question reviewers actually ask. The six files carry no
timestamps in their names and `started_at` lives inside the records, so "what
happened on Tuesday" means opening every file and parsing every line. Daily
rotation bounds the *question* and not the memory; size rotation bounds the
memory and not the question. A defensible compromise is size-triggered rotation
with a timestamped name (`feedback_record.2026-09-22T14.jsonl`), which keeps the
memory property and makes the directory listing answer the common question —
at the cost of an unbounded file count, which is what `MAX_ROTATIONS` exists to
prevent. Pick based on whether the log is read by the next turn (size) or by a
human next week (time); the lesson's "feedback versus telemetry" section says
these are different files precisely because they have different answers.

### 4 — the chain is a linked list, and rotation can cut it

`parent_command_id` ships, `retry_chain` walks it, and `load_all` deliberately
reads the rotated files "so parent-command lineage survives rotation". That claim
holds, and it has a bound: four linked records reconstruct oldest-to-newest, and
deleting the file holding the root — what the sixth rotation does — returns three
of four.

The failure is silent by construction. `while cursor and cursor in records` stops
both when `parent_command_id` is `None` and when the id is absent, so a truncated
chain and a complete one are the same shape. The caller sees a three-link chain
and cannot tell whether the first command was the origin or the first survivor.
One boolean — "the last link had a parent we could not find" — distinguishes them
and costs nothing, and it matters because the whole point of the chain is to
answer "what started this".

Two more observations.

`retry_chain` calls `load_all()` on every invocation, so reconstructing 20 chains
reads the six rotation files 20 times: 120 file reads for a question that needs
six. The lesson's memory argument is about the size of one load; the cost here is
repetition, and hoisting the load to the caller is a one-line change.

And the lineage records retries, not data flow. The exercise asks which command
produced the *input the next command consumed*, and `parent_command_id` is a
caller-supplied argument — `main` passes `parent_command_id=fail.command_id` by
hand, and `retry_chain` never reads `stdout_tail`. Nothing checks that the child
actually consumed the parent's output, and one of the lesson's five records
carries a parent at all. The field is an honest *assertion* by the agent, in the
same category as `agent_note`, and should be read that way rather than as
derived provenance.

### 5 — eight things the TUI has to show

**What goes in a feedback record** lists the seven fields and why each matters:
`command` (exact argv), `stdout_tail`, `stderr_tail`, `exit_code`, `duration_ms`,
`started_at`, `agent_note`. A review TUI is a *view* over those fields plus the
three the shipped record adds — `command_id` / `parent_command_id`, `error`, and
the `truncations` / `redactions` counters — and it earns its place only if it
answers questions a `tail -f` cannot.

The eight features, in the order a reviewer needs them:

**1. The latest non-zero exit, pinned and highlighted.** This is the exercise's
own requirement and the reason the TUI exists. Not "errors are red" — *the most
recent failure is always on screen*, even when 200 successes have scrolled past
it. A run that fails at step 14 and then runs 40 more commands has one line worth
reading and 40 lines of noise.

**2. `exit_code` rendered as three states, not two.** The lesson's "refuse to
advance without feedback" section is explicit: a runner that errored before
capturing exit writes `exit_code: null` with an `error`. `null` is not `0` and
not failure — it is *no signal*, and it is the only state where the loop must
stop. A TUI that colours non-zero red and everything else green makes the one
state that should halt the agent look like success.

**3. The retry chain, collapsed to one row.** `parent_command_id` exists so a
retry is not read as an independent event. Four attempts at the same thing should
occupy one line that expands, showing "4 attempts, 3 failed, 42s total" — and,
per exercise 4, showing whether the chain's root was reachable, because a
silently truncated chain reads as a short one.

**4. `duration_ms` as a distribution, not a number.** The lesson's own
justification is "surfaces slow probes and runaway processes", and a single
duration cannot surface either. What matters is *this command against its own
history*: `pytest` at 4s when its median is 1.2s is the signal. Colour on
deviation, not on absolute value.

**5. The truncation marker made obvious.** `truncations: {"stdout": 165}` means
the reviewer is looking at 35 lines of a 200-line output. A TUI that renders the
tail without saying so invites conclusions drawn from the visible part — and
exercise 2 showed the invisible part can contain secrets the redactor never saw.

**6. `agent_note` beside the result, always.** This is the field with no
mechanical source: one line the agent wrote about what it *expected*. The review
question is almost always "did this do what it meant to", and the note next to
the exit code answers it in one glance. It is also the cheapest lie detector in
the record — a note saying "expect hello" beside an exit of 2 is the whole
finding.

**7. Filter by `command`, grouped by `cwd`.** The most common review action is
"show me every `pytest` run". Exercise 1 is why `cwd` has to be in the grouping
key: without it, the same argv in two packages collapses into one misleading
series.

**8. Jump to the raw JSONL line.** Everything above is lossy. The TUI must be able
to hand the reviewer the underlying record — file, line number, full JSON —
because the moment a summary is wrong, the argument moves to the source. This is
also what keeps the TUI honest: it is a view, and the record stays the artifact.

**What it must not show.** No live tail of `stdout` — that is the terminal's job
and it competes with feature 1. No aggregate success rate: over a feedback file
that rotates at 1 MB, the denominator is "however many records survived
rotation", which exercise 3 showed is not a number anyone chose.
