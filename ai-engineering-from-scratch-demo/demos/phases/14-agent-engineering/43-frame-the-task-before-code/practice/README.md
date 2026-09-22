<!-- generated:start -->
# 14-agent-engineering / 43-frame-the-task-before-code

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/43-frame-the-task-before-code/) · upstream spec
`phases/14-agent-engineering/43-frame-the-task-before-code/docs/en.md`

```bash
uv run demo practice run 43-frame-the-task-before-code --ex 1
uv run demo explain 43-frame-the-task-before-code --ex 1
uv run pytest demos/phases/14-agent-engineering/43-frame-the-task-before-code
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Frame a real bug from one of your repositories without proposing a solution. | code | T0 | `ex01_a_frame_whose_receipts_all_resolve_and_a_validator_that_never_checks.py` |
| 2 | Find one claim in the frame that is actually an assumption. Replace it with evidence. | code | T0 | `ex02_the_shipped_example_has_two_facts_and_no_resolvable_receipt.py` |
| 3 | Add a human unknown whose answer would change the public contract. | code | T0 | `ex03_a_human_unknown_leaves_the_frame_reading_ready.py` |
| 4 | Split one broad allowed path into the smallest safe set. | code | T0 | `ex04_the_broad_path_is_405_files_and_the_receipts_name_two.py` |
| 5 | Add a scope receipt to the acceptance evidence. | code | T0 | `ex05_the_scope_receipt_is_the_only_acceptance_line_that_can_fail_on_a_green_test.py` |
<!-- generated:end -->

## Answers

### 1 — a frame whose receipts all resolve, and a validator that never checks

The bug framed here is one this repository already measured: Lesson 38's
verification gate writes a signed override log and never reads it, so a valid,
signed override changes no verdict. Framed without proposing a fix:

- **Goal.** A signed override lifts the finding it names on the verdict it names.
- **Facts.** The gate declares an override log path (`code/main.py:25`); overrides
  are written and signed (`code/main.py:174`); signatures are verified on read
  (`code/main.py:194`).
- **Allowed.** `code/main.py`, `code/tests/test_main.py`.
- **Forbidden.** `docs/en.md`, `outputs/**`.
- **Acceptance.** `python3 -m unittest discover code/tests`.
- **Unknown.** Whether an override is scoped to one `head_commit` or to every
  future one. (Human: it changes what an override means.)

The frame validates READY, and — the part worth doing — all three receipts resolve:
open the cited file, look at the cited line, find the claimed symbol. Shift every
line number by ten and 0 of 3 resolve. That check is four lines and `validate` does
not contain it: its evidence test is `if not fact.evidence.strip()`, so a fact
citing "obviously" passes. Five of its six branches are emptiness tests and none
opens a file.

One more thing the measurement showed: replacing the goal with "make `verify()`
read `overrides.jsonl`" — a solution, not a behaviour — still validates clean. The
discipline the lesson teaches lives entirely in the author.

### 2 — the shipped example has two facts and no resolvable receipt

An assumption is a claim whose receipt does not resolve, and the most honest place
to go looking is the frame the lesson ships to practise on. `example()` carries two
facts, `app/accounts.py:18` and `tests/test_accounts.py:44`, and **neither file
exists anywhere in the lesson directory**. What ships is `code/main.py` and
`code/tests/test_main.py`. Zero of two receipts resolve.

The one that matters is "Duplicate errors use status 409", because it is the claim
the whole frame turns on: it fixes the public contract the change has to match. Get
it wrong and the implementation can be clean, tested, and incompatible — the
failure the lesson's own opening describes. Replaced with a receipt that names a
real path and line, the fact resolves; shift that line by five and it stops. The
frame keeps its shape and gains a check.

And `validate` returns no issues and renders `Status: READY` for the fictional
version. A researched frame and an invented one are indistinguishable to the tool
that exists to tell them apart, which is worth knowing before trusting a READY.

### 3 — a human unknown leaves the frame reading READY

The human unknown: *does rejecting a duplicate return 409 or 422 to callers already
handling 409?* It is human rather than discoverable because no amount of reading
the repository answers it — it is a decision about what existing integrations are
allowed to break.

Adding it changes nothing any program can see. The frame goes from 3 unknowns to 4,
`validate` returns `[]` both times, and `render` prints `Status: READY` either way.
`TaskFrame` has 6 fields, `render` prints an `## Unknowns` section, and all 6
validator branches read the list zero times. "The agent should pause at human
unknowns before the choice is buried in code" is advice the program cannot hold
anyone to.

The missing piece is not the list — it is the class. The lesson names four
(discoverable, decidable, human, deferred) and then stores unknowns as bare
strings, so the example's own "Whether email comparison is case-insensitive" — a
discoverable unknown, answerable by reading the store — sits in the same untyped
list a contract-breaking one would go into. Tag each entry and the rule becomes one
line: refuse when any unknown is `human`. That takes the frame from 0 issues to 1
and from READY to BLOCKED, while a frame carrying only the discoverable unknown
still passes. Enforcement costs less than the paragraph explaining why it matters.

### 4 — the broad path is 405 files and the receipts name two

"Smallest safe" is derivable, not chosen. The frame already says which files the
change must touch: its receipts name the code, and its acceptance command names the
test directory. The allowed set is that union.

`phases/14-agent-engineering/**` matches **405** real files. The union of the
frame's own evidence is **2** — `code/main.py` and `code/tests` — a 202.5x
reduction that required no negotiation, only reading the frame back to itself.

Then the hole underneath. `validate`'s overlap check is
`set(allowed_paths) & set(forbidden_paths)`: a literal string intersection. Allow
`code/**` and forbid `code/main.py` and it reports **0 issues** — the forbidden
file sits inside the allowed glob and the two strings simply do not match. Expand
both sides against the real directory and intersect the *file sets* instead, and
the conflict shows up immediately.

The example makes this worse in a way that looks like good practice: its two
forbidden entries are globs (`migrations/**`, `deploy/**`) and its two allowed
entries are exact files. Negative space in globs and positive space in paths is the
right instinct — and it guarantees the two sets are written in different languages,
which is exactly when a string intersection means nothing.

### 5 — the scope receipt is the only acceptance line that can fail on a green test

A scope receipt is a command whose output is the file list, checked against the
frame's allowed paths. Its value is precise: it is the one piece of acceptance
evidence that can fail while every test passes.

Measured in a real temp checkout — the frame's two allowed files edited, plus one
write to `deploy/release.sh` — `git diff --name-only` reports 3 paths and 1 is
outside the allowed set. The unittest line in the same acceptance list passes
either way. Without the receipt, the frame closes green on a diff that touched the
release script.

Three things to get right when adding it:

**The frame does not run anything.** `validate` checks only that `acceptance` is
non-empty, so a frame whose sole evidence is `make coffee` returns no issues and
renders READY. The receipt is a command a human or a gate runs; the frame is a
promise that it was.

**Compare against the forbidden globs too, not just the allowed list.**
`deploy/release.sh` is simultaneously outside the allowed paths and inside
`deploy/**`. Lesson 38 scores those as `scope.off_scope` (warn) and
`scope.forbidden` (block). Checking only the allowed list collapses two severities
into one finding, and the blocking one is the one you wanted.

**`git diff --name-only` cannot see a new file.** It reports 3 paths where
`git status --porcelain` reports 4. Adding a file nobody asked for is among the
most common scope violations an agent commits, and the obvious command for the
receipt is blind to it. Use `git status --porcelain`, or add `--others` explicitly.
