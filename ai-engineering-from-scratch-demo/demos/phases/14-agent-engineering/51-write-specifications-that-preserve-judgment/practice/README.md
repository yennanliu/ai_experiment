<!-- generated:start -->
# 14-agent-engineering / 51-write-specifications-that-preserve-judgment

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/51-write-specifications-that-preserve-judgment/) · upstream spec
`phases/14-agent-engineering/51-write-specifications-that-preserve-judgment/docs/en.md`

```bash
uv run demo practice run 51-write-specifications-that-preserve-judgment --ex 1
uv run demo explain 51-write-specifications-that-preserve-judgment --ex 1
uv run pytest demos/phases/14-agent-engineering/51-write-specifications-that-preserve-judgment
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Convert a backlog ticket into the six specification surfaces. | code | T0 | `ex01_the_ticket_had_six_sentences_and_four_of_them_were_one_surface.py` |
| 2 | Replace three implementation instructions with one invariant and two examples. | code | T0 | `ex02_three_instructions_become_one_invariant_two_examples_and_a_lost_reason.py` |
| 3 | Mark every decision and justify each locked or bounded choice. | code | T0 | `ex03_moving_the_production_write_to_delegated_removes_its_rationale_requirement.py` |
| 4 | Add a proof receipt for every invariant. | code | T0 | `ex04_two_invariants_two_proofs_and_no_line_between_them.py` |
| 5 | Remove a constraint that has no evidence or risk rationale. | code | T0 | `ex05_the_constraint_with_no_rationale_is_the_one_the_validator_allows.py` |
<!-- generated:end -->

## Answers

### 1 — the ticket had six sentences, and four of them were one surface

The ticket is a real one from this repository: the scaffolder raised
`FileNotFoundError` on lessons shipping English documentation only, and the fix
landed in commit `98c2d51`. Written as six surfaces it compiles `executable` with
zero issues:

- **Outcome.** Every lesson in the phase can be scaffolded, whatever languages it
  ships.
- **Invariants.** A lesson with no Chinese document still scaffolds; a bilingual
  lesson keeps both texts distinct.
- **Examples.** An English-only lesson's manifest has `zh` equal to `en`; a
  bilingual lesson's differs.
- **Non-goals.** Translating anything; changing the exercise text.
- **Decisions.** Which exception the fallback catches (delegated); whether the
  fallback may invent Chinese text (locked); what fills the Chinese slot (bounded).
- **Proof.** Both manifests, read from the repository.

Four of the ticket's six sentences — "catch the error", "default to an empty
list", "keep the English text", "leave the bilingual path alone" — are all
*decisions*, and two of them were someone's implementation choice rather than a
requirement.

Two things the conversion exposes. The validator requires a rationale only when
the mode is **not** delegated, while the lesson's text says a delegated decision is
one the agent "owns and must explain" — so the surface carrying the agent's
reasoning is the one surface nothing checks, and the shipped example proves it by
compiling clean with an empty delegated rationale. And the invariant is checkable
where the instruction was not: 5 of 5 exercises in the English-only manifest carry
identical `en` and `zh` text, against 0 of 5 in a bilingual one, whereas "default
`zh_items` to an empty list" cannot be verified from any artifact.

### 2 — three instructions become one invariant, two examples, and a lost reason

The three instructions the ticket actually carried:

1. wrap the call in `try/except FileNotFoundError`
2. set `zh_items` to an empty list in the handler
3. let `build_manifest` fall back to the `en` text

They name three code symbols between them. The replacement names none:

> **Invariant.** A lesson with no Chinese document scaffolds, with the English text
> in both slots.
> **Examples.** `47-outcomes-before-output` (English-only) and
> `42-agent-workbench-capstone` (bilingual).

Confirmed by 5 of 5 exercises in the first manifest and denied by 0 of 5 in the
second.

What changes is the shape of a failure. An instruction fails when the code stops
matching the sentence; the invariant fails when the artifact stops matching the
claim — corrupting one exercise's Chinese text drops it from 5 of 5 to 4 of 5, a
failure visible without opening the scaffolder. Two examples suffice because the
scaffolder has exactly two paths (the file exists or it does not), and a third
bilingual lesson adds zero coverage — which is the test for whether an example
earns its place.

What is lost is the reason. `Specification` has 6 fields, `Decision` has 3, and
neither links back to the ticket, the commit, or the crash that prompted any of it.
The lesson's own further reading is about precisely this — preserving where a
requirement came from — and the artifact it ships has nowhere to put it.

### 3 — moving the production write to delegated removes its rationale requirement

Six decisions, marked: 2 delegated (which exception to catch, how to name the
helper — both cheap and reversible), 2 bounded (what fills the Chinese slot, how
many documents may be read), 2 locked (may the fallback invent Chinese text, may
the scaffolder overwrite an existing manifest). All four constrained decisions
carry a rationale.

Then the lesson's own experiment: move the production-write decision from `locked`
to `delegated` and delete its rationale. The contract still compiles `executable`
with zero issues, and `human_checkpoint` drops from 2 entries to 1. The schema
accepts it because `validate` only demands a rationale for non-delegated modes —
so the decision with the highest consequence becomes the one the document stops
asking about. The product risk does not change; the paperwork does.

`VALID_MODES` holds three strings and the check is membership. The words
consequence, reversibility and authority appear zero times in `validate` and
`compile_contract` — the decision table in the docs is advice the code cannot
apply, which is why marking is a human act that the schema merely records.

One shape problem worth fixing: `compile_contract` publishes
`{"question": ..., "boundary": item.rationale}`, so the same string is both the
justification and the limit. "Stop after five sources or two minutes" works as a
boundary; "production authority stays with the incident commander" is a reason, and
a bounded decision given that kind of string yields a boundary nobody can check.

### 4 — two invariants, two proofs, and no line between them

Two of each looks like a receipt apiece until you write the pairing down:

| Invariant | Proof |
|---|---|
| diagnosis is read-only | zero production writes |
| every source is included in the audit record | — |

**1 of 2 covered.** And the leftover proof, "ten recorded incident replays", is a
proof of the *outcome* — did it find the service — not of either invariant. The gap
is invisible while both are `list[str]` sitting side by side.

The lesson's own proof ladder names the fix and the error at once: it lists unit,
wire, journey, replay and audit-log proofs and warns against accepting a lower
layer for a higher claim. The uncovered invariant is about the audit record; zero
of the shipped proofs is an audit log. Adding one takes the list to three entries
and closes the gap.

Two consequences for the artifact. Adding a receipt per invariant means proof
becomes a mapping rather than a list — and `validate` currently checks only that
both lists are non-empty, so a specification whose proofs prove something else
entirely reports zero issues. And `compile_contract` returns six keys, none
relating invariants to proofs, so a reviewer counting receipts sees the same
document at 1-of-2 coverage as at 2-of-2.

### 5 — the constraint with no rationale is the one the validator allows

The specification carries two non-goals: "automatic remediation" and "changing
alert routing". Neither shares a single word with any decision's rationale, so
there is no mechanical way to tell them apart — the separation comes from reading.
Automatic remediation is exactly what the locked production-authority decision
exists to exclude. Alert routing is argued for nowhere.

So alert routing comes out. The trimmed contract still compiles `executable` with
zero issues, which is the point: nothing the document measures changes, because
nothing the document measures was holding that constraint up.

Three observations about why this had to be a judgment. Of the six surfaces, only
`decisions` carries a `rationale` field; invariants, examples, non-goals and proof
are lists of bare strings — four surfaces that cannot hold a justification at all.
Deleting the *defended* non-goal instead also reports zero issues, leaving a locked
decision standing with nothing to exclude; both deletions look identical to
`validate`, and only one of them loses an argument. And the one thing the validator
does catch is removing both, which reports `non_goals is empty`. Its opinion is
about presence, not content: a specification may carry constraints nobody can
defend, as long as it carries some.
