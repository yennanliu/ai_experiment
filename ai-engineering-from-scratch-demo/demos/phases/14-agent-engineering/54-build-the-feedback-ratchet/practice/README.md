<!-- generated:start -->
# 14-agent-engineering / 54-build-the-feedback-ratchet

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/54-build-the-feedback-ratchet/) · upstream spec
`phases/14-agent-engineering/54-build-the-feedback-ratchet/docs/en.md`

```bash
uv run demo practice run 54-build-the-feedback-ratchet --ex 1
uv run demo explain 54-build-the-feedback-ratchet --ex 1
uv run pytest demos/phases/14-agent-engineering/54-build-the-feedback-ratchet
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Turn one incident and one user complaint into ratchet actions. | code | T0 | `ex01_the_incident_from_this_session_routes_to_the_backlog.py` |
| 2 | Name the earliest layer that can prevent each recurrence. | code | T0 | `ex02_the_earliest_layer_for_this_incident_is_not_one_of_the_five.py` |
| 3 | Add verification commands or observations to the lab output. | code | T0 | `ex03_the_verification_string_becomes_a_command_that_already_exists.py` |
| 4 | Define a retirement condition for a policy rule. | code | T0 | `ex04_the_retirement_condition_names_a_fact_the_action_does_not_carry.py` |
| 5 | Trace one accepted correction back into the next task frame. | code | T0 | `ex05_the_action_carries_seven_fields_into_a_frame_with_room_for_five.py` |
<!-- generated:end -->

## Answers

### 1 — the incident from this session routes to the backlog

Both inputs are real. The **incident** happened while these solutions were being
written: a later commit changed a number three earlier solutions measured, and the
test suite went red. The **complaint** is the one recorded in this project's
memory: the agent cannot open the pull request it is told to open.

| Signal | Router | Layer that actually owned it |
|---|---|---|
| a later commit changed a measurement three solutions depended on | `backlog` | evaluation |
| permission denied opening the pull request from the agent | `policy` | policy |

One of two. The complaint matches on "permission" and lands correctly; the incident
matches none of the router's keywords and falls through to `backlog` — the one
destination with no verification a machine can run, for the one signal that had
already been resolved by a test.

Three properties of the machinery. **Priority is severity × frequency**, so in the
lesson's own example a production write at severity 5 seen once scores 5 and sorts
*below* a false positive seen four times at 12 — the only authority signal is the
last row of the backlog. **The router reads the wording, and the wording is written
after the fix**: adding "regression" — true, and how a reviewer would phrase it —
moves the same incident from `backlog` to `evaluation` without changing a fact. And
**`promote` copies the owner verbatim**, so a signal with an empty owner becomes an
action with an empty owner; of the seven fields on the action, the one that makes it
an action rather than an observation is the one with no validation.

### 2 — the earliest layer for this incident is not one of the five

"Earliest" means the layer at which the failure stops being possible, not the layer
that noticed.

For the complaint, that is `policy` — a permission the repository owner grants — and
the router agrees. For the incident, the test suite is where it surfaced, but the
layer that makes it impossible is the fixture the solution reads — measure the tree,
not the commit graph. None of the five destinations describes a fixture, so the
correct answer is unreachable from the router's vocabulary.

Three measurements underneath. **The five durable artifacts do not exist**:
`promote` assigns `evaluations/regression-suite.json`,
`policies/authority-boundaries.json` and three more, and zero of them are present in
this tree — every signal is promoted into a filesystem nobody has created.
**Detection and prevention are one step apart and only one is recorded**: eight
solutions measure this repository and zero read its commit graph, and
`RatchetAction`'s seven fields cannot say whether a control prevents a failure or
merely reports it. And **the fall-through is
backwards**: a signal matching no keyword becomes a backlog item, which is the most
expensive destination and the only one with no runnable verification — zero of the
five evidence strings name a command.

### 3 — the verification string becomes a command that already exists

The lab already emits `verification_evidence`, so the work is replacing five canned
phrases with something a machine can run:

| Destination | Replacement |
|---|---|
| evaluation | `uv run pytest demos/phases/14-agent-engineering` |
| policy | `uv run python scripts/audit_practice.py 14-agent-engineering` |
| context | `uv run python scripts/coverage.py --check` |
| runtime | `uv run python scripts/check_deps.py` |
| backlog | *observation:* the shaped item is reviewed against the outcome frame |

Four commands and one observation, naming four paths in this repository — all four
of which exist.

The shipped phrases all begin with "Record" and none contains a path or an
executable: the field holds the *name* of the evidence somebody should go and
produce. And the replacement is destination-shaped for a reason — evaluation,
policy, context and runtime each have a runnable check here, while `backlog` is a
decision about whether work should exist and no command settles that. A ratchet
that demanded a command for every destination would push backlog items into
whichever bucket happened to have one.

The limit is worth stating plainly: a command in a string is still a string. The
module executes nothing, so replacing the phrase changes what a human would paste,
not what the system runs. The lesson's loop ends at "verify that recurrence becomes
less likely" and the artifact stops one step earlier.

### 4 — the retirement condition names a fact the action does not carry

The lab already writes one — "Remove or revise after 180 days without recurrence" —
and making it evaluable shows which half the action can check.

The rule is three comparisons: the window has elapsed, nothing recurred inside it,
and the control has not been blocking legitimate work. Against a fixture with a
promotion date of 2026-03-01 and a today of 2026-09-22 — 205 days — the quiet policy
rule **retires**, the same rule with one recurrence is **kept**, and evaluated
before the window closes it is **held**.

What the artifact cannot do: `promote` formats `expires_after_days` into the
sentence, so the number survives as text while `Signal.frequency` — the only
recurrence figure in the model — is multiplied into `priority` and never stored.
Zero of the action's seven fields hold a date and none holds a count. The module
imports no clock either, which is the right shape (the caller supplies today, as
Lesson 46 also needed) and means the evaluation lives entirely outside the record.

And one of the docs' four retirement reasons is not measurable at all: "blocks
legitimate work more often than it prevents harm" needs a count of refusals that
were wrong, which nothing in this system produces. Three of four are checkable from
a log; the fourth is why the noisy case returns "review for noise" rather than a
verdict.

### 5 — the action carries seven fields into a frame with room for five

The accepted correction is the one this session made — pin every git window to a
fixed commit — and the next task frame is Lesson 43's.

Five of the seven fields land: `change` → goal, `durable_artifact` → allowed path,
`verification_evidence` → acceptance, `owner` → a fact's evidence,
`retirement_check` → an unknown. `priority` and `destination` have nowhere to go.
Lesson 43's `validate` returns zero issues either way, which is the point: the
boundary loses two fields silently.

**The trace runs forward and nothing runs back.** `TaskFrame` has six fields and
none cites the signal, the incident or the action, so a reviewer reading the frame
cannot tell it exists because three solutions broke. That is the provenance Lesson
51 measured as missing, and this lesson's loop depends on it.

**The trace preserves the intent and loses the location.** The action's durable
artifact yields one allowed path — the one that does not exist — while the
recurrence itself names the three solution files in lessons 45, 47 and 48 that the
fix actually touched. Deriving allowed paths from where the failure recurred, rather
than from the destination's canned artifact, is the difference between a frame that
can be worked and one that cannot.

**And the loop closes only if the frame's acceptance is the ratchet's
verification.** Both fields hold a command string; the two artifacts share zero
identifiers, so nothing makes them agree. Setting them to the same command — here,
the phase test suite — is what makes "verify that recurrence becomes less likely"
something the next task proves rather than something a report claims.
