<!-- generated:start -->
# 14-agent-engineering / 40-multi-session-handoff

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/40-multi-session-handoff/) · upstream spec
`phases/14-agent-engineering/40-multi-session-handoff/docs/en.md`

```bash
uv run demo practice run 40-multi-session-handoff --ex 1
uv run demo explain 40-multi-session-handoff --ex 1
uv run pytest demos/phases/14-agent-engineering/40-multi-session-handoff
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add an `assumptions_to_validate` field that surfaces every assumption the builder logged but… | code | T0 | `ex01_the_filter_as_written_selects_no_assumptions_at_all.py` |
| 2 | Trim the feedback summary differently for failing runs versus passing ones. Defend the asymme… | code | T0 | `ex02_the_shipped_trim_gives_the_passing_run_context_and_the_failing_run_none.py` |
| 3 | Include a "questions for the human" list. What is the threshold for a question to make it int… | explain | T0 | prose, below |
| 4 | Make the generator idempotent: running it twice produces the same packet. What needs to be st… | code | T0 | `ex04_it_is_already_idempotent_and_that_is_not_the_property_you_want.py` |
| 5 | Add a "next session prereqs" section listing exactly the artifacts the next session must load… | code | T0 | `ex05_the_packet_names_two_paths_and_the_next_session_needs_six.py` |
<!-- generated:end -->

## Answers

### 1 — the filter as written selects no assumptions at all

The field joins two artifacts: the builder's assumption list in `state`, and the
reviewer's score for the `assumptions` dimension. Check the join before writing
it. Lesson 39's `score_assumptions` returns 2 whenever the list is non-empty and 1
only when it is empty — so "assumptions the reviewer did not score above 1" is the
empty list when there is nothing to list, and the empty list again when there is
something. Swept over 0 to 5 logged assumptions, the field has length 0 in 6 of 6
cases. It is not a rare edge; it is every input.

The root cause is that the reviewer scores the *dimension*, never the assumption.
`DimensionScore` carries `name`, `score`, `note`, and the note is a count —
"3 assumptions recorded". Nothing in the review report names which assumption is
weak, so no filter over it can either. And even the per-dimension list Lesson 39
does write to `review_report.json` never arrives: the handoff module reads
`review["verdict"]` and `review["total"]` out of a `dict[str, object]`, so widening
this field means widening the read, not just the payload.

The version worth shipping does not go through the reviewer at all. Mark an
assumption unvalidated when no command that ran and no file that changed mentions
any word in it: on a three-assumption fixture that selects 2, leaving out the one
the test run actually exercised. Evidence the workbench already has, and a field
with content in it.

### 2 — the shipped trim gives the passing run context and the failing run none

The defence of an asymmetry has to say what each log is evidence *of*. A passing
run's log is evidence that the commands ran — the last K entries are the whole
story. A failing run's log is evidence of a causal sequence: what ran before the
failure, what was retried after. A tail is the wrong window for that, and
`trim_feedback` uses a tail for both.

On a 20-command log whose only failure is at index 2, it returns
`['c15', 'c16', 'c17', 'c18', 'c19', 'c2']`. Zero of the three commands adjacent to
the failure survive; a window of radius 2 keeps all three. The passing run,
meanwhile, keeps 5 of its last 5. The trim is generous with the log that needs no
context and stingy with the one that does.

Three specifics behind that:

**Order.** `tail + nonzero` puts the failure at position 6 of 6, after commands
that ran fifteen steps later. A next session reading the tail top to bottom sees
the effect before the cause.

**The cap that is not one.** `failed_attempts` is built from every non-zero exit
with no limit. A 180-command session with 60 failures writes 60 markdown lines and
64 `feedback_tail` records. The trim shrinks the quiet packet and lets the loud one
through — the opposite of a budget.

**Identity dedup.** The same failing record handed in twice collapses to 1 entry in
`feedback_tail` while `failed_attempts` reports 2, because the dedup key is
`id(r)`. One packet, two answers to "how many times did this fail".

### 3 — the threshold for a question to the human

**A question belongs in the packet when its answer changes `next_action`, and
nowhere else.** Everything else is a chat message, and the test is mechanical
enough to apply without judgment:

1. **Does the answer change what the next session does first?** If the next action
   is the same either way, the question is curiosity and belongs in chat.
2. **Can it be answered by running a command or reading an artifact the next
   session already loads?** Then it is not a question for a human — it is an
   evidence request (Lesson 39), and asking a person for it burns a round trip on
   something the workbench can produce.
3. **Is the answer a decision only a human has standing to make?** Scope changes,
   product trade-offs, risk acceptance, anything touching money or users. These are
   the questions that survive both filters.

The doc lists **Seven fields every handoff carries**, and the bar for an eighth is
the same bar: does the next session act differently because the field is there.
`questions_for_human` clears it on the strength of rule 1 alone — a blocked
`next_action` with no visible question is the state in which the next session
either guesses or stalls.

There is a concrete reason to add it here rather than lean on what exists. The
packet already carries questions, mislabeled. `derive_risks` files every entry of
`state["blockers"]` as `{"severity": "warn"}`, so the demo's markdown renders:

```
- [warn] off-scope: README.md
- [warn] open blocker: awaiting decision on rate-limit window
```

One of those is a mechanical finding a human can skim past. The other is a person
being asked for a decision, and the packet gives a reader no way to tell them
apart. A `questions_for_human` list with `{question, why_blocked, what_i_assumed}`
separates them, and the third key is what keeps the next session moving: state the
assumption you proceeded under, so an unanswered question degrades into a recorded
risk rather than a halt.

Two limits on the field. Cap it — a packet with nine questions is an agent that
should have stopped and asked hours ago, and the cap is what makes that visible.
And expire it: a question carried unanswered into a third session is not a
question, it is a decision nobody is making, and it belongs in `open_risks` where
it will be read as one.

### 4 — it is already idempotent, and that is not the property you want

Run it twice before changing anything. `generate_handoff` has no clock, no uuid and
no hash of a mutable, so on one snapshot both the markdown and the payload come
back byte-identical. The exercise's stated goal is already met; the interesting
question is the second half — what has to be stable — and the answer is that none
of it is the generator's to own:

- **The diff listing.** Reversing `diff_summary["touched"]` changes the markdown
  while the file set is equal. If `touched` comes from a set or a `git status`
  parse, the packet is unstable for no semantic reason. The generator should sort.
- **The feedback log.** Append the generator's own command — which is exactly what
  Lesson 37's runner does if the generator is invoked through it — and
  `commands_run` grows by 1 and the tail shifts. A generator logged by the thing it
  reads is not idempotent on the second run by construction.
- **The findings order.** `open_risks` follows `verdict["findings"]` order, so the
  packet inherits whatever order the gate emitted.

Three more things the measurement turned up. `HandoffPayload` has 9 fields and none
of `branch`, `head_commit` or `status`, so a re-run and a packet from last week are
byte-indistinguishable — which is the exact failure the doc's "one active handoff
per branch and topic" pattern exists to prevent, and it makes byte-identity
worthless as a freshness signal. `main` writes `handoff.md` and `handoff.json` to
one path with no history, so "produces the same packet" is unobservable from disk.
And `derive_risks` defaults `review["total"]` to 10, so a snapshot with no review at
all and one with a perfect review both come out at 3 risks: two different worlds,
one packet.

### 5 — the packet names two paths and the next session needs six

"Exactly the artifacts" is a list that can be checked against a real directory, so
that is how it should be built: paths, a reason each, and an existence check.

The packet names two: `outputs/verification/<task>.json` and
`outputs/review/<task>.json`, both f-strings that nothing verifies. The list the
next session actually needs is six, in load order:

| # | Artifact | Why, before acting |
|---|---|---|
| 1 | `agent_state.json` | where the work stopped and what the blockers are |
| 2 | `feature_list.json` | which feature is active; only one may be `in_progress` |
| 3 | `outputs/scope/<task>.json` | the globs that bound the next diff |
| 4 | `outputs/verification/<task>.json` | the gate's findings, including unresolved ones |
| 5 | `outputs/review/<task>.json` | the reviewer's per-dimension scores |
| 6 | `feedback_record.jsonl` | the full command log the packet only tails |

Four of those the packet never mentions. Run the check against a temp workbench
holding four of the six and it names the two that are missing — the scope contract
and the feedback log — while `generate_handoff` emits the same two pointers it
always emits, green either way. A pointer that cannot be wrong is not a receipt.

The order matters and is part of the deliverable: state before board (the board
says which feature is active, the state says where the work stopped inside it), and
both before the contract that bounds the next diff. An unordered set leaves the
next session to rediscover the sequence, which is the rediscovery the packet exists
to stop.

Last: the demo's `next_action` is "open PR with current diff and request review" —
prose naming 0 artifacts and 0 commands. `next_action` is the load-bearing field,
and the prereq list is what makes it executable rather than aspirational.
