<!-- generated:start -->
# 15-autonomous-systems / 15-propose-then-commit

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/15-autonomous-systems/15-propose-then-commit/) · upstream spec
`phases/15-autonomous-systems/15-propose-then-commit/docs/en.md`

```bash
uv run demo practice run 15-propose-then-commit --ex 1
uv run demo explain 15-propose-then-commit --ex 1
uv run pytest demos/phases/15-autonomous-systems/15-propose-then-commit
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Confirm that a retry of an approved proposal uses the durable record and… | code | T0 | `ex01_the_key_covers_three_of_the_seven_fields.py` |
| 2 | Extend the proposal record with a `rollback` field. Simulate an execution whose verify step f… | code | T0 | `ex02_verify_reads_the_list_execute_wrote.py` |
| 3 | Read Microsoft Agent Framework's `RequestInfoEvent` docs. Identify one metadata field the API… | explain | T0 | prose, below |
| 4 | Design a challenge-and-response checklist for a specific action (e.g., "post to a public Twit… | code | T0 | `ex04_the_third_question_has_no_true_answer_for_a_public_post.py` |
| 5 | Pick one case where a synchronous "Approve?" prompt would be sufficient (no durable store nee… | code | T0 | `ex05_one_of_the_three_shipped_actions_needs_no_store.py` |
<!-- generated:end -->

## Answers

### 1 — the key covers three of the seven fields

Both halves of the exercise hold: a committed record short-circuits, so commit
plus two retries produces **1** side effect; and replacing `key()` with one
that includes a timestamp makes every `propose` a new record, so three
proposals produce **3** executions.

The interesting part is in between.

**The key hashes 3 of the proposal's 7 fields** — `thread_id`, `action`,
`payload`. Not `intent`, `lineage`, `blast_radius` or `rollback`. So:

```text
propose(intent="Announce the v1.2 release")  -> key 5d3a…, approved, committed
propose(intent="Send to every customer")     -> key 5d3a…, returns the same record
```

The second proposal's intent is never stored, never surfaced and never
reviewed. **The approval transfers to a proposal the reviewer did not see.**
This is the more serious failure than double-execution, because it is silent:
the log shows one approved, committed action with the first intent attached.
The fix is one line — hash the reviewed fields, not the executed ones — and it
changes what the key *means*: from "this side effect" to "this side effect,
justified this way".

**The idempotency lives on the status, not on the key.** `commit`
short-circuits on `status == "committed"`, so two records with different keys
for the same effect both execute. The key protects against retrying one
record; nothing protects against proposing the same action twice. The
timestamp demo makes that visible, but a timestamp is not required to reach
it — two threads proposing the same email already suffice.

**And the checklist is advisory.** `rubber_stamp_approve` sets the same
`approved` status with `ack_mode="rubber_stamp"`, and `commit` never reads
`ack_mode`. A rubber-stamped `db.drop_table` executes exactly as a
checklist-approved one does. The distinction the lesson's third demo exists to
draw is recorded and not enforced — one `if` away from being real.

### 2 — verify reads the list execute wrote

The field the exercise asks for is already present, and it is the wrong type:
1 of the 7 fields is named `rollback`, and 0 of the 7 hold a callable. "Extend"
here means change a string into something that can fire.

**Verify cannot fail as written.** It searches `SIDE_EFFECTS` for the string
`execute` just appended, so after a real execute it returns True in 5 of 5
trials. To produce the failure the exercise asks for you have to break the
contract — substitute an execute that performs no side effect — at which point
verify returns False and an added `rollback_fn` fires once.

**And commit does not branch on verify.** The call appears once, inside an
f-string in a `print`, and the next statement is `return True`:

```python
print(f"  [commit] executed; verify={verify(p)}")
return True
```

There is nothing for an automatic rollback to hang off. So the exercise's
"automatically" needs a new *call site*, not a new field — which is the real
shape of the change and is easy to miss when the field name already exists.

**The checklist's third question was answered yes for the one proposal whose
record says no.** `checklist_approve(..., rollback_ready=True)` is called for
the email whose `rollback` field reads `no in-band rollback; follow up with
correction email`. Nothing compares the boolean to the record. The
anti-rubber-stamp mechanism is itself rubber-stampable, and a three-line check
(`rollback_ready and not record["rollback"].startswith("no ")`) would have
caught the one case in the lesson's own demo.

**A compensating action is not a rollback.** Of the three shipped rollback
strings, two restore from a backup — one nightly, one weekly with "data loss
up to 6 days" — and one is a follow-up email. **Zero** return the system to
its prior state. The prose field is honest about this; the boolean beside it
is not, and that gap is exactly where a reviewer's attention goes.

### 3 — the metadata field the toy engine is missing

*Draws on "The idempotency key".*

The field the toy engine lacks is an **expiry on the pending request** — the
Microsoft `RequestInfoEvent` carries a timeout/deadline alongside the request
id and payload, and nothing in `Proposal` or the stored record does. What it
protects against is the approval that arrives *too late to still be correct*:
the durability property the lesson is proud of — "an overnight approval
arrives and the workflow picks up in the morning" — is exactly what makes a
stale approval executable. A proposal to close a stale issue, approved and
then committed three weeks later, executes against a row that may have been
reopened; the idempotency key cannot help, because the key covers the payload
and the payload has not changed. The world has.

Adding it is small — a `not_valid_after` on the record and a check in
`commit` — and it changes the state machine from three states to four, because
an expired proposal is neither `waiting` nor `approved` nor `committed`: it has
to be re-proposed and re-reviewed. That fourth state is the one the engine's
own `blast_radius` strings already imply, since two of the three are
denominated in time windows.

### 4 — the third question has no true answer for a public post

**The three questions**, mapped onto the parameters `checklist_approve`
already takes:

| # | question | parameter | honest answer for a public post |
|---|---|---|---|
| 1 | Is the text exactly what will appear? | `understood` | yes |
| 2 | Is the account and its follower set what the proposal says? | `verified` | yes |
| 3 | Can this be withdrawn, and what does withdrawal cost? | `rollback_ready` | **no** |

Answered honestly, `checklist_approve` **rejects**, the status stays
`waiting`, `commit` refuses, and **0** side effects occur. Ticked anyway, it
approves, commits, and produces 1.

**Why those three.** They are the only three that cannot be inferred from each
other: content is *what the action does*, audience is *who it reaches*,
withdrawal is *what happens if the first two were wrong*. `blast_radius`
informs the second and `lineage` the first; `intent` informs none of them,
which is why a checklist keyed on intent would wave through a proposal whose
payload had changed underneath it — exactly the hole exercise 1 found in the
idempotency key.

For a public post the third answer is no, and that is the point of asking it:
deletion removes the post and not the copies. The correct outcome is that this
action is not checklist-approvable at all — it needs a different control
(a staging account, a second approver, a delay) rather than a reviewer being
asked a question with no true answer.

**Rejection is the only outcome the checklist can produce.** One of the eight
answer combinations approves; the other seven reject with the same message. A
reviewer who cannot answer the withdrawal question produces the same artifact
as one who did not read the text. Recording *which* question failed is one
field away, and it is the difference between "declined" and "declined because
it cannot be undone" — the second of which is a design input.

### 5 — one of the three shipped actions needs no store

**The case: the single-row `db.update`**, and the reason is in its own record —
`one DB row; reversible within 1h backup window`. Of the three shipped
proposals it is the only one whose blast radius declares reversibility;
`email.send` declares no in-band rollback and `db.drop_table` declares "not
reversible within 24h".

**The risk class accepted is a lost approval, not a double execution.**
Without a durable store, a process death between the prompt and the execution
loses the answer and the reviewer is asked again. That is acceptable exactly
when re-asking is cheap *and* the action is idempotent — and setting a row to
`closed` twice is the same as once. Naming the class precisely matters: what
you give up is **reviewer time**, and what you must not give up is bounded
side effects. The word "sufficient" in the exercise invites conflating them.

**The store buys lateness, and two of three proposals need it.** Two of the
three blast radii are denominated in time, so "may this approval arrive
tomorrow?" has a different answer per proposal. Durability is not a safety
property here; it is what makes an approval that outlives its process still
apply — the Lesson 12 result restated for humans.

**And re-approving a committed record resets it.** `checklist_approve` writes
`status = "approved"` without reading the current status, so a second approval
of an already-committed proposal makes `commit` execute again: **2** side
effects against **1** without it. The durable store prevents a *retry* from
double-executing and does nothing about a second *approval* — which is the
path a synchronous prompt takes on every re-ask, and the reason the
"synchronous is sufficient" argument needs the idempotency to live in the
action rather than in the store.
