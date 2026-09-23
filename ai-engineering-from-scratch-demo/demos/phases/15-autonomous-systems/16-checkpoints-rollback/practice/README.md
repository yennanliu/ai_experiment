<!-- generated:start -->
# 15-autonomous-systems / 16-checkpoints-rollback

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/15-autonomous-systems/16-checkpoints-rollback/) · upstream spec
`phases/15-autonomous-systems/16-checkpoints-rollback/docs/en.md`

```bash
uv run demo practice run 16-checkpoints-rollback --ex 1
uv run demo explain 16-checkpoints-rollback --ex 1
uv run pytest demos/phases/15-autonomous-systems/16-checkpoints-rollback
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Verify the four scenarios. For the crash-during-commit case, confirm the… | code | T0 | `ex01_the_retry_short_circuits_before_the_verify_it_motivates.py` |
| 2 | Modify the "mark as done first, then do it" pattern so the status write fires after the actio… | code | T0 | `ex02_one_duplicate_per_retry_and_no_bound_on_retries.py` |
| 3 | Design a rollback plan for a specific production action (e.g., "post to a Slack channel"). Cl… | code | T0 | `ex03_the_shipped_rollback_is_compensating_and_calls_itself_a_restore.py` |
| 4 | Take one workflow you know. Identify every state transition. Mark each with a durability requ… | code | T0 | `ex04_five_states_four_persisted_transitions_and_one_that_is_two.py` |
| 5 | Rehearsed-rollback test: design an end-to-end test that runs a real workflow, crashes it, and… | code | T0 | `ex05_the_shipped_assertion_cannot_tell_rolled_back_from_never_ran.py` |
<!-- generated:end -->

## Answers

### 1 — the retry short-circuits before the verify it motivates

**The injected crash does fire exactly once.** Crashing after
`persist_transfer` leaves the record at `committed` and the balance already
moved; the retry returns `idempotent-skip` and moves nothing more. One
transfer across two attempts.

**The adjacent crash gives exactly zero.** Crash between `cp.save` and
`persist_transfer` — one instruction earlier — and the record still says
`committed` while no money moved:

| crash point | balance moved | retry returns | total moved |
|---|---:|---|---:|
| after `persist_transfer` | 100 | `idempotent-skip` | **100** |
| before `persist_transfer` | 0 | `idempotent-skip` | **0** |

The marker records an *intention* and is read as an *outcome*. The module's
own comment says this, and it is worth restating as a number: "exactly once"
holds for one of two adjacent instruction boundaries, and the failure on the
other side is silent — the caller is told the work is done.

**The retry never reaches verify.** `committed` is one of the four terminal
states, so the short-circuit returns before the post-action read, and the
record stays at `committed` rather than advancing to `verified`. The scenario
that motivates a verify step is the one scenario in which it does not run —
which is why the zero-transfer case has nothing to catch it. Promoting
`committed` to a *non*-terminal state and letting the retry fall through to
verify would close the gap for free, at the cost of one extra read.

**And the four scenarios share one database.** `DB` is a module global every
scenario mutates, so the 1500 starting balance is 1300 by the time scenario
3's precondition runs. It does not change that scenario's verdict; it does
mean the four demonstrations are not independent of their order.

### 2 — one duplicate per retry, and no bound on retries

Moving the status write after the action, and retrying the same crash:

| retries | reordered (marker last) | shipped (marker first) |
|---:|---:|---:|
| 1 | 200 | 100 |
| 2 | 300 | 100 |
| 3 | 400 | 100 |

**One duplicate per retry**, because a crash between the effect and the marker
leaves no record at all, so every attempt re-executes. Nothing in the workflow
caps attempts.

**The reorder trades a silent zero for a loud multiple.** Both orderings have
a single-instruction window; they differ only in which side of the ledger
absorbs it. Marker-first fails closed and lies; marker-last fails open and
repeats. Neither is the fix — the module's own comment names it: carry the
idempotency key into the side effect so the destination enforces exactly-once,
which removes the window rather than moving it.

**The precondition is what bounds the damage.** Retrying the reordered crash
until the balance hits the floor stops at **13** transfers — 1500 down to 200
at 100 each. The only thing that ends an unbounded retry loop here is a
business rule that happens to be checked first, and it bounds the loss at
1300 rather than at 100.

**And the verify step cannot see a duplicate.** It compares
`DB["last_transfer_id"]` to `txid`; a second identical transfer sets the same
id. After three executions moving 300, verify still returns true. The
post-action read confirms that *a* transfer landed, never that *one* did —
so the fourth piece of the lesson's four-piece stack does not cover the
failure the second piece creates.

### 3 — the shipped rollback is compensating and calls itself a restore

**The action: post to a Slack channel. The classification: out-of-band** —
and the reasoning is that the three classes are distinguished by what an
observer can see afterwards, not by how the code is written.

| step | class | guarantee? |
|---|---|---|
| `chat.delete` the message | in-band | no |
| post a correction naming the error | compensating | no |
| page the on-call human | out-of-band | **yes** |

`chat.delete` is genuinely in-band *for the channel*: a reader arriving later
sees the prior state. It is not in-band for anyone present when it posted —
the notification fired, the email digest may have shipped, a screenshot may
exist. None of those is reachable. So the plan has three steps and only the
third is a guarantee, which is what "out-of-band" means here: the class of a
rollback is the weakest link in it.

**The module's own `rollback_transfer` is misclassified by the same test.**
Its comment says "restore balances and the prior transfer id", and it works by
adding back what it subtracted — a second write. Measured: after a rollback,
3 of 3 database fields match the starting state, reached by 3 further writes.
Identical final state; two transfers in any audit log. That is a compensating
transaction, and calling it a restore is the difference between "nothing
happened" and "two things happened that cancel".

**The distinction is invisible to the module because it has no history.** `DB`
holds three keys and none is a log, so nothing here can express "the balance
returned to 1500 having been 1400 in between". The class an action belongs to
is a property of the observer, and this observer has no memory.

**And the no-op class has no representation at all.** The lesson says a no-op
rollback "must be named in the proposal"; the checkpoint record carries six
fields and none is a rollback plan. Lesson 15's `Proposal` has the field, and
this module — where the rollback actually fires — does not, so the two halves
of the design do not meet.

### 4 — five states, four persisted transitions, one that is two

The workflow is `run_transfer`, chosen because its transitions can be counted
from the source rather than recalled.

| | count |
|---|---:|
| states named | 5 |
| `cp.save` calls | 4 |
| side effects | 1 |
| **transitions not persisted** | **0** |

**Every transition is durable.** That is the answer the exercise asks for, and
it is the answer a compliance checklist wants. It is also not the useful
answer.

**The useful count is one.** `committed` is written *before*
`persist_transfer` and never rewritten, so "intent recorded" and "effect
applied" are the same stored value — one state covering two situations, which
is exactly the ambiguity exercise 1 measures in money.

**The unpersisted transition is the side effect itself.** Four checkpoint
writes; one call that changes the world; no checkpoint of its own. The
durability requirement it needs is not another `cp.save` — a marker after the
effect has the same window on the other side, as exercise 2 shows — but a
receipt from the destination.

**And two of the five states are unreachable by retry.** `verified` and
`rolled-back` are written only on a path that completes, so a crashed
transaction is stuck at `committed` forever. Two of five states exist only in
the absence of the failure the checkpoint was built for — which is a useful
thing to check on any state machine whose job is recovery.

### 5 — the shipped assertion cannot tell rolled back from never ran

**What the test asserts — four things:**

1. the side effect **landed** before the rollback (balance 1400)
2. the state is **restored** afterwards (balance 1500)
3. the checkpoint records `rolled-back`
4. the return value says `verify-fail-rolled-back`

The shipped scenario asserts **one** of them: `balances_before ==
balances_after`.

**That assertion passes on an empty run.** Comparing the database to itself
across a workflow that never executed gives `True` — identically to a real
rollback. So the test cannot distinguish "the effect was applied and reversed"
from "the effect was never applied", which is precisely the pair exercise 1
shows the *engine* cannot distinguish either. The demonstration and the bug
have the same blind spot, which is how a bug survives a demonstration.

**Assertion (1) needs an observation the module does not make.** Proving the
effect landed means reading the intermediate state, and there is no hook
between `persist_transfer` and the verify. A real rehearsal instruments the
side effect rather than the workflow — here, by wrapping `persist_transfer` and
sampling the balance, which reads 1400 at that moment and 1500 after. Only the
first number is evidence.

**The shipped rollback does satisfy the other three**: 3 of 3 fields restored,
`rolled-back` in the checkpoint, and the matching return value. Adding
assertion (1) is the entire delta between a demonstration and a test — one
observation, and the one that makes the other three mean something.
