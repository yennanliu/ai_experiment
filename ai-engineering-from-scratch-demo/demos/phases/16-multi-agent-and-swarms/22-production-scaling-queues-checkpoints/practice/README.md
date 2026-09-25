<!-- generated:start -->
# 16-multi-agent-and-swarms / 22-production-scaling-queues-checkpoints

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/22-production-scaling-queues-checkpoints/) · upstream spec
`phases/16-multi-agent-and-swarms/22-production-scaling-queues-checkpoints/docs/en.md`

```bash
uv run demo practice run 22-production-scaling-queues-checkpoints --ex 1
uv run demo explain 22-production-scaling-queues-checkpoints --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/22-production-scaling-queues-checkpoints
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Confirm checkpoint resume works; measure async vs thread concurrency diff… | code | T0 | `ex01_resume_works_because_the_crash_is_placed_where_nothing_can_go_wrong.py` |
| 2 | Implement an outbox table: every tool call writes to outbox first, then a separate goroutine/… | code | T0 | `ex02_the_outbox_dedups_the_request_and_still_repeats_the_effect.py` |
| 3 | Simulate a rainbow deploy: two concurrent runtime versions; route half of new thread_ids to e… | code | T0 | `ex03_a_fifty_fifty_split_means_the_old_version_never_drains.py` |
| 4 | Read LangGraph's runtime doc (linked below). Identify which features of the runtime would tak… | explain | T0 | prose, below |
| 5 | Read MegaAgent (arXiv:2408.09955) Section 3. The two-layer coordination (intra-group + inter-… | code | T0 | `ex05_the_admin_queue_needs_a_batch_and_gets_one_message_per_three_steps.py` |
<!-- generated:end -->

## Answers

### 1 — resume works because the crash is placed where nothing can go wrong

**Resume works, and threads are about 1.3x slower than asyncio, not "several
seconds".** Worker 1 crashes at super-step 3 and leaves the checkpoint
`(2, {"counter": 3})`. Worker 2 resumes and finishes at 5. The table it leaves
matches, row for row, the table of a run that never crashed. On the lesson's
500 calls of 50ms each, asyncio takes about 0.05s and threads about 0.07s.
Starting a thread costs tens of microseconds, which is nothing next to a 50ms
sleep.

Two things make the resume look better than it is.

**The crash can only fall between steps.** `crash_at` is checked *before* the
increment, so every simulated crash lands just after a committed checkpoint.
The resumed run redoes nothing: across both workers the executed steps are
exactly 0, 1, 2, 3, 4. And the agent has no side effect, so "no duplicate
side effects" holds trivially.

**There is no lease, and `INSERT OR REPLACE` hides the double run.** Two
workers that both resume the crashed thread both run the remaining steps, in
the order 3, 4, 3, 4. The table still holds 5 rows identical to a clean run's,
so nothing records that the work was done twice.

Finally, **the demo never measures memory**, though the lesson says it reports
"peak memory (approximated)". The module imports no memory API, and "~1MB per
thread stack" is a hard-coded string. Measured as peak RSS in a fresh
interpreter, a sleeping thread costs about 36KB resident and a pending
coroutine about 1.4KB. That is roughly 25x: a real difference, but one order
of magnitude, not the "orders of magnitude" claimed, and far below 1MB.

### 2 — the outbox dedups the request and still repeats the effect

**Idempotent at the outbox, keyed on `(thread_id, super_step)`.** Here is what
happens each time a call gets repeated:

| repeated by | effects |
|---|---:|
| the agent issuing the same call twice | 1 |
| two workers resuming a crashed thread, via the outbox | 5 for 5 steps |
| the same two workers, calling the tool inline | 7 for 5 steps |
| an executor that crashes after the effect, before marking the row done | **2** |
| the same executor, when the tool dedups on the key | 1 |

The outbox makes delivery at-least-once, and exactly-once still requires the
receiving service to honour the key. The lesson's "both steps idempotent" is
load-bearing, and the outbox cannot supply the second step.

**The outbox must share the checkpoint's transaction.** The outbox table here
lives on the reference `CheckpointStore`'s own SQLite connection and never
commits by itself; `CheckpointStore.write` commits both. That matters when a
step's output varies between attempts, as an LLM's does. Suppose attempt 1
writes its payload and then crashes before the checkpoint commits:

- **shared transaction:** the outbox row rolls back with the checkpoint, and
  attempt 2's payload is the one executed.
- **outbox committed separately:** the dedup key discards attempt 2's row.
  The executor sends attempt 1's payload while the checkpoint records
  attempt 2's state.

In the second case the effect and the recorded state disagree, and no dedup
key can detect it.

### 3 — a fifty-fifty split means the old version never drains

Here a "version" is the reference agent run with a different goal: v1 stops
at 5, v2 at 7. Ten v1 threads are paused at super-step 2 using the reference's
own `crash_at`, then v2 ships.

**Pinned threads finish on v1, 10 of 10, and none is interrupted**, but only
because the pin is stored outside the checkpoint table. `checkpoints` has
three columns, `thread_id`, `super_step` and `state_json`, and no version.
After a kill-and-replace deploy, a v2 worker resumes all 10 without complaint
and ends them at 7. That gives 10 runs whose steps 0–2 ran on v1 and 3–6 on
v2, recorded exactly like clean runs.

**"Route half to each" keeps the old version alive forever.** With 2 new
threads per tick, each living 5 ticks, a 50/50 split keeps v1 at 4 to 6
in-flight threads at every tick after the fifth. Sending all new threads to v2
drains v1 at tick 3. A rainbow deploy can only retire a version once it stops
receiving new work, which is why the canary slice goes to the *new* version.

**Python's `hash()` cannot be the router.** `str` hashes are salted per
process, so `hash(thread_id) % 2` sends about half of 1000 thread ids to the
other version after a restart with a different seed. A restarted router would
send resumed threads to the wrong pool. `zlib.crc32` gives the same split in
every process.

### 4 — the lease and the double-text are the expensive half

*Draws on "A checkpoint-per-step runtime", checked against LangChain's "The runtime behind production deep agents".*

The page lists these runtime features:

- a checkpoint per super-step ("PostgreSQL by default")
- a managed task queue
- lease release on worker crash ("another worker picks it up from the latest
  checkpoint")
- `interrupt()` with `Command(resume=...)`
- streaming
- short-term memory in the checkpoint and long-term memory in a store
- cron
- four double-texting strategies (enqueue, reject, interrupt, rollback)
- time travel
- background runs
- retry policies
- sandboxes
- auth middleware
- tracing

Measured against this lesson's module, the checkpoint is the part that is
already done, and it was cheap: one table, one `INSERT`, one `SELECT ... ORDER
BY super_step DESC`. Time travel is nearly free as well, because every
super-step's row is kept and resuming from an earlier one is a different
`WHERE`.

The two that would take longest are the ones that are about concurrency, not
storage:

- **Leases.** Exercise 1 shows what their absence costs: two workers run the
  same steps and the table cannot tell. A correct lease needs expiry, a
  heartbeat, fencing so a stalled worker cannot write after its lease is
  gone, and tests for every one of those races.
- **Double-texting.** A second user message arriving mid-run has four
  defensible answers. Each touches the queue, the checkpoint and any
  outstanding side effects, and `rollback` needs the outbox from exercise 2
  to know what it can still undo.

`interrupt()` comes close behind, because resuming after an hour needs the
deterministic replay the lesson lists as a requirement.

**Defer, with a trigger.** With a single worker, leases and double-texting
cannot occur, and FastAPI + Postgres plus the outbox covers what is left:
Bedi's rule. The trigger to adopt is the second worker, or the first
human-in-the-loop wait longer than a request. Both are exactly the points
where the hand-rolled version starts needing the hard half.

### 5 — the admin queue needs a batch and gets one message per three steps

The mechanism is in the paper's §2.2 ("Hierarchical Task Management"; §3 is
Experiments). Its largest run is 590 agents; the paper does not claim the
"thousands of concurrent agents" the lesson attributes to it. The sketch below
is built from the reference `AgentQueue` at that size, as 59 groups of 10.

**Two families, one routing rule.**

| family | carries | who may send |
|---|---|---|
| `group.<g>.<agent>` | intra-group chat | members of group *g* |
| `admin.<g>` | inter-group chat | admins, and the members of *g* for outbound |

Ordinary agents cannot address another group. A cross-group message takes 3
hops: sender's admin, receiver's admin, receiver. An intra-group message takes
1. An all-hands round costs 8,732 deliveries (5,310 inside groups plus 3,422
between admins) against 347,510 for a flat mesh, 40x fewer. The saving comes
from admins summarising their groups instead of forwarding every message.

**The reference queue drains one message per three steps.** `AgentQueue.step`
pops a single message per Processing state and spends a step in each of the
three states. An admin holding one summary from each of the other 58 groups
therefore needs **174** steps to clear its inbox. MegaAgent's Processing state
takes "the message batch", and the same queue with a batch Processing state
needs 3. The batch matters most in the admin family, where fan-in
concentrates.

**The Response state does nothing, and no code reads `out_queue`.** The reply
is appended during Processing, and the Response branch is a single assignment
back to Idle. In the paper, Response verifies outputs "before being dispatched
through designated function call". No code in the module consumes
`out_queue`, and after the lesson's demo both replies are still in it. In the
two-family mapping the router is that consumer, which makes the router the
one component the reference leaves out.
