<!-- generated:start -->
# 16-multi-agent-and-swarms / 09-parallel-swarm-networks

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/09-parallel-swarm-networks/) · upstream spec
`phases/16-multi-agent-and-swarms/09-parallel-swarm-networks/docs/en.md`

```bash
uv run demo practice run 09-parallel-swarm-networks --ex 1
uv run demo explain 09-parallel-swarm-networks --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/09-parallel-swarm-networks
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. How much faster is swarm than sequential on the variable-duration workloa… | code | T0 | `ex01_the_swarm_wins_because_the_slow_tasks_happen_to_be_queued_first.py` |
| 2 | Add a priority queue variant (use `queue.PriorityQueue`). Assign priority by task "importance… | code | T0 | `ex02_low_priority_starves_exactly_when_high_priority_alone_fills_the_workers.py` |
| 3 | Implement a hot-spot detector: log when any worker processes 3× more tasks than the slowest w… | code | T0 | `ex03_the_worker_with_the_most_tasks_is_the_least_loaded.py` |
| 4 | Read the Matrix paper (arXiv:2511.21686) abstract and Section 3. Identify one specific tradeo… | explain | T0 | prose, below |
| 5 | Convert the swarm demo to use a `queue.Queue` of (task_type, payload) tuples, with workers su… | code | T0 | `ex05_splitting_the_swarm_by_type_rebuilds_fixed_assignment.py` |
<!-- generated:end -->

## Answers

### 1 — the swarm wins because the slow tasks happen to be queued first

**4.0x faster than sequential, 3.4x faster than fixed assignment.** The
workload is 4 tasks of 0.4s and 4 of 0.1s, 2.0 work-seconds:

| strategy | wall | counts |
|---|---:|---|
| sequential | 2.0s | {0: 8} |
| fixed assignment | 1.7s | {0: 5, 1: 1, 2: 1, 3: 1} |
| swarm | 0.5s | {0: 2, 1: 2, 2: 2, 3: 2} |

Worker 0 holds **5** fixed tasks, not the 4 the demo's comment implies: task 7
is fast and `(7 - 3) % 4 == 0`. The reference schedulers, with every sleep
scaled by 1/4, reproduce all three times.

Both ratios depend on choices `make_tasks` makes rather than on anything the
queue does.

**The swarm hits the ideal only because the queue is sorted longest-first.**
The four slow tasks are enqueued before the four fast ones, so each worker
takes one slow task at t=0. That is LPT scheduling. Over the **70** distinct
orders of the same 8 tasks the swarm reaches 0.5s on **16**, takes 0.6s on 34,
0.7s on 16 and 0.8s on 4. At 0.8s the speedup over fixed assignment is 2.1x.

**The fixed baseline is built to lose** — its docstring says "Pre-assignment
is pessimal". Assigning `task_id % 4` gives every worker one slow and one fast
task and finishes in **0.5s, equal to the swarm**. So the 3.4x is a
measurement of the `pre_assigned` formula, not of supervisor against swarm.

**The counts are even, not uneven.** The lesson says the swarm "distributes
unevenly but optimally", and the counts are 2, 2, 2, 2.

### 2 — low priority starves exactly when high priority alone fills the workers

**Yes, and it is a step, not a slope.** Four workers, 0.1s tasks, a burst of
*k* high-priority tasks every 0.1s, and one low-priority task waiting from t=0:

| k | low task runs at | backlog after 100 slots |
|---:|---|---:|
| 3 | slot 0 | 0 |
| 4 | never | 1 — the low task itself |
| 5 | never | 101, growing by one per slot |

It starves exactly when the high-priority arrivals alone use every worker.

The variant needed pieces the demo lacks. `run_swarm` fills its queue before
starting any worker and each worker returns on the first `queue.Empty`, so
there is no producer and no "continuous load". `Task` has no importance field.
And `PriorityQueue` of `(priority, Task)` raises `TypeError` on the first tie,
because `Task` is a plain `@dataclass` without `order=True`. The entry needs a
sequence number in between.

**Linear aging needs no re-scoring.** A heap cannot re-rank entries after they
are pushed, which looks fatal to aging. But effective priority
`p - a·(now - arrival)` differs from the static key `p + a·arrival` only by
`a·now`, which is the same for every entry. With a = 0.1 per slot the low task
runs at slot 10 under k=4 and at slot 12 under k=5.

**Aging picks who waits; it does not shrink the wait.** At k=5 the backlog is
101 with aging and 101 without, because 400 tasks are served either way. Under
overload only back-pressure helps.

### 3 — the worker with the most tasks is the least loaded

**A 3x count gap means the durations are spread, and the worker with the
fewest tasks is the one holding the long ones.** In a pull swarm a worker only
takes fewer tasks by being busy longer. On 1 task of 1.2s and 12 of 0.1s:

| worker | tasks | busy |
|---|---:|---:|
| 0 | 1 | **1.2s** = the makespan |
| 1–3 | 4 each | 0.4s each |

The count detector fires at 4x and flags workers 1–3, the idle ones. A
busy-time detector at the same 3x flags worker 0. The reference swarm produces
the same [1, 4, 4, 4].

On the lesson's own workload, though, the count detector is a **perfect
makespan-loss alarm**. Across the 70 queue orders it fires on 54, every one of
them slower than 0.5s, and stays silent on the 16 that reach 0.5s. With two
durations, equal counts means every worker drew one slow and one fast task.

In the shipped demo it fires once, on fixed assignment's {5, 1, 1, 1}. There
the most-counted worker really is the loaded one, because work was pushed to
it. The same signal means opposite things under push and pull.

### 4 — the orchestrator became a message; the fleet-wide view went with it

*Draws on "Matrix (arXiv:2511.21686)".*

**Accepted, for scale: no orchestrator process.** Matrix's orchestrator is
not gone; it is a *serialized object* that carries one task's state,
history and control flow. §3.2: "each input datum is encapsulated into an
orchestrator instance and passed to the initial agent". Agents are Ray actors
that run an event loop: take an orchestrator from the inbox, call
`process()`, forward it to the next agent. Since the plan travels with the
row, nothing central has to decide what happens next. That enables row-level
scheduling, where a slow task no longer stalls a batch. The paper reports
2–15x throughput on the same hardware and tens of thousands of concurrent
workflows.

**Given up: delivery guarantees and a global view, but not per-task
traceability.** On a crash, the per-role broker marks the tasks held by that
agent *failed*. The paper calls this at-most-once semantics (§4.5). Nothing
reruns them, so crash recovery means data loss plus a later re-run. That
cuts against this lesson's "worker idempotency" checklist item, which assumes
a task may be pulled twice. Matrix removes the double delivery and accepts
the loss instead.

Per-task traceability is *kept*: every orchestrator records its own full
trajectory (§4.7). What is lost is any single place where cross-task order
exists. Two runs over the same input can interleave differently, and the
paper states no reproducibility guarantee. So the lesson's "determinism and
traceability for scalability" is half right: Matrix trades determinism and
cross-task ordering, while each task stays fully traceable.

### 5 — splitting the swarm by type rebuilds fixed assignment

Type the shipped 8 tasks by duration, "slow" and "fast", and give 4 workers
2 and 2 subscriptions. Each routing rule below removes a failure that this
conversion actually produces:

1. **One queue per type.** On a single shared `queue.Queue`, fast workers pop
   slow tasks and have to put them back: **14 wasted gets for 8 tasks**, and
   FIFO order is lost.
2. **Dead-letter any type with no subscriber, at publish time.** A
   `run_swarm`-style worker only exits on `queue.Empty`, so one orphan task
   keeps it spinning, here up to the 1000-get cap.
3. **Size subscriptions by work-seconds, not by type count.**
4. **Let idle specialists steal from other queues.**

Rules 3 and 4 come from what partitioning does to throughput:

| subscriptions (slow/fast) | makespan |
|---|---:|
| untyped swarm | 0.5s |
| 1/3 | 1.6s |
| 2/2 | 0.8s |
| 3/1 | 0.8s |
| 2/2 with stealing | 0.6s |

Every strict partition is at least 60% slower than the untyped swarm. The slow
type has 1.6 work-seconds and 4 tasks do not split evenly over 3 workers.
Subscribing by type is fixed assignment at the type level, and it brings back
the imbalance the lesson credits the swarm with removing. Stealing recovers
most of the loss. The remaining 0.1s is the fast pair starting on cheap work
while slow work was still waiting.
