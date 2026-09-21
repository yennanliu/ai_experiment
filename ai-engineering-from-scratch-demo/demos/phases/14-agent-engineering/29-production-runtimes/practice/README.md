<!-- generated:start -->
# 14-agent-engineering / 29-production-runtimes

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/29-production-runtimes/) · upstream spec
`phases/14-agent-engineering/29-production-runtimes/docs/en.md`

```bash
uv run demo practice run 29-production-runtimes --ex 1
uv run demo explain 29-production-runtimes --ex 1
uv run pytest demos/phases/14-agent-engineering/29-production-runtimes
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Port your Lesson 01 ReAct loop to all six shapes in your stack. Which shape fits which produc… | code | T0 | `ex01_four_of_the_six_shapes_ship_and_one_of_them_prints.py` |
| 2 | Add a DLQ to the queue-based demo. Simulate 10% job failure; surface DLQ size. | code | T0 | `ex02_the_fail_policy_is_consulted_twice_and_the_draws_differ.py` |
| 3 | Write a cron-triggered eval agent that runs nightly against your top 20 traces from the day. | code | T0 | `ex03_the_scheduled_shape_is_a_print_statement.py` |
| 4 | Implement streaming with backpressure: if the client is slow, pause the agent. How does this… | code | T0 | `ex04_a_generator_already_has_backpressure_and_no_budget.py` |
| 5 | Read Claude Managed Agents docs. When would you move a self-hosted long-horizon agent to mana… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — four shapes ship, one of them is a print statement

The lesson names six shapes and the module implements four. Scheduled is a list
of `(time, event)` pairs that `main` prints — nothing fires — and durable
execution is absent entirely: the module defines none of `durable`, `scheduled`,
`checkpoint` or `cron`.

Porting Lesson 01's real loop into all six gives a 12-step run and one distinct
answer across the ports, which is the right baseline: the shapes differ in who
waits and in what survives, not in what the agent computes. The fit follows from
two numbers per shape.

**What the caller sees.** `request_response` returns `steps[-1]`, so the caller
gets 1 of 12 steps where streaming yields all 12. The shape is not only a latency
decision — it silently decides how much of the trace escapes the process. That is
the lesson's "opaque background work" pitfall built into the interface rather than
into the deployment, and it is why request-response needs OTel spans (Lesson 23)
more than streaming does, not less.

**What a failure costs.** Killed at step 7 and restarted, request-response,
streaming, queue and event each spend 12 steps reaching the answer again; a
durable port replays from its checkpoint and spends 5 — a 2.4x difference on a
12-step run, and linear in run length beyond that. That is the entire argument for
durable execution, and the lesson's "skipping durable state" pitfall priced.

One reason the shipped shapes look interchangeable: `_agent_fn` returns 5 lines
for every input, 3 of them byte-identical whatever the input, and calls nothing.
All four shapes do the same constant work. Porting the real loop is what makes
the step count depend on the task, and therefore what makes the shape choice
measurable at all.

So: request-response for sub-30-second work where the caller is a human waiting;
streaming when that human should see progress; queue and event when nobody is
waiting; durable when restarting is unaffordable; scheduled when nobody triggered
it. Which is the lesson's own table — the porting exercise is how you check that
your run is actually as short as you assumed.

### 2 — the DLQ ships, and the simulation finds a double-draw bug

`QueueRuntime` already has `dlq`, already appends to it, and `main` already prints
`len(rt.dlq)`. So the work is the simulation, and running a **random** 10% failure
through the shipped worker is where it gets interesting.

1000 jobs at 10% failure produce **108 DLQ entries** — 10.8% — where three
independent 10% draws predict 0.1%, about one job. The retry budget is doing
nothing.

The cause is that `worker` evaluates `fail_policy(job)` *twice* per iteration:
once for the retry branch and once for the DLQ branch. A random policy answers the
two calls independently, so a job that draws False then True goes straight to the
dead-letter queue on its first attempt with two of its three attempts unused. 100
of the 108 entries are exactly this. Caching one draw per attempt — the semantics
the code reads as having — takes the DLQ to 1, which is the number the policy was
designed to produce. The lesson's "no DLQ" pitfall is about failed jobs vanishing;
this is the subtler version, where the DLQ works and the retry policy silently
does not.

Two more things the simulation surfaces. Retries are appended at the tail, so a
failed job's wait grows with queue depth rather than with its own attempt count:
the retried jobs finish at positions 1001 to 1108, after 100% of the original
work. That is fine at depth 1000 and a multi-hour tail at depth 100 000 — a
separate retry queue, or a delay with a scheduled re-enqueue, bounds it. And
`QueueRuntime` declares `fail_rate` while `worker` references only `queue` and
`dlq`: the field where the exercise's "10%" would naturally live has no behaviour,
so the rate has to be smuggled in through the `fail_policy` closure.

### 3 — the scheduled shape is a print statement, and "top 20" is the whole answer

The module's scheduled runtime prints `(time, event)` pairs and `EventBus` has no
clock, so the cron is built on top of the shipped bus with a virtual clock — a
real one would make the result a fact about when the test ran.

Over a simulated week the job fires 7 times, selects the 20 lowest-scoring traces
of 150 per day, and evaluates 140 in total: mean 0.07 on the selected slice
against 0.50 across all 1050 traces.

The headline finding is about the exercise's own phrasing. "Top 20" is ambiguous
between *worst* and *most recent*, and the two readings are not close: worst-20
reports 0.07, random-20 reports 0.49, most-recent-20 reports 0.49 — a 7.0x spread
on the same agent and the same day. A nightly eval number means nothing without
the slice rule stated beside it, and the spread is larger than any regression the
eval exists to catch. Worst-20 is the right choice for *debugging* and the wrong
one for *trend*, because it is a fixed-size sample of the tail and will read 0.07
forever regardless of whether the agent improved.

Two structural gaps. A missed night is lost, not deferred: the cron fires on a
clock match, so skipping 2 of 7 nights leaves 40 traces never evaluated while the
weekly mean stays at 0.07 — the metric cannot tell a good week from a
half-measured one. That is why the lesson says to combine scheduling with durable
execution: the durable part is the backlog. And `EventBus.publish` runs handlers
in order with no isolation, so with three subscribers on the nightly event and the
first raising, neither of the other two runs and the exception propagates. A
scheduled runtime that fans out to several jobs needs each failure contained to
one subscriber.

### 4 — a generator already has backpressure; the budget is the part that is missing

`streaming` is a Python generator, so the agent is already paused between yields:
pull-based streaming *is* backpressure. Driving it with a client that consumes one
chunk per 3 ticks stretches the 12-step run to 36 ticks, with the producer idle 24
of them and the buffer never exceeding 1. The pause is the `yield`, and it costs
nothing to implement.

The case that needs work is the push variant. Replacing the pull with a push into
a 4-slot queue drops 5 of 12 chunks; gating the producer on buffer depth drops 0.
That is the version worth writing, because an SSE or WebSocket sender is
push-shaped and the socket buffer is the queue.

The interaction with a turn budget is the interesting half, and the answer is that
they measure different things. `max_turns` counts model calls, so a run idle for 24
ticks spends no extra budget and finishes inside 10 turns while taking 3.0x the
wall time. Backpressure converts a cost problem into a latency problem, and a
budget denominated in turns cannot see the latency it creates. Worse, it makes the
turn budget the wrong guard entirely: the run that needs stopping — 36 ticks for
12 steps — passes every turn check, while a 20-tick deadline cuts it after 6 of 12
steps and lets the unpaused run finish in 12 with 8 ticks to spare.

So a streaming agent needs both: a turn budget bounding what it will spend, and a
deadline bounding how long it will hold the connection. The module can express
neither — `streaming` takes one argument and yields until exhausted, and zero
fields across `QueueRuntime` and `Job` name a budget, a limit or a deadline. The
only cap anywhere in the file is the literal `3` inside `worker`.

### 5 — move to managed when the run outlives the thing you would have to operate

**2026 deployment patterns** lists the options this decision sits between: CrewAI
Flows for event-driven production, Agno's stateless FastAPI for Python
microservices, Mastra's server adapters for embedding, Pipecat Cloud / LiveKit
Cloud for managed voice, and **Claude Managed Agents for hosted long-running
async**. The lesson also notes that Claude Managed Agents covers event-driven
triggers out of the box.

The question is when to move a self-hosted long-horizon agent onto that. Four
conditions, each of which this lesson's exercises make concrete.

**When durable execution is the requirement and you have not built it.** Exercise
1 measures the cost of not having it: a run killed at step 7 costs 12 steps to
redo against a durable port's 5, and the gap is linear in run length. For a
long-horizon agent — Anthropic's computer-use announcement puts these at
dozens-to-hundreds of steps — that is the difference between a retry and a
restart. Building checkpointing correctly is not hard; building it correctly
*and* operating the storage, the resume semantics and the poison-message handling
is, and that operational half is what a managed runtime sells.

**When the queue is the product and you keep finding bugs in it.** Exercise 2 is
the cautionary tale: a hand-rolled worker with a correct-looking DLQ produced a
10.8% dead-letter rate because a policy was consulted twice. That bug is invisible
in review, survives a passing demo, and is exactly the class of thing a managed
runtime has already hit. If your queue logic is not a differentiator, the bugs in
it are pure cost.

**When the trigger surface is broad.** Exercise 3 needed a clock the module did
not have, and found that `EventBus.publish` has no failure isolation between
subscribers. Cron, webhooks, email, PR events — each is a small amount of code and
a permanent amount of operations. Managed covers this "out of the box" per the
lesson, and the out-of-the-box part is the on-call rotation, not the API.

**When nobody is waiting.** The shapes that suit managed hosting are exactly the
ones with no synchronous caller: queue, event, scheduled, durable. Exercise 4's
finding is the counterweight — a streaming agent holding a socket has a deadline
and a client, and moving it behind a managed async runtime changes the product,
not just the deployment. Voice (Pipecat Cloud / LiveKit Cloud) is listed
separately for that reason.

**When not to move.** If the run is under 30 seconds, request-response on your own
stack is simpler and observably so. If the agent's value is in tools that reach
private infrastructure, hosting the loop elsewhere moves the network boundary to
the worst possible place. And if you cannot export traces from the managed runtime
into your own backend, you have traded an operations problem for a debugging
problem — which, per the lesson's "observability is load-bearing" section, is not
a trade worth making: without GenAI spans and a Langfuse/Phoenix/Opik backend you
cannot debug a multi-step agent that failed at step 40, and that constraint does
not relax because someone else runs the worker.
