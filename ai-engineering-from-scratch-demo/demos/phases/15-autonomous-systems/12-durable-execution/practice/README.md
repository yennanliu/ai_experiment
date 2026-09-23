<!-- generated:start -->
# 15-autonomous-systems / 12-durable-execution

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/15-autonomous-systems/12-durable-execution/) · upstream spec
`phases/15-autonomous-systems/12-durable-execution/docs/en.md`

```bash
uv run demo practice run 12-durable-execution --ex 1
uv run demo explain 12-durable-execution --ex 1
uv run pytest demos/phases/15-autonomous-systems/12-durable-execution
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Observe the difference in activity-execution count between naive retry an… | code | T0 | `ex01_the_durable_count_does_not_depend_on_the_crash_point.py` |
| 2 | Convert the toy engine to use `thread_id` explicitly. Simulate two concurrent sessions sharin… | code | T0 | `ex02_two_sessions_on_one_log_replay_each_other.py` |
| 3 | Take one activity in the toy engine. Introduce a non-determinism (a wall-clock timestamp insi… | code | T0 | `ex03_the_clock_is_read_where_the_log_cannot_see_it.py` |
| 4 | Read the LangChain "Runtime behind production deep agents" post. List every state that the ru… | explain | T0 | prose, below |
| 5 | Design a checkpoint policy for a 6-hour autonomous coding task. Where do you checkpoint? What… | code | T0 | `ex05_eleven_segments_because_thirty_five_minutes_squares.py` |
<!-- generated:end -->

## Answers

### 1 — the durable count does not depend on the crash point

The exercise's last clause predicts the replay count moves with the crash
point. Run at both crash points:

| crash after | naive total starts | durable total starts |
|---:|---:|---:|
| activity 1 | 4 | **3** |
| activity 2 | 5 | **3** |

**The durable total is invariant.** It equals the number of distinct
`(name, args)` pairs the workflow calls — three — and a replay adds none. The
naive total is the crash point plus three, because every attempt re-executes
everything before it. That is the real shape of the saving: durable cost is
flat in the number of crashes, naive cost is linear.

**Replay is keyed by arguments, not by position.** `EventLog.lookup` matches
on name, args and `status == "done"`, so calling `fetch_docs("x")` twice in
one pass yields one start, two events, and a `[replay]` line on the *first*
run — before any crash. That is memoization. A real engine keys by sequence
position, because a workflow is allowed to call the same activity twice and
mean it: two identical `write_report` calls should write two reports.

**A crash inside an activity leaves an orphan the metric counts.** `lookup`
requires a `done` event, so the activity correctly re-runs — but the
`started` event from the failed attempt stays in the log forever, and
`count_runs` counts every `started`. After one mid-activity crash and a
successful retry the log holds three events, two of them starts. The number
the exercise asks you to observe is inflated by exactly the number of crashes,
which is a problem for a metric whose job is to show that crashes are cheap.

**And the log is rewritten whole on every append** — read the file, append in
memory, dump it back, with no temp file, no rename and no fsync. Writing the
*n*th event costs O(n), the workflow costs O(n²), and every event is a window
in which a crash truncates the log that exists to survive crashes.

### 2 — two sessions on one log replay each other

Baseline first, because "confirm they do not collide" is only meaningful
against a case where they do:

| | run lines | replay lines | total starts |
|---|---:|---:|---:|
| two sessions, shipped engine | 3 | 3 | **3** |
| two sessions, thread-keyed | 6 | 0 | **6** |

**Session B executes nothing.** It is handed session A's report and every one
of its activities prints `[replay] … (from log)` — the same line the
crash-recovery path prints. Nothing errors and nothing warns.

**The collision is silent and looks like a feature.** The only signal that
would distinguish it is that a replay happened before any crash, and the
engine does not record that: an event carries four keys — name, args, status,
result — and none of them is a run, an attempt or a time.

**The key is the whole conversion.** `lookup` matches `name` and `args`;
folding a `thread_id` into the event and into that comparison is the entire
change, and it works because the cache key was already the identity of the
call. The other valid answer is one file per thread, which the single-field
`EventLog` already allows — it is simpler and it scales worse, because the
cross-session queries that motivate a shared backend (which sessions are
waiting on a human?) become a directory walk.

**And sharing the log is the only concurrency the engine has.** No lock, no
sequence number, no writer coordination — `append` reads the whole file and
writes it back. Two genuinely concurrent sessions do not merely share a cache,
they lose events: the second writer's read-modify-write silently discards
anything the first wrote in between. Thread-keying fixes the *cache* collision
and leaves the *write* collision untouched, which is worth being explicit
about, since the exercise's wording invites treating them as one problem.

### 3 — the clock is read where the log cannot see it

The exercise says "inside a workflow decision", and that placement is the
point — the same read inside an *activity* replays fine. Both, run twice
against a surviving log:

| clock placement | starts after pass 1 | after pass 2 | results equal? |
|---|---:|---:|---|
| in the workflow | 3 | **4** | no |
| in an activity | 4 | 4 | yes |

**A pure replay executed a new activity.** The second pass has every prior
activity in the log and should do nothing; because the clock advanced, it took
the other branch, called an activity that was never recorded, and ran it. The
divergence is not subtle once measured — it is a run that was supposed to be a
read.

**The engine records arguments, not decisions.** An event carries name, args,
status and result, so a branch taken on a value the workflow computed itself
leaves no trace at all. `lookup` can tell you what an activity returned; it can
never tell you which call the workflow was *about* to make. That is why the
divergence is invisible until the results differ — and in a real workflow the
results often differ in ways nobody diffs.

**The fix is a decorator the engine already has.** Wrapping `time.time()` in
`@activity("now")` gives it an event, a recorded result and a replay hit — one
decorator, zero engine changes. That is precisely what `Workflow.now()` is in
Temporal and what side-effect registration is in LangGraph: the value is
written to the log on the first pass and read back on every replay, so the
workflow code stays straight-line and the non-determinism becomes data.

**And argument-keyed lookup makes the recorded clock a single value.**
`now()` called twice with no arguments returns the same recorded instant on
the first pass, because the cache key is `(name, args)` and both calls have
the same key. Real engines expose a replay-safe `now()` *and* a monotonic
sequence for exactly this reason — a workflow that needs two different
timestamps cannot get them from a cache keyed on its arguments, and the toy
engine has no way to express the difference.

### 4 — what the runtime persists, and what each covers

*Draws on "Checkpoints keyed by `thread_id`".*

The runtime persists four things, and they cover four different failures.
**The event log of completed steps**, keyed by `thread_id` — this is what
replay reads, and it covers process death: a crash loses only the incomplete
step, which is the whole result of exercise 1. **The current graph state**
(the channel values between nodes) — this covers resume after a deploy or a
host move, because a new process can reconstruct where the workflow was
without re-executing anything. **The pending-human-input state** — a
first-class "waiting on approval" checkpoint, which covers the failure mode
where an overnight approval arrives and there is no longer a process holding
the request in memory; without it, human-in-the-loop is best-effort and every
approval race loses. And **the thread's message history** — which covers
context reconstruction, since the model needs the conversation, not just the
activity results, to continue coherently.

The backend choice is a fifth thing the post treats as persistence policy
rather than mechanism: PostgreSQL survives deploys and is queryable across
threads; SQLite loses data across hosts, which makes it a dev-only answer;
Redis is ephemeral unless AOF is configured, so it covers latency and not
durability. The failure mode each *does not* cover is the one exercise 5 is
about — none of them records whether an activity's side effect reached the
outside world, so all four can be intact and the resume still cannot tell
"already pushed" from "already computed".

### 5 — eleven segments, because thirty-five minutes squares

**Where to checkpoint** has two answers at different scales. At the engine's
scale: every activity boundary, which is what the shipped `@activity`
decorator already does. At the policy's scale, the number has to come from the
reliability profile, and the lesson supplies it — decay past ~35 minutes, with
doubling the duration quadrupling the failure rate.

| segment ceiling | segments in 6 hours | failure rate per segment |
|---:|---:|---:|
| 17.5 min | 21 | 0.2× |
| **35 min** | **11** | **0.9×** |
| 70 min | 6 | 4.0× |
| 360 min (one stretch) | 1 | **105.8×** |

Six hours is 10.3× the 35-minute mark, and the square of that is the 105.8.
**Eleven segments** is the policy.

**The quadratic is what makes the number small** — and what makes tuning it
cheap in one direction and expensive in the other. Risk moves by the square of
the segment length while the human cost moves linearly, so halving the segment
buys a 4× risk reduction for 10 extra boundaries. That asymmetry is the whole
argument for short segments, and it is why "just run it overnight and check in
the morning" is a 100× decision rather than a convenience.

**Resume-on-crash** is the engine's replay: completed activities return from
the log, only the incomplete one runs. Cheap in activities, expensive in file
reads — each `lookup` calls `events()`, which parses the whole log, so
replaying a 360-activity segment parses about **129,960** event records to
execute one. The engine that exists to make long runs affordable gets
quadratically slower the longer they get, which is an argument for the
segment boundary being a *new log* rather than a marker in the old one.

**What requires fresh HITL:** every segment boundary, and any activity with a
side effect before it re-runs. The first is because reliability decay resets
only if the context does — a resumed run that carries 5 hours of history has
5 hours of drift in it, and the checkpoint is where a human decides whether
the plan is still the plan. The second is because the log cannot answer it:
one of the three shipped activities is commented as having a side effect, an
event carries four keys, and none of them records whether one occurred. On
re-entry the engine cannot distinguish "already pushed" from "already
computed" — and that distinction is exactly what the human is being asked
about.
