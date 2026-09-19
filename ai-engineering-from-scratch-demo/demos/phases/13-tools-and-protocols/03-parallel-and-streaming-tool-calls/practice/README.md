<!-- generated:start -->
# 13-tools-and-protocols / 03-parallel-and-streaming-tool-calls

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/03-parallel-and-streaming-tool-calls/) · upstream spec
`phases/13-tools-and-protocols/03-parallel-and-streaming-tool-calls/docs/en.md`

```bash
uv run demo practice run 03-parallel-and-streaming-tool-calls --ex 1
uv run demo explain 03-parallel-and-streaming-tool-calls --ex 1
uv run pytest demos/phases/13-tools-and-protocols/03-parallel-and-streaming-tool-calls
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py` and vary the simulated latencies. Confirm that the parallel-to-sequential… | code | T0 | `ex01_parallel_stops_mattering_when_one_call_dominates.py` |
| 2 | Extend the accumulator to handle a "call was cancelled mid-stream" case by dropping its buffe… | code | T0 | `ex02_the_accumulator_has_no_branch_for_anything_going_wrong.py` |
| 3 | Replace the thread pool with `asyncio.gather`. Benchmark both. You should see small wins on a… | code | T0 | `ex03_asyncio_loses_because_the_lesson_executor_does_no_io.py` |
| 4 | Pick two tools that should NOT parallelize (e.g. `create_file` then `write_file`). Add an `or… | code | T0 | `ex04_there_is_no_registry_here_and_no_fan_out_that_could_read_one.py` |
| 5 | Read OpenAI's parallel-function-calling section and Anthropic's `disable_parallel_tool_use` d… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

All five are T0 and stdlib. Four ship code; exercise 5 is prose, below, citing
the lesson section it rests on.

Exercises 1 and 3 are timing questions and 2 and 4 are structural ones, and both
pairs land on the same thing: **the exercises are written against a richer
lesson than the one they are attached to.** Exercise 4 asks you to modify a
registry this lesson does not have, and exercise 3's own escape clause — "only
if executors do real I/O" — is not met by the executor it is set on.

### 1 — parallel stops mattering when one call dominates

Speedup is `sum/max`, so it is capped at the number of calls and reached only
when they are equal:

| latencies | ideal speedup |
|---|---:|
| 500 / 500 / 500 | **3.00×** |
| 400 / 600 / 800 (the lesson's) | 2.25× |
| 800 / 50 / 50 | 1.12× |
| 1000 / 10 / 10 | **1.02×** |

**ANSWER: the ratio is `max/sum`, and it is scale-free.** Doubling every latency
leaves it unchanged, so the crossover is a property of the *spread*, not the
magnitude. Below a 1.1× speedup the fan-out is not worth its machinery; with two
10 ms companions that threshold is crossed once the third call passes **200 ms**.

**FINDING: the deviation from ideal is a fixed cost per call, not a percentage.**
Measured at 1/10 scale, the sequential arm runs ~15 ms over its 180 ms floor and
the parallel arm ~6 ms over its 80 ms. Neither can undercut its floor, because
`time.sleep` only overshoots — and the overhead is paid three times by the arm
being divided and once by the divisor, which is why the measured ratio lands
slightly *above* ideal rather than below.

**FINDING: the tool the lesson parallelises is not replayable.**
`executor_weather` returns `hash(city) % 35`, and Python salts `hash()` per
process: Bengaluru read 5 °C, then 2 °C, then 6 °C across three runs. Within a
process it is stable, so the two arms agree and the lesson's comparison is sound
— but no test can pin the value, and a cache keyed on the city would be wrong the
moment the process restarts.

### 2 — the accumulator has no branch for anything going wrong

**ANSWER: a `cancelled` event drops the buffer; the lesson has nowhere to put
it.** `on_event` branches on `call_start`, `args_delta` and `call_stop` and
returns `[]` for everything else, so feeding it a cancellation is accepted
silently and leaves the buffer in place. The subclass here pops it and records
the id.

**FINDING: an unfinished call is not an error, it is a leak.** Start a call, send
one `args_delta`, stop: the buffer stays with `done=False`, `try_parse()` returns
`None`, `replay_and_execute` never submits it, and nothing raises or logs. That
is exactly the state a cancellation leaves behind — so the work the exercise asks
for is *detection*, not cleanup.

**FINDING: the one thing that does raise is the one that should not.** An
`args_delta` for an id that never had a `call_start` raises `KeyError` straight
out of `on_event`, killing the loop and taking the other two in-flight calls with
it. A dropped first frame is a recoverable stream error and an unfinished call is
a silent one. The accumulator has the two backwards.

**ANSWER: neither provider documents cancellation — they document truncation.**
Anthropic's `content_block_stop` always arrives for a block that started, so its
absence is a transport failure rather than a signalled cancel. OpenAI's
`finish_reason: "length"` marks a stream the *model* stopped filling, leaving
valid-prefix JSON that `json.loads` rejects — `Unterminated string starting at`
on the lesson's own fixture cut to `{"city":"Tok`. Both say a call will never
complete; neither says a caller asked it to stop. That is a client-side concept,
which is why it is MCP's `notifications/cancelled` that names it.

### 3 — asyncio loses, because the lesson's executor does no I/O

| arm | 40/60/80 ms calls |
|---|---:|
| `ThreadPoolExecutor` | ~86 ms |
| `asyncio.gather` over `asyncio.sleep` | ~84 ms |
| `asyncio.gather` over the lesson's executor | **~196 ms** |

**ANSWER: both working arms hit the same 80 ms `max` floor**, a few milliseconds
apart. The cost being divided is a sleep, and both mechanisms overlap sleeps
perfectly.

**FINDING: `asyncio.gather` over `time.sleep` is not concurrent at all.** It
reaches the 180 ms *sequential* ceiling. The coroutine never awaits, so the event
loop has no point at which to interleave and the three sleeps run one after
another.

**FINDING: so the exercise's condition is the whole result.** "Only if executors
do real I/O" is doing all the work here. `time.sleep` releases the GIL, so
threads overlap it; `await` on a non-awaiting coroutine does not yield, so the
loop cannot. Against this executor the thread pool is not slightly better than
async — it is the only one of the two that works.

**FINDING: and the fix makes the two identical, not async-favourable.** The
context-switch saving the exercise predicts is real and far below the resolution
of a three-call fan-out. It is a throughput argument for thousands of concurrent
calls, not a latency argument for three.

### 4 — there is no registry here, and no fan-out that could read one

**ANSWER: `create_file → write_file → read_file`, gated by a topological
layering.** Tools with an edge land in different layers, independent ones share
one; the scheduler fans out per layer and waits. A chain of 3 costs **3**
sequential windows, three independent tools cost **1**.

**FINDING: this lesson has no registry to add the graph to.** Its namespace holds
`executor_weather`, `run_sequential`, `run_parallel`, `StreamAccumulator`,
`CallBuffer` and `fake_openai_stream` — **no `Tool`, no `REGISTRY`**. Lesson
13.01 has both.

**FINDING: and there is no fan-out that could consult one.** `run_parallel`'s
code names only `executor_weather` — it maps one function over a list of
arguments, with no tool name in it anywhere. Gating that on a dependency graph
would require the calls to *be* different tools first, which is what 13.01's
registry provides and this lesson dropped.

**FINDING: the gate costs the whole speedup exactly when it is needed.** On the
lesson's own 400/600/800 latencies, three independent tools run in **800 ms** and
a three-long chain in **1,800 ms** — the sequential time, a **2.25×** loss, which
is `sum/max` from exercise 1 again. A dependency-aware scheduler does not make
ordering cheap. It makes the unsafe version unavailable.

### 5 — the tool type is a mutation on one resource

*Cites "Enabling parallel".*

**The tool type is a consequential write against a single shared resource — and
the canonical case Anthropic gives is a file or record that two calls would both
modify.** The lesson's own line names the condition directly: disable parallel
"when tools have ordering dependencies (`create_file` then `write_file`), when
one call's output informs another's input, or when the rate limiter cannot handle
fan-out."

Three conditions, and they are not equally serious. Exercise 4 measures why.

The rate-limiter case is a *throughput* problem: fan-out is still correct, you
just cannot afford it, and the fix is a semaphore rather than `disable_parallel_tool_use`.
The output-informs-input case is a *planning* problem: the model cannot emit the
second call until it has seen the first result, so parallelism was never on the
table — the loop serialises it for you by construction.

Only the first is a *correctness* problem, and it is the one that fails silently.
Two calls that mutate the same resource both succeed, both return, and the
surviving state depends on which thread finished last. Nothing in the four-step
loop notices: exercise 4's scheduler shows the calls carry no tool identity at
the fan-out point, so the host has nothing to key an ordering decision on, and
exercise 1 shows the parallel arm and the sequential arm return byte-identical
results for tools where order does not matter — which is exactly why the case
where it does matter leaves no trace.

That is the argument for `disable_parallel_tool_use: true` being a per-request
flag rather than a per-tool one. You are not marking a tool unsafe; you are
marking a *turn* in which the model might emit two calls whose interleaving you
cannot inspect after the fact. Anthropic's framing puts the switch where the
uncertainty is, which is the same reasoning behind Lesson 13.01's consequential
gate: the host confirms before the write, because after the write there is
nothing left to confirm.
