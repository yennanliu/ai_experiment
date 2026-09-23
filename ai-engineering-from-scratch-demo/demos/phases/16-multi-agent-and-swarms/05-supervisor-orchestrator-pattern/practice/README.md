<!-- generated:start -->
# 16-multi-agent-and-swarms / 05-supervisor-orchestrator-pattern

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/05-supervisor-orchestrator-pattern/) · upstream spec
`phases/16-multi-agent-and-swarms/05-supervisor-orchestrator-pattern/docs/en.md`

```bash
uv run demo practice run 05-supervisor-orchestrator-pattern --ex 1
uv run demo explain 05-supervisor-orchestrator-pattern --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/05-supervisor-orchestrator-pattern
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`, then modify the lead to spawn 5 workers instead of 3. Observe the wall-cl… | code | T0 | `ex01_the_crossover_does_not_exist_because_the_work_is_a_sleep.py` |
| 2 | Implement a worker timeout: kill any worker that runs longer than 0.5 seconds and have the le… | code | T0 | `ex02_nothing_is_killed_and_the_cut_worker_answers_afterwards.py` |
| 3 | Add a conflict-detection step to the lead's synthesis: if two workers return contradictory an… | code | T0 | `ex03_the_simulator_cannot_produce_the_contradiction_it_asks_you_to_find.py` |
| 4 | Read Anthropic's Research-system engineering post. List three practices that this toy demo wo… | explain | T0 | prose, below |
| 5 | Compare LangGraph's `create_supervisor` (legacy) vs the new tool-calling recommendation. Whic… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — the crossover does not exist because the work is a sleep

Never, at any worker count.

| N | parallel wall clock | sequential baseline |
|---:|---:|---:|
| 3 | ~0.30s | 0.9s |
| 5 | ~0.30s | 1.5s |
| 10 | ~0.30s | 3.0s |
| 50 | ~0.31s | 15.0s |

A thread costs tens of microseconds to start and join; `fake_web_fetch`
sleeps **0.3s**, four orders of magnitude more. Parallel time is
`0.3 + N × spawn` against a baseline of `0.3 × N`, so overhead could only
overtake savings if spawning a thread cost more than 0.3s. The gap does not
narrow with N — it widens.

The reason the question has no answer is the stand-in for the work. `sleep`
releases the GIL and models a task that costs no CPU at all. Replace it with
arithmetic and the demo's headline evaporates:

```
three threads   0.069s
three in series 0.041s
a real 3× win   0.014s
```

Threading it is *slower*. The measurement has to be counted in operations
rather than wall clock, incidentally — a burn loop with a wall-clock deadline
lets each thread quietly do less work and hides the contention entirely.

So the supervisor win as demonstrated is a property of `time.sleep`, not of
the pattern. A real worker that fetches *and parses* sits between the two arms,
and where it sits is the number this demo cannot produce.

Two smaller things. `plan()` never reads the query: it returns three fixed
suffixes — historical origins, state of the art 2026, open problems — for any
input, so "spawn 5 workers" means editing a list literal. And the demo prints
`stats["wall_clock_seconds"]`, which is measured, directly above `~0.9s` and
`~0.35s`, which are string literals: change the worker count as the exercise
instructs and the measured number moves while both claims about it stay put.

### 2 — nothing is killed, and the cut worker answers afterwards

**What observability you need:** a third event and a terminal status. A worker
logs exactly two event kinds, `start` and `done`, and `TraceEntry` carries four
fields — `worker_id`, `event`, `t`, `sub_question` — none of which is a status
or a duration. A cut worker has logged `start` and nothing else, which is
byte-for-byte what a slow worker still in flight looks like. Nothing short of a
`timeout` event plus a terminal status separates them.

But "kill any worker" cannot be implemented as written. `Thread` has no
terminate, so `join(timeout)` stops the *lead waiting* and does not stop the
worker:

```
deadline passes   -> cut worker is_alive() == True
lead synthesizes  -> 2 of 3 results
0.3s later        -> the results list holds 3
```

The lead publishes an answer, and the state it published from then changes
underneath it. In a longer-lived process that list is the audit record.

The output does not carry the hole either. `synthesize` filters
`r is not None`, so a three-worker run with one cut produces text
**byte-identical** to a two-worker run that only planned two sub-questions.
The same absence the trace cannot report, the synthesis cannot report either.

And the timeout does not save what it was spent to save: `tokens_spent` is the
literal `800`, assigned *after* the fetch returns, so the abandoned worker
records its full cost once it finishes — which it does. Cutting a worker
shortens the answer, not the bill.

### 3 — the simulator cannot produce the contradiction it asks you to find

**How to detect contradiction without an LLM:** compare quantities that share a
subject. Extract `(number, noun)` pairs from each summary, group by noun, and
flag any noun whose values are not all equal. On a labelled set of eight pairs
it scores **8/8** — four contradictions found, four agreements left alone.

It catches exactly the disagreements that are *commensurable*. "Latency fell to
120 ms" versus "latency fell to 340 ms" is a conflict; "latency fell to 120 ms"
versus "throughput rose to 340 rps" is not. That boundary is the honest limit
of a model-free detector, and it is worth stating rather than implying it
catches contradiction in general.

Then run it on the lesson's own workers and it finds nothing, because there is
nothing to find. `fake_web_fetch` is one template:

```python
return f"Summary for '{query}': 3 key findings from 5 sources."
```

The lead's three summaries reduce to **one** distinct claim set, so **0** of
the three possible pairs conflict. The exercise's premise is unreachable from
the lesson's own code, and the only way to test the detector is to inject
answers — which is what the labelled set above is for.

One accident worth keeping. The template interpolates the query, and the demo
asks about "2023 to 2026", so `2023` enters the claim set as a reported value.
The detector reads query numerals as claims. Here they agree — every
sub-question carries the same years — but sub-questions naming different years
would conflict on nothing at all.

Finally, the cost of adding the step cannot be measured here either.
`tokens_spent` is the literal 800 per worker and the lead adds a flat 1200, so
a run costs `800N + 1200` by construction. A synthesis pass that reads every
summary changes the real bill and changes this formula by zero.

### 4 — three practices, and the one this demo inverts

*Draws on "Engineering lessons (Anthropic 2025)".*

Three the toy would need to adopt:

**Scale effort to query complexity.** The post's rule is one agent and 3–10
tool calls for simple queries, 10+ agents for complex ones, with the *lead*
estimating which. `plan()` returns three sub-questions for every query, with
fixed suffixes, without reading the input. Worker count is a property of the
lead's source file. This is the single largest gap, and exercise 1 measures the
consequence: adding workers is free in this demo, so nothing pushes back on
over-decomposition.

**Token usage dominates — multi-agent is ~15× single-agent.** Here tokens are
`800N + 1200`, literals assigned after work completes. The demo therefore
cannot express the post's central caution, and exercise 2 shows the sharpest
version: a worker that gets cut still books its full 800.

**Rainbow deployments, because agents are long-running and stateful.** The
whole run is one process, one `Trace`, one in-memory results list. Exercise 2's
finding is exactly the hazard that makes rainbow necessary — a worker that
outlives the lead's attention and writes into shared state afterwards. At
production lifetimes that is a version-skew problem, not a race.

The fourth lesson, **broad then narrow**, the demo actively inverts: `plan()`
fans out once with fixed breadth and never spawns depth on a promising
sub-question, because the lead never reads what came back before synthesizing.

### 5 — control over what the supervisor sees, and why sub-answers only

*Draws on "The graph-native turn".*

**Tool-calling gives better control, and the control it gives is over
context.** `create_supervisor` wires a supervisor node with handoff tools and
a shared message list; the supervisor's view is the accumulated thread. Under
the tool-calling recommendation each worker is a tool, and a tool returns a
value — so what the supervisor sees is the return, and everything the worker
did to produce it stays in the worker. The difference is not ergonomics. In
one the worker's context leaks into the supervisor by default and you narrow
it; in the other nothing leaks and you widen it deliberately.

**Why Anthropic passes only sub-answers:** three reasons, and this module shows
two of them.

*Cost.* Raw worker context is the bulk of the tokens. Synthesis over N raw
transcripts grows the lead's context with N, and the lead's context is the one
that must hold the whole problem. Sub-answers make the lead's input a function
of the number of sub-questions rather than of how hard each one was.

*The lead is the only agent that must stay coherent.* Workers can fail, retry,
wander; the lead has one job. Admitting raw context admits every worker's dead
ends into the one context where confusion is unrecoverable.

*Auditability.* A sub-answer is a claim with an owner, which is what makes
exercise 3's conflict detection possible at all — you can compare two workers'
answers because they are commensurable artifacts. You cannot diff two raw
transcripts for contradiction without a model, which is precisely the
dependency the exercise asks you to avoid.
