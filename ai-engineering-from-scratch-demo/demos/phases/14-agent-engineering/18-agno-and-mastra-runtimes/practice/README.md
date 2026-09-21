<!-- generated:start -->
# 14-agent-engineering / 18-agno-and-mastra-runtimes

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/18-agno-and-mastra-runtimes/) · upstream spec
`phases/14-agent-engineering/18-agno-and-mastra-runtimes/docs/en.md`

```bash
uv run demo practice run 18-agno-and-mastra-runtimes --ex 1
uv run demo explain 18-agno-and-mastra-runtimes --ex 1
uv run pytest demos/phases/14-agent-engineering/18-agno-and-mastra-runtimes
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Read Agno's docs. Port the stdlib ReAct loop (Lesson 01) to Agno. What disappeared? What stayed? | code | T0 | `ex01_the_loop_disappears_into_the_request_and_so_does_the_budget.py` |
| 2 | Read Mastra's docs. Port the same loop to Mastra. What changed in tool typing (Zod vs nothing)? | code | T0 | `ex02_the_input_schema_is_a_field_that_dispatch_never_opens.py` |
| 3 | Benchmark: measure agent instantiation latency on your stack. Does Agno's 2μs matter to your… | code | T0 | `ex03_the_handler_times_itself_and_calls_the_number_instantiation.py` |
| 4 | Design a migration: if you've been running CrewAI in Python, what breaks if you move to Agno? | explain | T0 | prose, below |
| 5 | Read Mastra's `ee/` license terms. What restrictions would affect an open-source fork? | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

All five are T0 and stdlib. Three ship code — the two ports drive Lesson
01's own `AgentLoop`, `ToolRegistry` and `ToyLLM` through this lesson's
runtimes, so the comparison is between real objects. Exercises 4 and 5 ask
for a migration design and a licence reading, so they are answered in prose.

The recurring finding is that **each runtime keeps the part of the loop that
was already a function and drops the parts that were state**. Porting Lesson
01 to the Agno shape, the tool registry ports unchanged — **3** tools,
**5** dispatches, identical observations — while `history` changes from
`list[Turn]` with four fields to `list[str]`, and `max_turns` has nowhere to
live at all: the handler calls `agent.run` exactly once with no cap, so the
budget becomes the caller's problem and nothing replaces it.

The second thread is that **the declared types are declarations**.
`MastraTool` carries an `input_schema` and `MastraAgent.run` reads it **0**
times, binding `fn(**args)` directly — so a wrong argument name is a
`TypeError` out of the agent where Lesson 01's registry returns
`error: bad args` as an observation. The untyped runtime is the one that
survives the mistake.

What holds: the structural contrast the lesson is drawing. Agno's shape
really is one function per request with the state outside; Mastra's really
is three primitives that compose. Both ports work, which is the point.

### 1 — the loop disappears into the request, and so does the budget

**ANSWER: the trace and the turn budget disappear; the tools and the stop
condition stay.** `AgentLoop` has **4** fields and `AgnoAgent` has **2**.
Driving the same scripted policy the ported handler returns
`the total including 15% tax is 138.0` and leaves **12** turns in the session
store against **12** in `AgentLoop.history`.

**FINDING: the history moved out of the agent and changed type.**
`list[Turn]` — `kind`, `content`, `tool_call`, `observation` — becomes
`list[str]`. The same run is 12 structured turns one way and 12 strings the
other, so "what did the user say" survives the port and "which tool produced
this observation" does not. Lesson 07's `conversation_search` would still
work; Lesson 01's `tool_use_id` correlator could not be built at all.

**FINDING: the budget has nowhere to live.** `max_turns=10` bounds the loop
inside `AgentLoop`; in the Agno shape the loop belongs to the caller and
`agno_request_handler` has no cap. This is the honest cost of "each request
starts a fresh agent": statelessness moves the turn budget to whatever is
calling, and a runaway agent is now a runaway *caller*.

**FINDING: what stayed is the part that was already a function.** The
registry ports with no changes because `ToolRegistry.dispatch` never needed
the loop. That is a useful rule for any port in this phase — the ingredients
that survive a runtime change are the ones with no state in them.

### 2 — the input schema is a field that dispatch never opens

**ANSWER: the loop becomes an agent plus a workflow, and the schema becomes
a docstring.** Three tool calls give a **3**-entry trace with the same
observations, the turn sequencing moves into a **3**-step `MastraWorkflow`,
and `MastraAgent.run` reads `input_schema` **0** times.

**FINDING: a wrong argument name is a crash, not a validation error.**
`fn(**args)` binds directly, so **2** of **4** probes raise `TypeError` out
of the agent, where Lesson 01's registry returns `error: bad args…` as an
observation the model can read and retry against. The framework with the
type field has the worse failure mode, because the field is not used.

**FINDING: the schema would have caught three of four probes.** Checked, a
missing key, an unknown key and a wrong type are all rejected and one passes.
Unchecked, **2** run — including the wrong-typed one, which no exception
would ever have caught. That is the concrete answer to "Zod vs nothing": the
difference is not that Zod catches crashes, it is that Zod catches the call
that *succeeds* with the wrong value.

**FINDING: the workflow has no failure path.** `MastraWorkflow.run` threads
`current = fn(current)` with **0** try blocks, `MastraAgent.run` has **0**,
and neither has a predicate or an early exit. Of the three primitives the
lesson names, none has a validation seam — so the typing has to happen at the
tool boundary or not at all.

### 3 — the handler times itself and calls the number instantiation

**ANSWER: 2μs matters below about 200μs of per-request work and nowhere
else.** It is **0.0004%** of a 500ms model call, **0.02%** of a 10ms local
call, **1.96%** of a 100μs in-process one, and reaches **1%** of a request
only when everything else finishes in **198μs**. No wall clock was read for
this: the ratio is arithmetic anyone can recompute against their own
per-request cost, which is what makes it portable.

**FINDING: the shipped handler measures the handler, not instantiation.**
`agno_request_handler` starts its timer *after* the agent exists and times
`agent.run` plus two session appends. The microseconds in its reply string
are the whole request — so the demo's own number, read carelessly, is
evidence for the opposite of what the exercise asks about.

**FINDING: instantiation is two attribute assignments.** `AgnoAgent` is a
dataclass with **2** fields and no `__post_init__`. A number that small is a
statement about the *absence* of a constructor: framework startup benchmarks
at this scale measure how much setup a framework insists on, not how fast it
runs. The lesson says as much — "they matter less when one agent runs for 10
minutes" — and the arithmetic above puts the threshold at 200μs.

**FINDING: the session store is the part that scales with traffic.**
`AgnoSession.append` never trims, so 1000 turns across 100 sessions leave
**1002** strings resident. The lesson quotes **3.75 KiB** per agent; per
*session* the toy has no bound, and in a stateless-backend design the
sessions are the only thing that survives a request. That is the memory
number worth benchmarking.

### 4 — what breaks is everything CrewAI was doing for you

*Cites "When to pick each".*

**The migration is easy where CrewAI was a wrapper and hard where CrewAI was
a runtime.** Lesson 15's exercises measured exactly which is which, and the
answer maps onto this lesson's ports almost line for line.

**What ports in an afternoon.**

- **Tools.** Lesson 18's exercise 1 finding generalises: the tool registry
  ported to Agno unchanged because dispatch has no state. CrewAI's `@tool`
  functions are plain callables too — Lesson 15 exercise 4 measured the
  decorator setting **2** attributes — so they move as-is.
- **Agent roles.** `role + goal + backstory` is three strings. In Agno they
  become the system prompt you assemble yourself. Nothing is lost because
  nothing was enforced: Lesson 15 measured `expected_output` being read **0**
  times.
- **Sequential crews.** A task list executed in order is a `for` loop.
  Lesson 15's exercise 1 converted one to a Flow with byte-identical output;
  converting it to a request handler is the same move.

**What breaks, in descending order of how long it takes to notice.**

1. **Memory.** CrewAI's four stores behind `memory=True` become "a DB you
   configure", which is the whole premise of the Agno shape — "session state
   lives in a DB". Lesson 15 exercise 2 found the *toy's* entity store
   unwired and its long-term retrieval hashing rather than embedding, so the
   honest version of this risk is: you were relying on four stores whose
   behaviour you had not measured, and now you own all four.
2. **The hierarchical process.** A manager LLM routing to specialists is a
   CrewAI *process*, not a prompt. In Agno you write the router. Lesson 15
   exercise 3 is the warning: the manager needs to see the output to gate on
   it, and the moment you own the loop you also own the retry budget, which
   exercise 1 here shows Agno has nowhere to put.
3. **`output_pydantic`.** Lesson 15 exercise 5 built the parse-and-retry
   CrewAI gives you as a field. In Agno that is yours. In Mastra it is the
   Zod schema — which exercise 2 here shows is only worth anything if the
   runtime actually reads it.
4. **The trace.** CrewAI returns `list[str]`, one line per task. Agno returns
   a reply per request, and the history is whatever you wrote to the DB
   (exercise 1: `list[str]` with no tool attribution). Anything on-call was
   reading has to be rebuilt.

**And the decision the lesson frames it as:** pick Agno for "Python backend,
many short-lived agents, strong perf requirements, FastAPI shop". If the
reason for moving is the third of those, exercise 3's arithmetic is the check
to run first — **2μs** only matters below **200μs** of request work, and a
CrewAI crew that makes model calls is nowhere near that. Moving for
per-request overhead when the workload is model-bound is the lesson's own
"perf-for-perf's-sake" pitfall, and it costs you items 1–4 above.

### 5 — source-available is not open source, and a fork inherits the line

*Cites "Where this pattern goes wrong".*

**The restriction that matters is not what you may read, it is what you may
ship.** The lesson states the shape: Mastra is Apache 2.0 with `ee/`
directories under a source-available enterprise licence, and "read the
licenses if you're planning to fork". Three consequences for an open-source
fork follow from that split, and none of them depends on the specific
wording.

**1. The fork is not uniformly licensed, so neither is anything built on
it.** A repository with two licences is two projects sharing a tree. A fork
that keeps `ee/` cannot describe itself as Apache 2.0, cannot be packaged by
distributions that require an OSI licence, and cannot be vendored by
downstream projects whose own licence audit is automated. The cheapest
mitigation is the one that also costs the most: delete `ee/` and lose
whatever it contains.

**2. Source-available typically restricts the use that a fork is *for*.**
Enterprise directories in this pattern usually carry a non-compete or
internal-use-only clause — you may read and modify, you may not offer it as a
competing service. An open-source fork's whole value proposition is that
anyone may run it for anyone, which is the exact use such a clause names. So
the question to answer before forking is not "may I copy this" but "may my
users run what I publish", and the two have different answers.

**3. The boundary moves.** `ee/` is a directory, and directories are a
runtime decision by the upstream project. A feature that is Apache-licensed
today can be reorganised into `ee/` in the next release, so a fork tracking
upstream inherits a licence boundary that is not under its control. That
argues for pinning to a commit and auditing per upgrade, rather than
tracking a branch — a maintenance cost that lands on the fork forever.

**What a fork should actually do, concretely.**

- Exclude `ee/` at the fork point and record the commit.
- Check what stops working. This is the measurable part, and it is the same
  question the code exercises above ask about the runtime: which
  capabilities are in the core and which are in the part you cannot ship.
  Observability backends and studio tooling are the usual answer, and
  exercise 3's finding about the session store applies — the part that scales
  with traffic is the part you most want a supported implementation of.
- State the licence of the result in the README, per directory, and keep the
  upstream notice files. Apache 2.0 requires attribution and a NOTICE; the
  enterprise licence has its own terms that an excluded directory does not
  discharge if any of its code survived in a diff.
- Re-audit on every upstream merge, because of point 3.

**The honest framing:** none of this is unusual or hostile — it is the
standard commercial-open-source shape, and the lesson's pitfall is named
"enterprise license confusion" rather than "enterprise license trap". The
failure mode is a team that forks, ships, and discovers the boundary during
a customer's licence review. The fix is fifteen minutes of reading at the
fork point and a note in the README.
