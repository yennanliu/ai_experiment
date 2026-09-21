<!-- generated:start -->
# 14-agent-engineering / 17-claude-agent-sdk

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/17-claude-agent-sdk/) · upstream spec
`phases/14-agent-engineering/17-claude-agent-sdk/docs/en.md`

```bash
uv run demo practice run 17-claude-agent-sdk --ex 1
uv run demo explain 17-claude-agent-sdk --ex 1
uv run pytest demos/phases/14-agent-engineering/17-claude-agent-sdk
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a subagent spawner that batches 20 tasks into groups of 5 parallel subagents. Measure orc… | code | T0 | `ex01_the_spawner_folds_nothing_back_so_the_parent_learns_nothing.py` |
| 2 | Implement a `PreToolUse` hook that rate-limits `write_file` calls (5 per minute per session).… | code | T0 | `ex02_the_hook_cannot_deny_and_cannot_see_the_session.py` |
| 3 | Wire `list_subkeys` to render a subagent tree. What does deep nesting look like? | code | T0 | `ex03_delete_prunes_one_level_so_grandchildren_outlive_the_root.py` |
| 4 | Port the toy to the real `claude-agent-sdk` Python package. What changes about tool registrat… | explain | T0 | prose, below |
| 5 | Read the Claude Managed Agents docs. When would you switch from self-hosted to managed? | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

All five are T0 and stdlib. Three ship code; exercises 4 and 5 read the SDK
and the Managed Agents docs, so they are answered in prose.

The recurring finding is that **the harness gives the caller a handle and
keeps no channel back**. `spawn_subagents` returns `list[AgentRun]` and
appends nothing to the parent's session, so the orchestrator's context is
bounded because it is *empty*, and the `AgentRun` objects are the only copy
of the results. A `PreToolUse` hook's return value is discarded by
`_dispatch`, so a hook cannot deny a call — it can only raise, and the hook
loop sits above the `try`, so a refusal kills the whole run. `delete`
cascades one level, so grandchildren outlive the root.

The second thread is that **the identifiers carry meaning the API does
not**. A subagent's session id is built by string concatenation, so
`root.sub01.sub04` tells you the depth and the global creation order and
nothing about position under its parent. And the hook signature is
`(tool_name, args)` — the session is not in it, so "5 per minute per
session" is not expressible as written.

What holds: the shape. Built-in tools in a registry, subagents with isolated
sessions, lifecycle hooks at four points, a session store with a subkey
index. Every finding above is a gap *inside* that shape rather than a
disagreement with it.

### 1 — the spawner folds nothing back, so the parent learns nothing

**ANSWER: batching 20 tasks into 5 subagents costs the orchestrator 23
tokens against 83 one-per-task.** Folding one summary line per subagent,
**5** groups of **4** cost **23** where **20** groups of **1** cost **83** —
a **3.6x** reduction — while running all 20 calls inline costs **107**. That
is the lesson's "context isolation" claim, measured.

**FINDING: the shipped spawner folds nothing back.** Through
`spawn_subagents` the parent session holds **0** turns for both 5 subagents
and 20. The comparison the exercise asks for cannot be made without first
writing the fold, because the shipped orchestrator's context does not grow
with subagent count at all — which looks like the ideal result and is
actually the absence of a result.

**FINDING: the redistribution is not free.** Inline costs **107** tokens in
one session; **5** subagents cost **145** in total and **20** cost **280**,
because each one re-reads its own prompt and writes its own output.
Batching hides **122** tokens from the orchestrator and adds work everywhere
else. This is the arithmetic behind the lesson's "subagent over-spawn"
pitfall: per-subagent overhead is fixed, so it dominates as tasks shrink.

**FINDING: `context_tokens` is a word count.** `len(text.split())` at **4**
sites, so a **200**-character single line costs **1**. Every number above is
in that unit, which is the only one the harness offers — worth stating
because the decision the exercise is about is a token-budget decision.

### 2 — the hook cannot deny, and cannot see the session

**ANSWER: a raising rate limiter, and the sixth call kills the run.** Five
`write_file` calls pass; the sixth raises, propagates through `_dispatch` and
out of `run_agent`, leaving **5** tool turns in a **6**-turn session.
Advancing the virtual clock past the window allows another **5**.

**FINDING: returning `False` does not block anything.** `_dispatch` calls
`hook(tool_name, args)` and assigns the result **0** times, so a hook
returning `False` on all **8** calls lets all **8** through. Denial has
exactly one implementation available and it is an exception — which is a
design decision the type signature (`-> None`) states and the docs' framing
("gate or audit tool calls") does not.

**FINDING: the raise takes the whole run with it.** The hook loop is above
the `try` that wraps the tool, so a rate-limit refusal is an unhandled
exception rather than a failed tool call: `session_end` fires **0** times for
the killed session against **2** for one that completed. A rate limit is
indistinguishable from a crash, and the session is never closed — so the
lesson's own "session bloat" pitfall is made worse by its own hook system.

**FINDING: "per session" cannot be written down.** The hook receives **2**
parameters and neither is a session id, so the counter has to key on
something in the *arguments*. Two sessions writing under one path prefix
share a budget: **5** allowed between them rather than 5 each. The exercise's
specification is unimplementable against the shipped signature, and noticing
that is the exercise.

### 3 — `delete` prunes one level, so grandchildren outlive the root

**ANSWER: a recursive walk over `list_subkeys`.** A root with **3** children
and **3** grandchildren each renders **13** lines at depths **0**–**2**, with
ids like `root.sub01.sub04`. Deep nesting looks like a filename, because the
id is string concatenation in `spawn_subagents`.

**FINDING: `delete` prunes one level.** The docs promise "delete(session_id)
— with cascade to subagent sessions", and the implementation pops the
session's own subkeys only: deleting the root leaves **9** of **13**
sessions behind — exactly the grandchildren — reachable by nobody, because
the `_subkeys` entry naming them went with their parent. One-level cascade on
a two-level tree is a leak whose size grows with depth.

**FINDING: the last component is a global sequence, not a position.**
`_sub_counter` increments on the `Harness`, so the root's children are
`sub01`–`sub03` and the first child's children are `sub04`–`sub06`. The
suffix records when a subagent was created across the whole harness and never
where it sits under its parent — so two runs of the same workload produce
different ids for the same logical node.

**FINDING: a subagent created outside the spawner is an orphan.**
`run_agent` links a parent only when `parent_session` is passed, so a direct
call leaves **2** sessions in the store and **1** node in the tree. The tree
is a view over `_subkeys`, not over the sessions that exist — which means
`list_subkeys` cannot be used to audit what is in the store.

### 4 — registration moves from a dict to a decorator and a permission

*Cites "Built-in tools".*

**Three things change, and only the first is syntax.**

**1. Registration stops being a dict insert.** The toy's `ToolRegistry` is
`dict[str, Tool]` with `register(tool)`, and `Tool` carries `name`,
`description`, `fn` — **3** fields. The real SDK registers a custom tool
through the standard tool-schema interface: a decorated function whose
*parameters* become a JSON schema and whose docstring becomes the
description. The consequence is not ergonomics, it is that arguments are
validated before `fn` is called, where the toy's `_dispatch` does
`tool.fn(**args)` inside a `try` and turns a `TypeError` into a string.

**2. Built-in tools are not registered at all.** The lesson says the SDK
ships 10+ — file read/write, shell, grep, glob, web fetch — and the toy
registers stand-ins for two of them by hand. In the real SDK they are
present unless you restrict them, so the interesting configuration is
*subtractive*: which built-ins this agent may use, rather than which tools
you remembered to add. That inverts the failure mode. Here a missing tool is
`error: unknown tool 'write_file'`; there, an unexpected tool is a shell
command you did not intend to allow.

**3. Which is why registration and permission become the same decision.**
Exercise 2 is the toy's whole answer to "gate a tool": a `PreToolUse` hook
that can only raise, receives **2** parameters, and cannot see the session.
The SDK's model puts allow/deny lists and permission modes beside the tool
list, and keeps hooks for cross-cutting behaviour — audit, rate limits,
notifications. Porting the rate limiter therefore means deciding whether it
is a *permission* (declarative, per agent) or a *hook* (imperative, per
call), and exercise 2 shows the toy forces the second and makes it a crash.

**What also moves, briefly:** MCP servers become a tool source (Phase 13),
so "registration" includes connecting a server and inheriting its tools —
something with no analogue in a `dict[str, Tool]`. And tool results become
structured content blocks rather than `str`, which is what exercise 1's
word-count `context_tokens` is standing in for.

### 5 — managed is the answer when the session store is the product

*Cites "Claude Managed Agents".*

**Switch when the things you are maintaining are the three this lesson
measures: session lifecycle, context budget, and long-running work.**

The lesson frames it as "trade control for managed infrastructure", with
long-running async work, built-in prompt caching and built-in compaction as
what you get. The useful version of the decision is to ask which of the
self-hosted parts you are currently getting wrong — and the exercises name
three.

**Switch if session lifecycle is load-bearing.** Exercise 3: `delete`
cascades one level and leaves **9** of **13** sessions orphaned; sessions
created outside the spawner never appear in the tree. The lesson's own
"session bloat" pitfall says to use `list_sessions` plus an expiry policy,
which is a piece of infrastructure with a retention question, a storage bill
and a correctness bug waiting in it. If your product is not a session store,
running one is pure cost.

**Switch if compaction is the hard part.** Exercise 1 is the whole argument
for subagents — **122** tokens hidden from the orchestrator — and it is a
manual technique: you decide the batch size, you write the fold, you measure
the result. Managed agents bring built-in compaction, which is the same
problem solved once by people who can measure it against real traffic.
Against that, self-hosting gives you `PreCompact` and the ability to decide
*what* survives compaction, which matters when the thing that must survive is
domain-specific.

**Switch if runs outlive the process.** The toy's `Harness` holds sessions in
a dict and its rate limiter (exercise 2) holds timestamps in memory; both
die with the process. Anything that needs to resume after a deploy needs
durable session storage and a scheduler, and that is the part of
"self-hosted" that is genuinely a distributed system rather than a library.

**Stay self-hosted if any of these is true.** Tools must run inside your
network or against your filesystem — the built-ins in exercise 4 are the
point of the SDK and they are local by nature. Hooks are how you enforce
policy, and you need them to *deny* rather than crash, which means owning
the dispatch loop. Or the trace has to join your existing spans: the lesson
notes W3C trace context propagates into the CLI subprocess so the whole
multi-process run is one trace, and that is a self-hosted property.

**The honest summary:** managed wins on the parts where correct is expensive
and undifferentiated — retention, compaction, resumption. Self-hosted wins
on the parts where the *policy* is the product. Exercises 1–3 are three
concrete measurements of the first category, which is why they read as bugs
rather than as design choices.
