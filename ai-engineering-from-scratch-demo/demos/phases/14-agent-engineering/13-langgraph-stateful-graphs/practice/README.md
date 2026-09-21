<!-- generated:start -->
# 14-agent-engineering / 13-langgraph-stateful-graphs

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/13-langgraph-stateful-graphs/) · upstream spec
`phases/14-agent-engineering/13-langgraph-stateful-graphs/docs/en.md`

```bash
uv run demo practice run 13-langgraph-stateful-graphs --ex 1
uv run demo explain 13-langgraph-stateful-graphs --ex 1
uv run pytest demos/phases/14-agent-engineering/13-langgraph-stateful-graphs
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a conditional edge from `classify` to `end` when classification confidence is below a thr… | code | T0 | `ex01_the_checkpoint_keeps_the_pause_flag_so_resume_pauses_again.py` |
| 2 | Swap the SQLite-like fake for a real SQLite checkpointer. Measure per-step serialization over… | code | T0 | `ex02_the_checkpointer_stores_the_whole_state_every_step.py` |
| 3 | Implement parallel edges: two nodes run concurrently, merge by a custom reducer. What does im… | code | T0 | `ex03_immutability_makes_the_merge_possible_not_correct.py` |
| 4 | Read `langgraph-supervisor` reference. Port the toy to `create_supervisor`. Compare the trace… | explain | T0 | prose, below |
| 5 | Add streaming: each node yields partial state while it runs. Print the deltas as they arrive. | code | T0 | `ex05_a_node_returns_one_dict_so_a_partial_has_nowhere_to_go.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib — including the SQLite checkpointer, since
`sqlite3` ships with Python. Four ship code; exercise 4 reads the
`langgraph-supervisor` reference, so it is answered in prose.

The recurring finding is that **`Runner.run` saves the checkpoint before it
pops `_pause_reason`**, so the durable copy of the state is the one that
cannot be resumed from. Resuming from `load_latest` pauses at the same node
forever; the only resumable copy is the one attached to the exception, which
is in memory and dies with the process. Exercise 2 shows the bug survives a
port to real SQLite — the flag is written to disk exactly as it is written to
the dict, so "durable execution" is durable and not executable.

The second thread is that **three of the four extensions the exercises ask
for are blocked by a type rather than by an algorithm**. `_next` returns
`str | None`, so a parallel edge cannot be written down. `NodeFn` is
`Callable[[State], Update]`, so a streaming partial has nowhere to go.
`_classify` returns a route with no confidence, so a threshold has nothing to
compare. In each case the change is one signature wide and the code after it
is short.

What holds: immutability. Every shipped node returns an update dict rather
than editing the state, and exercise 3 shows that is exactly what makes a
parallel merge definable at all.

### 1 — the checkpoint keeps the pause flag, so resume pauses again

**ANSWER: a confidence-gated edge to END, and a manual resume.** At a
threshold of **0.5** a gibberish input stops at `classify` with confidence
**0.0**, route `unknown` and **no** ticket; setting `route` by hand and
resuming carries it to `sent BUG-zzz qqq` at step **4**.

**FINDING: the shipped classifier has no way to be unsure.** Its `else`
branch returns `sales`, so `zzz qqq` is routed to sales and a `SAL` ticket is
raised. **3** of its **4** branches are keyword matches and the fourth is a
default that looks like one — which is why the threshold needed a new
classifier before it needed a new edge.

**FINDING: the saved checkpoint cannot be resumed from.** `run` calls
`checkpointer.save(...)` and *then* `state.pop("_pause_reason")`, so the
stored state still carries the flag. Resuming from `load_latest` pauses at
`human_gate` again, and again, and the only copy that resumes is the one on
the exception. Two lines swapped fixes it; nothing in the demo's output would
have shown it, because the demo resumes from the exception.

**FINDING: resume re-runs the node it paused at.** Coming back into
`human_gate` executes it a second time, so `step` reaches **5** over a
**4**-node path. Either every node is idempotent or the runner resumes at the
*next* node — and `resume_from` is the caller's guess either way, which is
the same class of problem as the lesson's "non-deterministic nodes" pitfall.

### 2 — the checkpointer stores the whole state every step

**ANSWER: a real SQLite checkpointer, 3 rows and 374 bytes to the pause.**
Each step writes the whole state as JSON — **90**, **120**, **164** bytes —
and round-tripping every row reproduces the in-memory history exactly: **3**
of **3** states match. Overhead is reported in bytes rather than seconds
deliberately: a wall-clock figure would describe this machine, and the byte
count describes the design.

**FINDING: the overhead is the snapshot, not the backend.** The in-memory
store keeps the same **3** deep copies, holding **15** key-value pairs in
total, where a delta-encoded checkpointer would write **8** changed keys.
Switching backends moves the bytes; it does not remove them. If per-step cost
matters, the lever is the encoding, not the store.

**FINDING: an unserializable field is only a problem for the real one.** A
callable in the state is *accepted* by the in-memory checkpointer, because
`copy.deepcopy` handles it, and raises `TypeError` in `json.dumps`. The fake
accepts states the durable one cannot, so the failure appears the first time
durability is switched on — in production, by definition.

**FINDING: the pause flag is written to disk too.** The `human_gate` row
contains `_pause_reason`, so exercise 1's resume bug survives the port
unchanged. Porting the backend ports the bug, which is worth knowing before
treating "we moved to a real checkpointer" as a durability milestone.

### 3 — immutability makes the merge possible, not correct

**ANSWER: two branches, a reducer per contested key, and an order-independent
result.** Run in either order the reduced state is identical — step **3**,
owner `oncall+triage`. Each node reads the same base and returns an update,
so the merge is a function of the two updates rather than of when they ran.

**What immutability buys, precisely:** it makes the merge *definable*. Under
the runner's own rule — `{**state, **update}`, last write wins — the two
orders disagree on `owner`, a key both branches write with different values.
Immutability does not prevent that disagreement; it makes it a visible
property of two update dicts instead of a hidden property of execution order.
The reducer is what resolves it, and the guarantee holds only while the
reducer table is *total* over the contested keys.

**FINDING: `step` is the key that breaks.** Both branches compute
`state.get("step", 0) + 1` from the same base and return **2**, so
last-write-wins records **2** for two nodes of work while an add-the-deltas
reducer records **3**. Every node in this lesson writes `step`, so every
fan-out in this graph needs that reducer — which is exactly what LangGraph's
annotated state fields are for.

**FINDING: a mutating node loses the property entirely.** A node that edits
the state in place leaves the base changed, and whichever branch runs after
it reads the edit: the two orders disagree on `severity` although neither
branch ever wrote a conflicting value for it. The shipped nodes all return
updates and **nothing enforces it** — no frozen dict, no copy at the call
site, no check.

**FINDING: the graph cannot express a fan-out at all.** `_next` is typed
`str | None` and returns the first matching edge, so the parallel step had to
be built beside the graph rather than inside it.

### 4 — a supervisor is a router whose targets are agents

*Cites "Three topologies".*

**The port is mostly a relabelling, and the two things that change are worth
naming separately: what the graph looks like, and what the trace looks
like.**

**What the toy already is.** `build_graph()` is a supervisor topology drawn
with the supervisor's job factored out into `_classify` plus
`add_conditional_edges`: one entry node decides, three specialists do the
work, and everything reconverges on `human_gate` and `send`. That is
`create_supervisor` with the router written as a function instead of as an
LLM. The port replaces `_classify` with a model call and the three handler
*nodes* with three agents — and the edge structure does not move.

**What changes in the graph.** Three things, in order of how much they
matter:

1. **The router's output stops being a closed set.** The toy's conditional
   edges are built from `targets={"refund": ..., "bug": ..., "sales": ...}`,
   so the router can only return one of three strings and `_next` returns
   `None` — silent termination — for anything else. A supervisor that picks
   from a tool list has the same problem in a form that is harder to see,
   which is exercise 1's finding restated: there is no confidence and there
   is no "none of these".
2. **Control returns to the supervisor after every agent.** The toy's
   handlers edge straight to `human_gate`; a supervisor loops back so it can
   dispatch again. That is the difference between a router and a supervisor,
   and it is the difference between a bounded graph and one that needs a step
   cap — the lesson's own "over-use of conditional edges" pitfall arrives
   here.
3. **Each agent has private state.** The toy has one flat `State` dict that
   every node reads and writes, which is why exercise 3's `step` collision
   exists at all. Subagents with their own scratch state remove that class of
   collision and add a new one: deciding what the supervisor sees, which the
   LangChain team's 2026 note about "doing this through tool calls directly
   for more context control" is about.

**What changes in the trace, which is what the exercise asks to compare.**
The toy's trace is `checkpointer.history(session)` — one row per node, each
the *whole state*, measured in exercise 2 at **90**, **120**, **164** bytes.
A supervisor trace is nested: one span per supervisor turn, containing one
span per agent turn, each with its own messages. Three concrete
consequences:

- **Depth replaces sequence.** The toy's history is a flat list whose length
  is the number of nodes run; a supervisor's is a tree whose depth is the
  number of hand-offs. "Where did this run spend its time" stops being
  answerable by reading rows in order.
- **The state snapshot stops being the whole picture.** Exercise 2's finding
  — that the overhead is the snapshot — cuts the other way here: a flat
  state that serialises in 164 bytes is *complete*, and a supervisor's
  per-agent message lists are not captured by any single row unless the
  checkpointer is taught about them. This is the lesson's "checkpoints too
  small" pitfall, and it is the specific thing a port has to get right.
- **Streaming becomes the primary interface.** Exercise 5 measures the toy
  emitting **7** deltas where the runner takes **4** checkpoints. With
  agents inside nodes, the gap widens by whatever each agent does internally,
  and the only way to watch a long supervisor run is the stream — which is
  why LangGraph makes streaming a runtime property rather than a node one.

### 5 — a node returns one dict, so a partial has nowhere to go

**ANSWER: generator nodes and a streaming runner.** The same four-node path
emits **7** deltas instead of **4** updates, and the final state matches the
shipped run on **6** of **6** keys. The deltas arrive in order, `classify`
first with its route and `send` last with its step count.

**FINDING: the checkpoint cadence did not change.** The runner still saves
once per node, so **4** checkpoints cover **7** deltas and a crash after the
second delta of a node loses all of them. Streaming makes a run *observable*
without making it *resumable* — two features that look like one from the
outside, and the lesson's durable-execution claim is about the second.

**FINDING: a pause can now be seen mid-node.** The shipped runner checks
`_pause_reason` after `fn(state)` returns, so a node cannot stop halfway. The
streaming version stops with **1** of `human_gate`'s **2** deltas applied —
a state no checkpoint row describes, which is a new resume problem created by
the new feature.

**FINDING: the node type is the whole constraint.** `Runner.run` calls
`fn(state)` **once**, and all **6** of the lesson's nodes are written as a
single `return`. Streaming is not an addition to this design; it is a
different signature. That is the honest answer to why frameworks expose
streaming at the runtime: the alternative is rewriting every node.
