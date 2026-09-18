<!-- generated:start -->
# 11-llm-engineering / 16-langgraph-state-machines

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/11-llm-engineering/16-langgraph-state-machines/) · upstream spec
`phases/11-llm-engineering/16-langgraph-state-machines/docs/en.md`

```bash
uv run demo practice run 16-langgraph-state-machines --ex 1
uv run demo explain 16-langgraph-state-machines --ex 1
uv run pytest demos/phases/11-llm-engineering/16-langgraph-state-machines
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Implement the four-node ReAct graph above with a calculator tool and a web-search tool.… | code | T1 | `ex01_ten_checkpoints_against_a_threshold_of_four.py` |
| 2 | Medium. Add a `planner` node that runs before `agent` and writes a structured `plan: list[str… | code | T1 | `ex02_the_resume_keeps_the_plan_and_the_node_throws_it_away.py` |
| 3 | Hard. Build a supervisor graph that routes between three subgraphs (`researcher`, `writer`, `… | code | T1 | `ex03_a_checkpoint_is_a_point_in_time_not_a_branch.py` |
<!-- generated:end -->

## Answers

The lesson imports `langgraph`, `langchain-core` and `langchain-anthropic` at
module scope, so all three exercises are **T1** in the `agents` group and CI
skips them at `DEMO_TIER=T0`. None of them needs an API key: every node returns
a deterministic string in place of a model call, because every claim the
exercises make is about the graph — checkpoints, reducers, interrupts, `Send`
and time travel — and not about what a model says.

### 1 — ten checkpoints against a threshold of four

```text
stream call        checkpoints
turn 1, to pause        3
turn 1, resumed         5
turn 2, to pause        8
turn 2, resumed        10
```

**ANSWER: a two-turn conversation leaves 10 checkpoints.** "At least four" is
satisfied before the first turn has finished, so a test written to that
threshold passes whether the graph checkpoints once per node or once per turn.

**FINDING: "four-node" counts the two nodes LangGraph adds.** `add_node` is
called twice; the compiled graph reports `['__end__', '__start__', 'agent',
'tools']` with four edges, two of them conditional.

**FINDING: two turns take four `stream` calls.** `interrupt_before=["tools"]`
pauses before every tool call, so each turn is a stream ending in an
`__interrupt__` event with `next == ('tools',)` plus a second stream carrying
`Command(resume=True)`. Exactly **2** of the 10 snapshots are paused at `tools`
— a property of the interrupt, not of the traffic, and the number a test should
actually assert.

**FINDING: `web_lookup` cannot answer the question the lesson asks it.** The
facts dict is keyed `"anthropic headquarters"` and matched by exact equality
after `strip().lower()`, so the demo's own *"Where is Anthropic
headquartered?"* returns `"unknown"`. The tool works only if the model emits
the dictionary key verbatim, and the prompt never says what the keys are.

**FINDING: the calculator's allow-list admits exponentiation.**
`set(expression) <= set("0123456789+-*/(). ")` passes `"9**9**9"`, because `*`
is in the list and `**` is two of them. `__import__` *is* blocked and `1/0`
returns `ERROR: ZeroDivisionError(...)`, so the guards that exist work — the
gap is the one a character class cannot see.

### 2 — the resume keeps the plan and the node throws it away

The planner writes three steps; `agent` returns only `["research the topic
(done)"]`.

| reducer for `plan` | at the interrupt | after the resume |
|---|---:|---|
| none (last write wins) | 3 steps | **1 step** |
| `operator.add` | 3 steps | **4 steps**, one duplicated |
| merge by index | 3 steps | **3 steps**, one marked |

**FINDING: nothing is lost across the resume.** All three compilations hold the
identical three steps at the interrupt, and re-reading the interrupt's *own*
checkpoint config after the resume returns them byte-for-byte. The checkpointer
is not the mechanism — `Command(resume=True)` replays from a snapshot that
still has the plan, and then `agent` writes over it. A test that fails "if
`plan` is lost across a checkpoint resume" is watching the wrong edge.

**FINDING: the two obvious reducers fail in opposite directions** — one drops
the steps the update did not mention, the other keeps both copies of the step
it did.

**FINDING: `add_messages` on a `list[str]` raises.** The obvious copy-paste —
reuse the reducer already in `State` — fails with
`NotImplementedError: Unsupported message type: <class '…ChatPromptTemplate'>`,
because `add_messages` coerces every element through the message converter and
a plain string is read as a prompt template. The error names a class the state
never contained.

### 3 — a checkpoint is a point in time, not a branch

```text
supervisor -> researcher -> supervisor -> [interrupt] -> writer -> supervisor -> reviewer -> supervisor
8 node executions, 9 checkpoints
```

**ANSWER: the interrupt fires through `Send`.** The pause arrives with
`next == ('writer',)` and the brief already in state, so a human can approve
it. `interrupt_before` is checked when a node is *scheduled*, and a `Send`
schedules one.

**FINDING: `Send(node, state)` double-counts every accumulating channel.**

| payload | log entries | node executions |
|---|---:|---:|
| only the fields the subgraph reads | 7 | 8 |
| the whole state | **22** | 8 |

The payload carries `log`, the subgraph's `operator.add` re-adds it on the way
in, and the subgraph returns the accumulated list, which `operator.add` merges
again on the way out. `Send` looks like a message and behaves like a state
update.

**FINDING: each subgraph's own checkpointer is inert.** `researcher` is
compiled with a `MemorySaver` and holds **0** entries after the run; all nine
checkpoints belong to the parent. "Each subgraph has its own state and
checkpointer" is half true — the state is its own, the checkpointer is
decoration unless `checkpointer=True` is passed at compile time.

**FINDING: time travel re-runs the whole suffix.** Forking at the `writer`
checkpoint re-runs `writer → supervisor → reviewer → supervisor`: four of the
original eight, every node downstream. Forking at `researcher` re-runs
`researcher → supervisor` and stops at the same interrupt again. There is no
branch to isolate.

**FINDING: the fan-out `Send` exists for cannot be used here.** One payload has
to serve three nodes:

```text
Send(name, {"topic": ...})                 -> KeyError: 'brief'
Send(name, {topic, brief, draft})          -> InvalidUpdateError: At key 'topic':
                                              Can receive only one value per step
```

Each subgraph's schema contains every key and returns every key, and the parent
declares a reducer only for `log`. Sequencing through the supervisor is what
makes the graph work — which is why the supervisor runs **4 of the 8**
executions, and which turns `Send` into a goto.
