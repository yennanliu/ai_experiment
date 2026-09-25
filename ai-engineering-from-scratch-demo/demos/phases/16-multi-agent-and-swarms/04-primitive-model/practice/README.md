<!-- generated:start -->
# 16-multi-agent-and-swarms / 04-primitive-model

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/04-primitive-model/) · upstream spec
`phases/16-multi-agent-and-swarms/04-primitive-model/docs/en.md`

```bash
uv run demo practice run 04-primitive-model --ex 1
uv run demo explain 04-primitive-model --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/04-primitive-model
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py` three times with different agent policies. Observe how the orchestrator ch… | code | T0 | `ex01_all_three_orchestrators_produce_the_same_transcript.py` |
| 2 | Implement a fourth orchestrator type: a queue-driven one where agents poll shared state for w… | code | T0 | `ex02_nothing_in_the_agent_primitive_can_say_not_yet.py` |
| 3 | Take the LangGraph quickstart (https://docs.langchain.com/oss/python/langgraph/workflows-agen… | explain | T0 | prose, below |
| 4 | Read the OpenAI Swarm cookbook (https://developers.openai.com/cookbook/examples/orchestrating… | explain | T0 | prose, below |
| 5 | Find one framework in this table that hides shared state entirely. Explain what breaks when a… | code | T0 | `ex05_the_only_order_the_demo_runs_is_the_one_where_it_does_not_matter.py` |
<!-- generated:end -->

## Answers

### 1 — all three orchestrators produce the same transcript

Run it. The three pools are identical — same three messages, same order, same
content, **0 of 3 pairs differ**:

```
[0] researcher | note 1: FIPA-ACL ratified 2000; 20 performatives. -> writer
[1] writer     | Draft summarizing: note 1: ...                    -> reviewer
[2] reviewer   | Review verdict: approved.                         -> done
```

So the demo's closing line — *"only the orchestrator choice changes who speaks
when"* — is refuted by the output directly above it.

The exercise says to vary the agent policies, so vary the one field a policy
controls, the researcher's `handoff` target:

| researcher hands off to | static | handoff-driven | round-robin |
|---|---|---|---|
| `writer` (shipped) | r, w, rev | r, w, rev | r, w, rev |
| `reviewer` | r, w, rev | **r, rev** | r, w, rev |
| `done` | r, w, rev | **r** | r, w, rev |
| `researcher` | r, w, rev | **10 × r** | r, w, rev |

The static and round-robin columns never move — across every value, in 4 of 4.
Only the handoff orchestrator responds to what an agent decides. The claim is
true of one orchestrator out of three.

Why is easy to check: after a static run, **3 of 3** messages in the pool carry
a `handoff` key, and `StaticOrchestrator.run` never mentions the word. It walks
`self.order`. The round-robin selector likewise reads only position. Both store
the agents' decisions and route by something else.

Two details from the bottom row. A self-handoff emits exactly **10** messages —
`max_steps` — and returns normally; nothing in the pool, the return value or
the state records that the loop was cut off, so a cycle and a completion look
the same to the caller. And `max_steps` means two different things:
`StaticOrchestrator` slices `self.order[:max_steps]`, capping the *plan*, while
the other two use `range(max_steps)`, capping the *run*. The slicing one
silently drops the tail of a longer order rather than stopping at the limit.

### 2 — nothing in the Agent primitive can say "not yet"

The fourth orchestrator sweeps a poll order, runs whoever is ready, once each,
and repeats until a sweep appends nothing.

The obvious deadlock detector — "a sweep that appends nothing" — is wrong,
because that is also how a successful run ends:

| team | stalls on sweep | agents still waiting | means |
|---|---:|---:|---|
| writer needs researcher, reviewer needs writer | 2 | **0** | completion |
| circular preconditions | 1 | **3** | deadlock |

The signal is the zero-append sweep **plus a non-empty waiting set**. One sweep
is enough either way, well before any timeout, because the pool is the only
input any policy reads — if a full pass over every agent changed nothing, a
second pass cannot change anything either.

Writing it exposes where readiness has to live. All **3** shipped policies
contain exactly **one** `return` and **zero** early exits: each produces a
message unconditionally. `Message` is a bare `dict` with no "no work for me"
shape. So the preconditions are a table the orchestrator keeps on the side — a
fifth thing, in a lesson whose argument is that there are four.

And because no agent can decline, the queue orchestrator produces the same
three messages in the same order as the other three. A fourth identical pool.

The interesting case is what happens *without* the precondition table, which is
the honest version of "agents poll shared state". Sweeping the writer first
reaches `writer_policy`'s `"Draft with no research yet."` branch — unreachable
under every shipped orchestrator. The reviewer then approves by searching the
writer's own output for the substring `"summarizing"`, does not find it, and
returns **needs revision**. Polling order decides whether the work passes
review, and the gate is a word the writer happens to use.

### 3 — three of the four map, and Handoff is the one that does not

*Draws on "How every 2026 framework maps to it".*

Rewriting the LangGraph quickstart as the four primitives, three map cleanly
and one has no counterpart:

| primitive | LangGraph | 1:1? |
|---|---|---|
| Agent | node function | yes — a callable over state returning an update |
| Shared state | `StateGraph` + reducer | yes, with a caveat below |
| Orchestrator | the compiled graph | yes — `StaticOrchestrator` is this, exactly |
| **Handoff** | **conditional edge** | **no** |

The caveat on state is the interesting one. This module's `SharedState` is
append-only full history; a `StateGraph` reducer *projects*. `add_messages` is
the reducer that makes it behave like this module's pool, and choosing a
different one changes what every downstream node can see — which is the
mechanism exercise 5 breaks on purpose.

Handoff is the real mismatch. In the primitive model the **agent** names its
successor; the message carries `handoff`. In LangGraph a node cannot name its
successor — it writes to state, and a conditional edge, which is part of the
graph rather than the node, reads that state and routes. The decision moves
from the agent to the orchestrator. That is not a wrapper over Handoff, it is
its absence, and the measurement in exercise 1 is the same fact from the other
side: `StaticOrchestrator` stores every `handoff` field and reads none.

Convenience wrappers, not primitives: `create_react_agent` (an Agent with a
tool loop already written), the checkpointer (durability over Shared state),
`interrupt` (a pause in the Orchestrator), and streaming (an observation of it).
Each is implementable on top of the four, which is the lesson's actual claim
and holds up.

### 4 — Swarm makes Handoff ergonomic and pushes Shared state to you

*Draws on "The stateless insight".*

**Most ergonomic: Handoff.** In the cookbook a handoff is a tool that returns
an `Agent`. There is no graph to declare, no registry, no edge — the agent's
successor is the return value of a function it already knows how to call. It
is the shortest expression of the primitive in any framework in the table.

**Pushed to the caller: Shared state.** The lesson's own table says so in the
cell: `caller's problem`. `context_variables` is a dict the caller threads
through the loop; nothing accumulates history unless you accumulate it.

The Orchestrator is the second casualty, and it goes somewhere unusual rather
than to the caller: *"the LLM's next handoff call"*. You do not write one. That
is the bet, and this module measures both sides of it. The handoff-driven
orchestrator is the **only one of three** whose transcript responds to what an
agent decides (exercise 1) — so if routing is what you want the agents to own,
Swarm's shape is the one that delivers it. And **all three** shipped policies
read more than the message that reached them (exercise 5) — so everything this
lesson's agents do is exactly what Swarm makes you carry yourself.

### 5 — the only order the demo runs is the one where it does not matter

**OpenAI Swarm / Agents SDK.** Of the six framework rows it is the only one
whose shared-state cell names no mechanism at all: `caller's problem`.

What breaks, on this module, with `SharedState` projected down to the last
message:

| order | full pool | projected pool |
|---|---|---|
| researcher, writer, reviewer | `approved` | `approved` |
| researcher, researcher, writer | draft joins **2** notes | joins **1** |
| researcher, writer, researcher, reviewer | `approved` | **`needs revision`** |

Nothing raises in any row. The third is the one that matters: one extra
interleaved message and `last_by("writer")` — a backward scan of history, not a
read of the previous message — finds nothing, so the reviewer's verdict flips.
A correctness change with no error, no warning, and no difference in the code.

The second row is the quieter version. `writer_policy` joins every researcher
message out of `snapshot()`; projected, it joins one. `" | ".join(...)` is
happiest exactly when there is nothing to join, so a shorter draft is the only
symptom.

And the first row is why the lesson never sees any of this. All three shipped
orchestrators produce the strictly linear researcher, writer, reviewer — one
message per agent — and under that order every read happens to land on the
previous message, so the projected pool and the full pool are byte-identical.
The demo can only run the case where hiding state is free.
