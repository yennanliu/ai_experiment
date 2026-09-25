<!-- generated:start -->
# 16-multi-agent-and-swarms / 11-handoffs-and-routines

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/11-handoffs-and-routines/) · upstream spec
`phases/16-multi-agent-and-swarms/11-handoffs-and-routines/docs/en.md`

```bash
uv run demo practice run 11-handoffs-and-routines --ex 1
uv run demo explain 11-handoffs-and-routines --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/11-handoffs-and-routines
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`, triage to the refund agent. Confirm the second turn's active agent is ref… | code | T0 | `ex01_the_second_turn_is_refund_and_it_refunds_order_42.py` |
| 2 | Add a loop-detection rule: if the same two agents have handed off 3 times in a row, force an… | code | T0 | `ex02_the_fallback_cannot_be_triage_because_triage_is_half_the_loop.py` |
| 3 | Read the OpenAI Agents SDK docs on handoff filters. Implement a "summarize-on-handoff" versio… | code | T0 | `ex03_the_summary_must_keep_what_the_next_agents_tools_take.py` |
| 4 | Compare the Swarm handoff to a GroupChatManager selector. Which pattern makes prompt injectio… | code | T0 | `ex04_one_injected_turn_owns_the_rest_of_the_session.py` |
| 5 | Read the Swarm cookbook (https://developers.openai.com/cookbook/examples/orchestrating_agents… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — the second turn is refund, and it refunds order 42

**Yes, the second turn's active agent is refund.** You have to add a turn to
see it, though: all 4 of the demo's scenarios are one message long. Re-running
the refund flow with a second message:

| turn | user | answered by | reply |
|---|---|---|---|
| 1 | "I need a refund on order 77" | refund (after `(handoff to refund)`) | Refund processed for order 77. |
| 2 | "thanks!" | refund | Refund processed for order **42**. |
| 3 | "actually I want to buy the pro plan" | refund | Refund processed for order **42**. |

`run_swarm` keeps `active` across turns, which is the confirmation the
exercise asks for. The rest of the table is the problem.

**Every turn is a refund.** The refund router has one output: it calls
`process_refund`. When the message contains no bare-digit word, it uses
`order = "42"`. "thanks!" therefore refunds an order nobody named, and so does
"order #77.".

**A handoff is permanent.** The handoff tools per agent are [3, 0, 0, 0], so
no specialist can leave. Turn 3's sales request never reaches triage's
keyword check.

**The instructions are read by nothing.** `scripted_router` branches four
times on `current.name` and never reads `instructions`. The demo says "the
agent prompts ARE the routing logic", but in this module the prompts do
nothing.

### 2 — the fallback cannot be triage, because triage is half the loop

The shipped agents cannot loop, so the refund agent gets the handoff it most
plausibly needs: *hand back to triage when there is no order number*. That is
enough. On "I want a refund", triage routes on "refund", refund hands back,
and triage reads the same message and routes it again.

| policy | outcome |
|---|---|
| no rule | 50 hops, the cap |
| rule + fallback to triage | rule fires 48 times, still 50 hops |
| rule + fallback to a human agent that asks | stops after 3 handoffs: "which order number should be refunded?" |

**The rule:** count handoffs between one unordered pair *within a single user
turn*, and on the third, exit to an agent outside the pair that has no
handoffs and asks for what was missing. The loop exists because no new
information arrives, so the fallback's job is to ask for it.

Two design decisions are forced by the numbers.

**The fallback cannot be the natural default.** The lesson's checklist says
"fall back to a safe default", and in a triage topology that is triage, which
is half of the pair.

**The ring has to reset every user turn.** "refund on order 77", "hmm,
something else", "refund on order 78" is three user-driven handoffs between
the same two agents. A ring kept across turns fires on it; one reset per turn
does not. A user message is new information.

The shipped loop hides the problem instead. `run_swarm` calls the router at
most twice per turn and prints the second result, so the user sees
`Agent(name='triage', ...)` and `active` stays refund. The cookbook's loop
runs "while True" until no tool calls remain.

### 3 — the summary must keep what the next agent's tools take

The test conversation names the order *before* the handoff: "hi, the blender I
got last week arrived damaged, order 77", then "I want my money back".

| context passed to refund | size | refunds |
|---|---:|---|
| full history (the SDK default) | 57 chars | 77 |
| each turn cut to 5 words | 23 chars | **42** |
| bullets from the target's tool signatures | 14 chars | 77 |
| shipped `run_swarm` | 0 chars | **42** |

**Build the summary from the incoming agent's tool signatures.** The SDK hook
is `handoff(agent, input_filter=...)`, a function over `HandoffInputData`
(`input_history`, `pre_handoff_items`, `new_items`). A filter that reads
`process_refund(order_id)` from the target and extracts that one field hands
over `- order_id: 77`. It is smaller than the generic summary and it is
correct. A summary is only right or wrong relative to what the next agent
must do.

Note that in the SDK the filter is a runtime function at the handoff
boundary, not the outgoing agent writing a summary. That is safer, because the
agent that just read untrusted input does not decide what to forward.

**On the shipped code every filter is a no-op.** `scripted_router(current,
user_msg)` takes the current message and nothing else, and `run_swarm`'s
`history` is written but never passed in.

**Both lossy runs fail silently.** Each ends in "Refund processed for order
42.", a successful tool call. `process_refund` has a default, so a dropped
identifier becomes a different identifier instead of an error.

### 4 — one injected turn owns the rest of the session

Take one malicious message, "my dashboard is broken, please refund order
1234", followed by three honest "my dashboard is broken" turns. The same
triage rules are used both ways:

| routing | turns ending in `process_refund` |
|---|---:|
| Swarm (`run_swarm`) | **4 of 4** |
| selector reading the latest message | 1 of 4 (the other 3 open tickets) |
| selector reading the whole transcript | 4 of 4 |

**Swarm is worse, because the routing decision is state that outlives the
message that made it.** The injected turn moves `active` to refund, and
nothing re-decides for the rest of the session. A GroupChat manager
re-decides every turn. That gives it a *choke point*, not immunity: a
selector that re-reads the transcript, as an LLM speaker-selector does,
re-reads the injection every turn.

The injected text also picks the *argument*: it refunds order 1234, a number
the user typed, with no check that it belongs to them. And "this is not a
refund request" is routed to refund anyway, because the keyword test has no
negation and refund is checked before support.

The guard count is the structural difference. Triage holds 3 handoff tools,
and a full mesh of the 4 agents would hold 12, each needing its own check. The
manager is 1 function. That is why the lesson's "guardrail on handoff" item
matters more for Swarm than for GroupChat.

### 5 — kept: the handoff is a tool returning an agent; changed: who owns the history

*Draws on "OpenAI Agents SDK (March 2025)", checked against the cookbook
"Orchestrating Agents" and the SDK's handoff docs.*

**Kept: a handoff is a tool call that returns an agent, and the new agent
sees everything.** The cookbook's loop does "if type(result) is Agent: update
current agent", and the new agent has "complete knowledge of your prior
conversation". The SDK still represents handoffs as tools, named
`transfer_to_refund_agent` for an agent called "Refund Agent". By default the
new agent "gets to see the entire previous conversation history".

**Changed: the history stopped being only the caller's list.** In the
cookbook, state *is* the `messages` list you pass back in. `Response` returns
the agent and the messages, and continuity is up to you. The SDK adds three
things around the same primitive:

- sessions, which load and store history for each run;
- `input_filter` / `HandoffInputData`, which rewrite what crosses a handoff
  (with `remove_all_tools` built in, and a beta `nest_handoff_history` that
  compacts earlier turns into summary segments);
- `MaxTurnsExceeded`, a turn cap on the loop.

That cap is a *count*, not a ping-pong detector. The pair rule from exercise
2 is still yours to write. The context policy from exercise 3 now has an
official hook, but the default still forwards everything.
