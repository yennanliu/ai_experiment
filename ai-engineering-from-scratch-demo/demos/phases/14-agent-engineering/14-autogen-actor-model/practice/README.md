<!-- generated:start -->
# 14-agent-engineering / 14-autogen-actor-model

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/14-autogen-actor-model/) · upstream spec
`phases/14-agent-engineering/14-autogen-actor-model/docs/en.md`

```bash
uv run demo practice run 14-autogen-actor-model --ex 1
uv run demo explain 14-autogen-actor-model --ex 1
uv run pytest demos/phases/14-agent-engineering/14-autogen-actor-model
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a dead-letter queue: when a handler raises, park the failing message for human inspection… | code | T0 | `ex01_the_dlq_is_shipped_and_the_sender_is_never_told.py` |
| 2 | Implement `SelectorGroupChat`: a selector actor picks who processes the next message based on… | code | T0 | `ex02_the_selector_needs_a_conversation_the_runtime_does_not_keep.py` |
| 3 | Add distributed transport: swap the in-process queue for a JSON-over-HTTP server so actors ca… | code | T0 | `ex03_the_runtime_reference_is_what_cannot_cross_a_process.py` |
| 4 | Wire an OTel span per message (or a no-op stand-in). Emit `gen_ai.agent.name`, `gen_ai.operat… | code | T0 | `ex04_the_span_ends_where_the_causal_chain_continues.py` |
| 5 | Read AutoGen v0.4's architecture post. Port your toy to the real `autogen_core` API. What did… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

All five are T0 and stdlib. Four ship code; exercise 5 reads AutoGen v0.4's
architecture post, so it is answered in prose.

The recurring finding is that **the runtime records what it did and never
what it stopped doing**. A handler that raises has its message parked in
`dead_letters` and the *sender* is told nothing, so a request/response pair
hangs with no error anywhere: exercise 1 measures a checklist waiting at
**2** of **3** results with `consensus` left at `True`. Exercise 4 finds the
same hole from the tracing side — the crash produces one span and the reply
that never came produces none, so a trace of a hung conversation looks like a
short one.

The second thread is that **`Message` has five fields and every extension
needs a sixth**. No conversation id, so a selector cannot separate two chats
(exercise 2). No parent id, so a span per message gives eight roots until the
runtime threads causality itself (exercise 4). No node id in `mid`, so two
runtimes both mint `m001` and every id collides on merge (exercise 3).

What holds: fault isolation, which is the property the demo set out to show.
The reviewer's crash really does leave the other messages processing, and the
JSON round-trip in exercise 3 really is lossless for the demo's own traffic.

### 1 — the DLQ is shipped, and the sender is never told

**ANSWER: the demo hits the DLQ once in 8 messages, 12%.** The user sends
**2** and the actors generate the other **6**, so the honest denominator is
the whole message count rather than the caller's. Adding one message to an
unregistered actor takes it to **2** of **9**, **22%**. The queue itself
needed no building — `Runtime` already parks failures and keeps going.

**FINDING: the queue mixes two different failures.** `dead_letters` holds
`(message, reason)` pairs and the reason is free text:
`no actor 'ghost'` beside `RuntimeError: simulated handler failure`. The
first belongs to whoever addressed the message and the second to whoever
wrote the actor; separating them for triage means parsing a string.

**FINDING: nobody is told.** Parking notifies neither the sender nor a
supervisor. A `review` that raised is simply never answered, and the
checklist waits with **2** of **3** results — a hang indistinguishable from
work still in flight. This is the cost of the fault isolation the demo
advertises: the failure is contained so thoroughly that it is also
invisible.

**FINDING: `consensus` is set before the results are in.**
`ChecklistAgent` assigns `consensus = True` as soon as every result *so far*
is ok, so after the first clean snippet it reads **True** with **1** of
**3**. The final message corrects it to **False**. Any reader in between —
including the hung case above, which never reaches the third result — sees a
verdict that has not been earned.

### 2 — the selector needs a conversation the runtime does not keep

**ANSWER: a selector actor that owns the transcript and readdresses every
turn.** Over **5** turns it routes to **3** distinct specialists and ends
holding **10** transcript entries, because every reply comes back through it.

**FINDING: `Message` has no conversation id.** Its five fields are `sender`,
`recipient`, `topic`, `body`, `mid`. Two interleaved chats through one
selector are distinguishable only by `sender`, and `sender` is the selector
for every reply — so the transcript the selector accumulates is the only
place a conversation exists at all.

**FINDING: the selector serialises the group.** Five turns cost **15**
messages where addressing the same specialists directly costs **10**. That
is the standing price of central selection, and it is the concrete form of
the note in the lesson that the LangChain and AutoGen communities both drift
toward direct tool calls for context control.

**FINDING: the selector chooses all five turns before any answer arrives.**
`Runtime` holds one *global* FIFO queue, so the five `turn` messages are
delivered before the first `answer` returns: the number of answers in the
transcript at each choice is **0, 0, 0, 0, 0**. A selector that routes on
conversation state has none at the moment it needs it, and the continuity
rule never fires. The lesson's own framing — "inbox + transport is the same
abstraction" — presumes a per-actor inbox; this runtime has one queue, and
`SelectorGroupChat` is the pattern that notices.

### 3 — the runtime reference is what cannot cross a process

**ANSWER: a JSON codec and a loopback transport.** All **8** messages of the
demo encode and decode back into equal `Message` objects — **8** of **8** on
all **5** fields — producing the same **16** trace lines and the same single
dead letter. The socket is deliberately not built: it is the half that
works.

**FINDING: the runtime reference is the blocker.** `Actor.receive` takes
`(message, runtime)` and both shipped actors call `runtime.send(...)` inside
their handlers — **2** call sites in **2** actors. A remote actor needs a
proxy runtime whose `send` posts instead of appending, which means the
`runtime` parameter is a capability handed to every actor and the port's real
work is deciding what that capability becomes over a wire.

**FINDING: a body that is not JSON does not survive.** Of three probes, a
list survives, a tuple comes back as a list — a silent type change that
`==` on the message will not catch if the handler compares by index — and a
set raises `TypeError` at `json.dumps`. The demo's own bodies are a list and
a dict, so nothing in the shipped code would have shown this.

**FINDING: the message ids stop being unique.** `Runtime.counter` is a field
on the runtime, so a second node also mints `m001` and all **8** demo ids
collide on merge. The DLQ is keyed by nothing else, so two nodes' dead
letters cannot be told apart after collection. A distributed `mid` has to be
a pair.

### 4 — the span ends where the causal chain continues

**ANSWER: a span per delivered message, carrying both required attributes.**
**8** spans, **8** with `gen_ai.agent.name` and `gen_ai.operation.name`,
operations `handle` and `dead_letter`.

**FINDING: without threading, every span is a root.** Spanning each message
as it is dequeued gives **8** roots, because `Message` carries **0** parent
fields. Threading the parent through `Runtime.send` — the id of the message
being handled when the send happened — gives **2** roots, one per externally
sent message, in a tree **3** levels deep. The causality exists; it is in the
runtime's call stack and not in the message.

**FINDING: the span ends before its consequences start.** A handler's
`runtime.send` only appends to the queue, so between a `review` span and its
`review_result` child the runtime handles **2** unrelated messages. The
span's duration measures the handler, not the work it caused — which means a
latency chart built from these spans describes the dispatcher and not the
conversation. This is the actor model's decoupling showing up as a
measurement problem.

**FINDING: the failing message gets a span and the sender gets nothing.**
The crash is **1** span with `gen_ai.operation.name = dead_letter`, and the
request whose reply never came has none. Tracing what the runtime *did* is
easy; tracing what it stopped doing needs the sender notified, which is
exercise 1's missing piece arriving from the observability side.

### 5 — the layer you skipped is the one with the subscriptions

*Cites "Three API layers".*

**The toy is a partial Core and nothing else, and the gap that matters is
inside Core rather than above it.**

AutoGen v0.4 splits into Core (`AgentRuntime`, `Agent`, `Message`, `Topic`),
AgentChat (`AssistantAgent`, `RoundRobinGroupChat`, `SelectorGroupChat`) and
Extensions (model clients, tools, memory). The toy implements a runtime, an
agent base class and a message — three of Core's four names. The one it does
not implement is `Topic`, and that omission is the source of three of the
four findings above.

**What `Topic` buys, measured against what the toy does instead.**

1. **Addressing becomes a subscription rather than a name.** The toy's
   `Runtime.send(sender, recipient, ...)` requires the sender to know the
   recipient's string name, which is why exercise 1's `no actor 'ghost'`
   is a dead letter rather than a compile-time or registration-time error,
   and why exercise 2's selector must hold the routing table itself.
   Publishing to a topic with subscribers registered at startup moves that
   failure to registration.
2. **Conversations get an identity.** Exercise 2's finding — `Message` has
   **5** fields and **0** conversation ids, so two chats through one selector
   are indistinguishable — is exactly what a topic id solves. The real Core
   carries a topic *and* a source, so `RoundRobinGroupChat` and
   `SelectorGroupChat` can exist as thin AgentChat wrappers rather than as
   the transcript-hoarding actor exercise 2 had to write.
3. **Per-actor inboxes become expressible.** The toy's single global FIFO
   queue is what makes exercise 2's selector choose all five turns before any
   answer arrives. Nothing in the toy's shape prevents per-actor queues; the
   shape just does not have them, and the difference is invisible until a
   pattern depends on request/response interleaving.

**What I would list as skipped-and-it-matters, in production order.**

- **Notifying the sender on failure.** Exercise 1: the DLQ is a graveyard
  with no next of kin. The real runtime's intervention handlers and the
  framework's retry policy exist for this; without them, fault isolation and
  a silent hang are the same observable.
- **Async everything.** The toy is a `while self.queue` loop, so "natural
  concurrency" — the second of the three consequences the lesson lists — is
  not present at all. Every finding about ordering in exercise 2 is
  downstream of that, and it is the one thing a port to `autogen_core`
  changes for free.
- **Serialization contracts.** Exercise 3: the port needs a declared message
  schema, not `dataclasses.asdict`. A tuple silently becoming a list is the
  kind of thing that passes every test written in one process.
- **Identity across nodes.** Exercise 3: `mid` is a per-runtime counter, and
  `AgentId` in real Core is `(type, key)` for exactly this reason.
- **Trace context in the message.** Exercise 4: threading a parent id through
  `send` reproduces the tree, but only for sends that happen inside a
  handler. Anything that enqueues from outside a span — a timer, a retry, an
  external caller — reverts to a root, and the real fix is a trace context
  field on the message.

**And the part worth being relaxed about:** the lesson's own note that
AutoGen is in maintenance mode while Microsoft Agent Framework takes over,
and that "the actor model is the durable idea". Everything above is a
statement about `Message` and `Runtime` having too few fields — which is a
portable finding, because the successor framework has the same two objects
under different names.
