<!-- generated:start -->
# 16-multi-agent-and-swarms / 02-fipa-acl-heritage

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/02-fipa-acl-heritage/) · upstream spec
`phases/16-multi-agent-and-swarms/02-fipa-acl-heritage/docs/en.md`

```bash
uv run demo practice run 02-fipa-acl-heritage --ex 1
uv run demo explain 02-fipa-acl-heritage --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/02-fipa-acl-heritage
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Observe the round-trip encoding. Identify which FIPA performative corresp… | code | T0 | `ex01_the_round_trip_has_no_return_leg.py` |
| 2 | Extend the contract-net demo with a `cancel` performative that lets the manager withdraw the… | code | T0 | `ex02_a_retry_reruns_an_auction_whose_answer_cannot_change.py` |
| 3 | Read FIPA ACL Message Structure (http://www.fipa.org/specs/fipa00037/) sections 4.1–4.3. Pick… | explain | T0 | prose, below |
| 4 | Read Liu et al., arXiv:2505.02279. For each of MCP, A2A, ACP, ANP, list the FIPA performative… | explain | T0 | prose, below |
| 5 | Design a minimal JSON-Schema for the `content` field of a `request` performative in your own… | code | T0 | `ex05_the_schema_cannot_key_on_the_performative.py` |
<!-- generated:end -->

## Answers

### 1 — the round trip has no return leg

`tools/call` is a **request**: invoking a tool asks the receiver to act.
`resources/read` is a **query-ref**: it asks for the referent of an expression
rather than for an action. A2A task creation is a **request** again. Three
message types land on two performatives, so what separates an MCP tool call
from an A2A task is the `:protocol` and `:ontology` slots, not the
performative — which is the lesson's real point, stated more precisely than
its own takeaway line does.

The banner over the demo reads `Round-trip: 2026 JSON-RPC / REST <-> FIPA-ACL
envelope`. The module defines **four** functions named `*_to_acl` and **zero**
named `acl_to_*`. The arrow only runs one way.

It could not run the other way as written. The MCP message sets
`:language JSON`, and `render()` emits `:content` with `{self.content!r}` —
Python's repr. The line is `{'symbol': 'IBM'}`, single-quoted, and
`json.loads` raises `JSONDecodeError`. The envelope declares a language it
does not emit, so there is nothing for a decoder to parse.

`PERFORMATIVES` holds sixteen names and `__post_init__` enforces them. The
module constructs seven. The nine left over — agree, cancel, confirm,
disconfirm, failure, inform, not-understood, query-if, refuse — are a
whitelist and nothing else, which is how exercise 2 comes to ask for one that
is already there.

### 2 — a retry reruns an auction whose answer cannot change

The failure case is the **committed worker**. A retry re-issues the call for
proposals: three more envelopes on a nine-message log, and it re-awards
`worker-b`, because the bids are fixed data and `min(price + eta/10)` is
deterministic. Retrying answers "I got no reply". It has nothing to say to a
worker that replied, won, and started. The cancel is one message —
`performative="cancel"`, scheduler to `worker-b`, conversation `cn-1` — and it
is the only envelope in either log that tells a committed worker to stop.

The extension is a call site, not a protocol change: `cancel` is already in
`PERFORMATIVES` and `__post_init__` already accepts it.

What the extension cannot do is say *what* it cancels. `ACLMessage` has nine
fields. `reply_with` is one; `in_reply_to` is not. `reply_with` is set by five
constructors and read twice, both inside `render()` — printed, never
consulted. So the cancel names a conversation and cannot name the
`accept-proposal` it revokes, and in a conversation with several awards there
is no way to say which.

While running the auction, its scoring is worth reading:

| divisor | worker-a | worker-b | worker-c | winner |
|---:|---:|---:|---:|---|
| 5 | 6.60 | 7.00 | 6.00 | worker-c |
| 7.5 | 5.40 | 5.33 | 5.33 | flip |
| 10 (shipped) | 4.80 | 4.50 | 5.00 | worker-b |

`price + eta_minutes / 10` prices ten minutes at one unit of money and says so
nowhere. The shipped divisor sits 1.3× above the point where the answer
changes. And `worker-a` wins at no divisor at all: beating `worker-b` needs
one below 7, beating `worker-c` needs one above 8.

### 3 — not-understood is the one act whose JSON-RPC analog changes its addressee

*Draws on "The twenty FIPA performatives (partial list)".*

Taking `not-understood`, which this lesson declares and never builds.

FIPA defines it as a communicative act like any other: agent *i* sends
`not-understood` to agent *j* with content naming the offending act and a
reason. It is a peer message. It can be sent about any message, at any time,
and it is itself subject to the same envelope — sender, receiver,
conversation, `:in-reply-to`.

The JSON-RPC analog is the error object: `-32601 Method not found` for an
unrecognised act, `-32602 Invalid params` for a well-named act with
unreadable content, `-32600 Invalid Request` for a malformed envelope. MCP
uses all three.

The difference is structural rather than semantic. A JSON-RPC error is not a
message an agent sends; it is the *response half* of a request, and it must
carry the `id` of the request it answers. A server cannot volunteer
`-32601` about something it was not asked, and it cannot send one to a third
party. FIPA's `not-understood` is addressable; JSON-RPC's is bound to a
correlation id. That is the same asymmetry the module has already run into:
this `ACLMessage` has `reply_with` and no `in_reply_to`, so it can raise the
act and cannot bind it to what it is about.

### 4 — the negotiation family is the part of this module with no encoder

*Draws on "The 2026 specs, mapped to speech-act heritage".*

The checkable version of this question is in the module itself. It builds
seven performatives. Three — `request`, `query-ref`, `subscribe` — have
encoders from real 2026 wire formats: `mcp_tools_call_to_acl`,
`mcp_resources_read_to_acl`, `a2a_task_create_to_acl`,
`a2a_subscribe_to_acl`. The other four — `cfp`, `propose`,
`accept-proposal`, `reject-proposal` — appear only inside `ContractNet`, and
there is no `contract_net_to_mcp` or `contract_net_to_a2a`, because there is
nothing to write one against.

So the split the survey describes is visible in this file without reading it:
the **directive** family (request, query-ref) and the **subscription** family
survive into MCP and A2A, and the **commissive/negotiation** family does not.
Following Liu et al.'s taxonomy across the four protocols it covers:

| protocol | keeps | drops |
|---|---|---|
| MCP | directive (`tools/call` → request, `resources/read` → query-ref), error acts as JSON-RPC codes | negotiation entirely; no cfp, no propose |
| A2A | directive, subscription (SSE → subscribe), and an explicit `tasks/cancel` | negotiation; task assignment is addressed, not auctioned |
| ACP | directive and inform over REST | negotiation |
| ANP | directive plus identity/discovery, which FIPA put in the Directory Facilitator rather than in ACL | negotiation |

The per-protocol rows follow the survey's categorisation; the column that is
verifiable from this repository is the last one, and every protocol agrees on
it. Contract Net is the oldest and most cited thing FIPA standardised, and
none of the four 2026 protocols carries it. What replaced it is a scheduler
that picks an agent — the manager's job, with the bidding removed.

### 5 — the schema cannot key on the performative

The lesson builds two `request` messages. One carries `{'symbol': 'IBM'}`, an
object. The other carries `'def f(x): return x'`, a string. So:

| schema | keyed on | admits | keywords |
|---|---|---:|---:|
| `{"type": "object"}` | performative | 1 of 2 | 1 |
| `{}` | performative | 2 of 2 | 0 |
| `lookup_stock` + `review-python` | ontology | 2 of 2 | 5 |

The only performative-keyed schema that accepts the lesson's own messages is
the empty one, which checks nothing. A useful schema has to key on
`:ontology`.

**What it gives you**: a rejection before dispatch. A `tools/call` with
`arguments` empty builds a perfectly valid `ACLMessage` — `__post_init__`
raises nothing — and the `lookup_stock` schema rejects it on the missing
`symbol`. "This call needs a symbol" is a sentence natural language can say
and cannot make checkable; the schema is the same sentence in a form the
receiver runs.

**What it costs**: one artefact per ontology. This module already uses five,
and every new tool adds another. The thing worth constraining is the
argument list, and the argument list is precisely what the performative
abstracts away — so the schema buys its precision by giving up the generality
that made the envelope worth having.

That trade is also why `__post_init__` stops where it does: it validates one
field of nine, against a sixteen-name whitelist, and leaves `language`,
`ontology` and `protocol` as free strings. Which is how an envelope comes to
claim `:language JSON` over a Python repr with nothing noticing.
