<!-- generated:start -->
# 13-tools-and-protocols / 07-building-an-mcp-server

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/07-building-an-mcp-server/) · upstream spec
`phases/13-tools-and-protocols/07-building-an-mcp-server/docs/en.md`

```bash
uv run demo practice run 07-building-an-mcp-server --ex 1
uv run demo explain 07-building-an-mcp-server --ex 1
uv run pytest demos/phases/13-tools-and-protocols/07-building-an-mcp-server
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Remove capabilities from one request and prove the server does not reuse the previous request… | code | T0 | `ex01_statelessness_is_proved_by_the_absence_of_a_place_to_keep_it.py` |
| 2 | Reverse the `TOOLS`, `PROMPTS`, and note insertion order. Confirm all list results remain sta… | code | T0 | `ex02_three_lists_are_stable_and_a_created_note_lands_anywhere.py` |
| 3 | Add a destructive `notes_delete` tool and require an authorization check inside the executor.… | code | T0 | `ex03_the_hint_is_a_hint_because_no_handler_ever_reads_it.py` |
| 4 | Add `resources/templates/list` with `ttlMs`, `cacheScope`, and deterministic ordering. | code | T0 | `ex04_a_template_list_is_public_because_it_holds_no_instances.py` |
| 5 | Build a separate legacy adapter for `2025-11-25`. Add tests proving a modern request never en… | code | T0 | `ex05_the_two_eras_are_separated_by_a_key_neither_one_shares.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib, and all five ship code.

Lesson 13.06's server had no state at all. This one has three lists, a mutable
note store, and five scoped cache policies — so the exercises stop being about
whether the protocol works and start being about **which of the server's
guarantees are enforced by code and which are advertised and then not kept.**

### 1 — statelessness is proved by the absence of a place to keep it

**ANSWER: `-32602` on every method, regardless of what came before** —
`complete / -32602 / complete` across three requests, and all **6** methods
reject the omission identically.

**FINDING: this server does keep state between requests, and it is not protocol
state.** `NOTES` is a module dict `notes_create` mutates: `resources/list`
returns **3** entries before a create and **4** after. "Stateless" here is a
claim about `_meta` only, and the exercise's proof strategy would be false
applied to the whole server.

**FINDING: the two kinds are distinguished by where they are read.**
`validate_request` reads `params._meta`; all three executors read `NOTES` and
touch `_meta` in **0 of 3** cases. Protocol context is re-established per
message; application data lives behind the handlers.

**FINDING: and the required capability is still never consulted.** `{}` passes,
and the **3** advertised capability groups are read by no handler.

### 2 — three lists are stable, and a created note lands anywhere

**ANSWER: all three lists are unchanged by reversal.** Two sort by `name` and
one sorts `NOTES.items()` by id, so insertion order never reaches the wire.

**FINDING: a created note sorts by a random hex id.** `note-{uuid4().hex[:6]}`
against a lexicographic sort — an id beginning `0`–`9` lands *before* the seeded
`note-1`. Over **200** creates the new note appeared at positions **0, 1, 2 and
3**. Deterministic given the ids; unreproducible across runs.

**FINDING: stability and freshness are different, and the wrong one is
advertised.** A create takes `resources/list` from 3 to 4 while the server
declares `resources.listChanged: False` and the list caches for **10,000 ms**. A
client honouring both serves a stale list for up to ten seconds and is never
told. Sorting made the list stable; nothing made it current.

**FINDING: the sort key is absent from the payload on exactly one list.**
`resources/list` sorts by note id and returns `uri, name, mimeType` — a client
cannot reproduce the ordering from what it received.

### 3 — the hint is a hint because no handler ever reads it

**ANSWER: `notes_delete` with the check inside the executor, denying by
default.** Unauthorised → `rpc_error -32003`, note survives. Authorised →
`isError: False`, resources drop 3 → 2.

**FINDING: no handler reads an annotation.** Searching every `handle_*` for
`annotations` finds **0**, and `handle_tools_call` looks up `TOOL_EXECUTORS` by
name without consulting the tool record. `destructiveHint` could not have been
anything but advisory.

**FINDING: the two error channels mean different things, and the exercise does
not say which to use.**

| raised as | becomes | the model |
|---|---|---|
| `ValueError` | `isError: True` tool result | reads it, may retry |
| `RpcProblem` | JSON-RPC `error` | never sees it as content |

`handle_tools_call` catches `(KeyError, TypeError, ValueError)`. A denial raised
the first way invites a retry; raised the second way it terminates the call.
Authorization is the second kind.

**FINDING: the destructive tool is the only one whose failure is
unrecoverable.** Retried after success it reports a missing note. The annotation
that matters for retry is `idempotentHint` — set on **2** of the 3 original
tools, and also read by nobody.

### 4 — a template list is public because it holds no instances

**ANSWER: `resources/templates/list`, `public`, 60,000 ms, sorted by
`uriTemplate`** — stable under reversal, `-32601` before installation and after
removal.

| method | scope | ttl |
|---|---|---:|
| `server/discover` | public | 3,600,000 |
| `tools/list` | public | 60,000 |
| `prompts/list` | public | 60,000 |
| **`resources/templates/list`** | **public** | **60,000** |
| `resources/list` | private | 10,000 |
| `resources/read` | private | 5,000 |

**FINDING: it is the only `resources/*` method that can be public.** A template
is a URI grammar — a function of the server's routing and of nothing the caller
owns.

**FINDING: the ttl follows the same rule.** Scope and freshness are one
decision: the server's own shape gets the long ttl, the caller's data the short
one.

**FINDING: and the templates describe a route `resources/read` does not
implement.** `notes://tag/{tag}` expands to a URI whose `removeprefix` leaves
`tag/design`, giving **-32602 Resource not found**. Advertising a template is not
serving it, and nothing cross-checks the two.

### 5 — the two eras are separated by a key neither one shares

**ANSWER: a selector on `params._meta`, an adapter holding connection state, and
8 cases.** Four route modern, four legacy; none routes both, none neither.

**FINDING: the adapter needs state the modern server has nowhere to put.**
Uninitialised it refuses `tools/list` with `-32002`; after `initialize` it
answers with capabilities the later request never re-sent — exactly what exercise
1 proved the modern path cannot do. **The eras differ by one dict that outlives
a request.**

**FINDING: rejecting the legacy entry point is not the same as routing.**
`initialize` through the lesson's own `dispatch` gives `-32601 Method not
found` — a modern rejection, not a handover, and indistinguishable from a
genuinely unknown method.

**FINDING: and a stale version is a modern error, not a legacy handover.**
`2025-11-25` on a modern-shaped message still routes modern and answers
`-32022`, because the selector reads the metadata key rather than the version
value. Each era is total over its own inputs and empty over the other's.
