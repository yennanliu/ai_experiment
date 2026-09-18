<!-- generated:start -->
# 11-llm-engineering / 14-model-context-protocol

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/11-llm-engineering/14-model-context-protocol/) · upstream spec
`phases/11-llm-engineering/14-model-context-protocol/docs/en.md`

```bash
uv run demo practice run 14-model-context-protocol --ex 1
uv run demo explain 14-model-context-protocol --ex 1
uv run pytest demos/phases/11-llm-engineering/14-model-context-protocol
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a `subtract` tool and confirm `tools/list` remains alphabetically ordered. | code | T0 | `ex01_sorted_is_codepoint_order_so_a_capital_s_precedes_a.py` |
| 2 | Remove the protocol-version key and verify Invalid Params (`-32602`). Then send the well-form… | code | T0 | `ex02_discovery_is_gated_on_the_version_it_exists_to_report.py` |
| 3 | Add a server-minted `draftId` to a create operation, then require it as an argument to update… | code | T0 | `ex03_the_draft_id_travels_in_the_arguments_and_the_protocol_never_sees_it.py` |
| 4 | Return `input_required` from a tool that needs user confirmation. Retry the original call wit… | code | T0 | `ex04_the_destructive_hint_is_published_and_unenforced.py` |
| 5 | Sketch a dual-era stdio client. Treat a result or recognized modern error as modern, and perm… | code | T0 | `ex05_four_of_the_five_recognized_errors_are_standard_json_rpc.py` |
<!-- generated:end -->

## Answers

The lesson is pure stdlib, so all five exercises are **T0** and run in CI. Each
solution builds its own `MCPServer` carrying the lesson's primitives rather
than mutating the module-level `server` singleton.

Two things recur. `handle` validates `_meta` **before** it dispatches, so every
method — including the one that reports the protocol version — is gated on the
protocol version. And every guarantee in the lesson applies only to messages
with an `id`: `handle` returns `None` for a notification before any validation
runs.

### 1 — `sorted()` is codepoint order, so a capital S precedes a

```text
after registering subtract      ['add', 'delete_user', 'subtract']       ordered
plus Subtract, _hidden, 2fa     ['2fa', 'Subtract', '_hidden', 'add', …]  still sorted()
```

**ANSWER: `tools/list` stays ordered**, and not because of care at
registration — the handler sorts on every call, so `subtract` lands third by
name rather than by arrival.

**FINDING: the ordering is ASCII, not alphabetical.** Digits before capitals,
capitals before underscore, underscore before lowercase. A client rendering the
list as-is shows `Subtract` above `add`. `key=str.casefold` is the one-word fix.

**FINDING: nothing validates `inputSchema`.** `tools/list` publishes
`integer` for both of `add`'s arguments and `tools/call` hands `arguments`
straight to the handler as `**kwargs`, so `add(a="x", b="y")` returns
`{"sum": "xy"}` with `isError: false`. The schema is documentation.

**FINDING: a missing required argument is an internal error.** `add(a=1)`
raises `TypeError`, which `except Exception` turns into **-32603**; an unknown
tool *name*, checked one line earlier, is **-32602**. The same class of mistake
gets two codes depending on who noticed it.

**FINDING: registering a name twice replaces it silently** — the registry is a
dict keyed by name, and the second registration wins with no error.

### 2 — discovery is gated on the version it exists to report

| request | code | carries `data`? |
|---|---:|---|
| no `_meta` | -32602 | no |
| `_meta` without the key | -32602 | no |
| the key holding an integer | -32602 | no |
| version `2025-11-25` | **-32022** | **yes** — `supported` + `requested` |

**ANSWER: all three malformed requests return -32602** with three different
messages, and `2025-11-25` returns -32022 whose `data` echoes `requested` and
lists `supported`; retrying with `supported[0]` is accepted.

**FINDING: `server/discover` is not exempt.** The method that exists to tell a
client which versions the server speaks is refused with the same -32022 unless
the client already knows one. For a client with no prior knowledge, the -32022
body *is* the discovery mechanism and the well-formed result is the
confirmation afterwards.

**FINDING: -32022 is the only self-describing error.** A client that omitted
the key learns the key's name only by parsing the English message.

**FINDING: `supported` has one entry**, so "choose from `supported`" is not a
choice — a client holding a list of revisions can only match or fail.

**FINDING: notifications skip validation entirely.** A notification carrying
the version `1999-01-01` is accepted silently and does nothing.

### 3 — the draftId travels in the arguments and the protocol never sees it

**ANSWER: `create_draft` mints `draft-0001`; `update_draft` requires it.** The
id is on the wire in exactly one place — `params.arguments.draftId` — minted
and interpreted by the handler. `handle`, `_validate_metadata` and `_complete`
never read it.

**FINDING: the id is unscoped, which is what makes it application state.** A
second `MCPClient` with its own counter updates the first client's draft and is
accepted; both clients send request id **1** first, so the server could not
tell them apart if it wanted to.

**FINDING: there is nowhere to put a session.**

| | contents |
|---|---|
| request `_meta` | `protocolVersion`, `clientCapabilities`, `clientInfo` |
| response `_meta` | `serverInfo` |
| `MCPServer` state | `name`, `server_info`, `tools`, `resources`, `prompts` |
| `MCPClient` API | `request` |

A session needs a new `_meta` key, a store and a check in
`_validate_metadata`. None of the three exists.

**FINDING: `resources/read` advertises mutable state as cacheable** —
`ttlMs: 30000`, `cacheScope: "private"`, applied unconditionally. `tools/call`
is the one method that is *not* cacheable, so the mutation is fresh and the
read of it is not.

**FINDING: the required argument is enforced by Python.** A missing `draftId`
(`TypeError`) and an unknown one (`KeyError`) both arrive as **-32603** with no
data, so a client cannot tell "you forgot the id" from "that draft is gone".

### 4 — the destructive hint is published and unenforced

```text
ask    -> resultType "input_required", inputRequests [confirmed], requestState, deleted []
retry  -> resultType "complete", {"deleted": 7},                              deleted [7]
replay -> resultType "complete" again,                                        deleted [7, 7]
```

**FINDING: `input_required` cannot come from a tool handler.** `_complete`
writes `"resultType": "complete"` into every result and **all six methods**
answer `complete`; `tools/call` also hardcodes `isError: false` and
JSON-encodes the handler's return value into `content[0].text`. Elicitation is
a change to the result builder, not to the tool.

**FINDING: the server never validates the extra params.** `inputResponses` and
`requestState` sent to the unmodified server produce an ordinary `complete`
result — `handle` reads `name`, `arguments`, `_meta` and ignores the rest. The
retry contract is application convention, so a typo in `requestState` is not an
error.

**FINDING: "a new ID" is a convention the server cannot enforce.** Replaying
the retry with the *original* id deletes a second time. And the alternative the
exercise warns against is not merely discouraged: `MCPClient` exposes one
method and `handle` returns one response per message, so no channel exists for
a server-to-client request.

**FINDING: the destructive annotation is decoration.** `tools/list` publishes
`{"destructiveHint": true}` for `delete_user`, and the `tools/call` branch
never reads `tool.destructive` — an unconfirmed delete straight to the server
returns `complete`.

### 5 — four of the five recognized errors are standard JSON-RPC

| probe | code | verdict | version learned |
|---|---:|---|---|
| `server/discover` @ current | result | modern | ✓ (but needs the answer to ask) |
| `server/discover` @ `2025-11-25` | **-32022** | modern | **✓** |
| `server/discover`, no `params` | -32602 | modern | — |
| `initialize` | -32601 | modern | — |
| not an object | -32600 | modern | — |
| sent as a notification | *(silence)* | **fallback** | — |
| hand-built legacy | -32000 | **fallback** | — |

**FINDING: only one probe proves modernity and yields the version at once** —
`server/discover` at a deliberately wrong revision. The rest settle the era and
say nothing about the version.

**FINDING: `initialize` answers -32601, which the rule reads as modern.** The
legacy handshake is just an unknown method here, so sending it first detects a
JSON-RPC server, not an old one. The exercise's ordering is load-bearing.

**FINDING: a notification is indistinguishable from a timeout** — and would
send a dual-era client into the fallback against a fully modern server.

**FINDING: four of the five recognized codes are in the JSON-RPC 2.0
pre-defined block.** -32600, -32601, -32602 and -32603 are emitted by pre-2026
servers too; only **-32022** belongs to this protocol. "Recognized modern
error" is an era signal for one code out of five.
