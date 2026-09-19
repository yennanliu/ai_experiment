<!-- generated:start -->
# 13-tools-and-protocols / 08-building-an-mcp-client

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/08-building-an-mcp-client/) · upstream spec
`phases/13-tools-and-protocols/08-building-an-mcp-client/docs/en.md`

```bash
uv run demo practice run 08-building-an-mcp-client --ex 1
uv run demo explain 08-building-an-mcp-client --ex 1
uv run pytest demos/phases/13-tools-and-protocols/08-building-an-mcp-client
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Make a fake server return `-32022` with no mutually supported version. Confirm the client fai… | code | T0 | `ex01_one_request_is_sent_and_the_retry_path_is_the_one_not_taken.py` |
| 2 | Allowlist a fake legacy server, make its bounded `initialize` probe time out, and prove the p… | code | T0 | `ex02_failing_closed_leaves_the_peer_in_the_state_it_started_in.py` |
| 3 | Add `cacheScope: "private"` tool lists for two authorization contexts. Confirm the client nev… | code | T0 | `ex03_the_client_cannot_leak_a_cache_because_it_has_none.py` |
| 4 | Change the collision policy to rejection and make startup fail with both peer names in the er… | code | T0 | `ex04_the_policy_called_reject_drops_the_tool_and_says_nothing.py` |
| 5 | Add a finite `subscriptions/listen` simulator. On stream loss, re-listen with a new request i… | code | T0 | `ex05_a_new_id_is_a_new_stream_and_the_gap_is_resynced_not_replayed.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib, and all five ship code.

Lesson 13.07 built the server. This one builds the client that has to talk to
servers it did not write, and every exercise lands on the same seam: **the
lesson's prose describes a protocol, and `code/main.py` implements the part of
it that fits in a request/response pair.** Four of the five exercises are
answerable only by building the thing the prose assumes and the code omits.

### 1 — one request is sent, and the retry path is the one not taken

**ANSWER: it raises after one request, and `initialize` is not among them.**
The transport records **1** message, `server/discover`; the peer stays
`era="unknown"`, `available=False`, `protocol_version=None`.

**FINDING: `-32022` is a negotiation signal, not a failure.** Probing at
`2099-01-01` against a server supporting `2026-07-28`, the client reads
`data.supported`, picks the mutual version and re-sends `server/discover` — **2**
requests, both modern, and the peer activates. Only the empty intersection is
fatal, which is why the exercise specifies it.

**FINDING: legacy is never reached, because `-32022` proves the peer is
modern.** `_connect_peer` falls back for an unrecognised code or a dead
transport, but the three recognised modern errors mean the server spoke modern
well enough to reject the version. Even with `allow_legacy=True` the transport
sees **0** `initialize` messages.

**FINDING: and the allowlist does not change the outcome.** Both configurations
raise the same text. The allowlist governs the fallback path, and this peer
never enters it.

### 2 — failing closed leaves the peer in the state it started in

**ANSWER: `bounded legacy probe failed closed`, and every field is untouched** —
`era="unknown"`, `available=False`, `protocol_version=None`, `capabilities={}`,
`server_info={}`, `tools=[]`, exactly what `add_server` created. The peer is not
partially connected; it is not connected.

**FINDING: the bound reaches the transport, and it is the legacy one.**
Configured to 1,000 and 250, the transport sees `[1000, 250]` across
`server/discover` and `initialize`.

| setting | bounds |
|---|---|
| `discovery_timeout_ms` | an unreachable server |
| `legacy_probe_timeout_ms` | one that answers slowly enough to hold a startup open |

**FINDING: a timeout and a refusal are the same outcome; a bad reply is not.**
`TimeoutError` and `ConnectionError` both give `failed closed`; a wrong protocol
revision gives `unsupported legacy protocol revision`. The client separates
"could not ask" from "asked and did not like the answer", and leaves the peer
unavailable either way.

**FINDING: and the allowlist is checked before the probe, not after.** Without
it the same dead transport raises `legacy compatibility is not allowlisted`
after **1** message. It is a gate on whether the second request is sent, not a
filter on its result.

### 3 — the client cannot leak a cache, because it has none

**ANSWER: the client never shares a cached result because it never keeps one.**
`MultiServerClient` has **0** attributes matching a cache and `discover_tools`
re-sends `tools/list` every call — **2** messages for two rounds against one
peer. The property the exercise asks about holds vacuously, and `cacheScope` is
read by nobody.

So the exercise is only answerable by building the cache it assumes:

| key | entries | reader is served |
|---|---:|---|
| `peer` | 1 | the admin's 3-tool list — the leak |
| `(peer, principal)` | 2 | its own 1-tool list |

**FINDING: `private` does not forbid caching — it names the key.**

**FINDING: and the client has no principal to key on.** `CLIENT_INFO` is a name
and a version; `client_capabilities` is a feature list. Neither varies per
authorization context. The key `private` requires would have to come from the
transport, which is where the credential lives. That is why the cache was never
written.

### 4 — the policy called reject drops the tool and says nothing

**ANSWER: `reject` silently keeps the first peer's tool and discards the
second.** Two peers exporting `notes_list` merge to **1** entry,
`{'notes_list': 'alpha'}`. No exception, no diagnostic, and beta's tool is
unreachable.

**FINDING: which peer wins is alphabetical, not deliberate.** `merge` iterates
`sorted(self.peers)`, so renaming alpha to zulu hands the bare name to beta. The
survivor is chosen by a client-side configuration string the server never sees.

**FINDING: the strict policy names both peers from values the shipped code
already holds.** `tool name collision: notes_list exported by alpha and beta`,
assembled from the incumbent's `peer_name` — which the registry stores — and the
current peer. Both are in scope at the line that currently says `continue`.

**FINDING: so the three policies differ in what they lose.**

| policy | loses | noticed? |
|---|---|---|
| prefix-on-collision | name stability | at the model's next call |
| reject | a tool, silently | no |
| raise | the whole startup | yes |

Only the last cannot go unnoticed in production, which is the argument for it.

### 5 — a new id is a new stream, and the gap is resynced, not replayed

**ANSWER: three listens, ids `[2, 4, 6]`, three refetches.** Two scheduled
stream losses; the registry goes `['notes_list']` → `['notes_delete',
'notes_list']` → `['notes_archive', 'notes_delete', 'notes_list']`, matching the
server at every step, with **6** events delivered.

**FINDING: the client has nowhere to put a stream, and the type signature says
so.** `MultiServerClient` has **0** members matching listen, subscribe, stream
or notification; `_send` is annotated `dict[str, Any] | None` — one message per
request, so a multi-event stream cannot pass through `_request` at all. The
lesson's `ModernFakeServer` answers `subscriptions/listen` with **-32601**.

**FINDING: the new id is load-bearing, because the id is the only correlation
key.** A replayed stream-1 event is accepted **0 of 1** times against the live
stream, and **1 of 1** had the re-listen reused the old id. Reusing the id does
not resume the stream — it makes the dead stream's traffic indistinguishable
from the live one's.

**FINDING: and the refetch recovers the state, not the events.** The server
dropped **2** `tools/list_changed` with no stream open to carry them and the
client received **0**; its tool list is right anyway, because `tools/list`
reports the current state and the notification reported a transition. That is
what "modern streams do not resume with `Last-Event-ID`" costs — and why a
resync, not a replay, is the documented recovery.
