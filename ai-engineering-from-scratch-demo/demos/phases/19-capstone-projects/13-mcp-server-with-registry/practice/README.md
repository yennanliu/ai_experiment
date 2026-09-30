<!-- generated:start -->
# 19-capstone-projects / 13-mcp-server-with-registry

Solutions to all 7 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/13-mcp-server-with-registry/) · upstream spec
`phases/19-capstone-projects/13-mcp-server-with-registry/docs/en.md`

```bash
uv run demo practice run 13-mcp-server-with-registry --ex 1
uv run demo explain 13-mcp-server-with-registry --ex 1
uv run pytest demos/phases/19-capstone-projects/13-mcp-server-with-registry
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Change the published remote URL while leaving the live server unchanged. Make the registry va… | code | T0 | `ex01_the_lessons_registry_checks_pass_3_of_3_drifted_remote_urls_because_nothing_probes_the_url.py` |
| 2 | Send `tools/list` twice with identical inputs and prove byte-stable tool order. Then expire `… | code | T0 | `ex02_tools_list_is_byte_stable_but_its_private_cache_hands_a_1_scope_caller_2_tools.py` |
| 3 | Send a valid body with a different `MCP-Protocol-Version` header. Return `-32020` and do not… | code | T0 | `ex03_the_lesson_has_no_header_layer_so_a_mismatched_version_header_runs_policy_and_the_tool.py` |
| 4 | Mint a token for the read-only server and present it to the state-changing server. Prove audi… | code | T0 | `ex04_the_audience_check_stops_a_read_only_token_but_tools_list_shows_it_the_destructive_tool.py` |
| 5 | Bind an approval to one normalized argument digest. Change one field and prove the approval c… | code | T0 | `ex05_a_changed_field_cannot_replay_the_approval_but_the_unchanged_action_replays_5_of_5_times.py` |
| 6 | Route consecutive calls to alternating replicas. Replace hidden process memory with an explic… | code | T0 | `ex06_with_cursor_state_in_process_memory_alternating_replicas_answer_0_of_75_page_reads.py` |
| 7 | Break a request-scoped SSE connection and retry with a new JSON-RPC request ID. Verify that n… | code | T0 | `ex07_a_new_id_retry_never_resumes_and_after_a_late_break_it_creates_the_jira_issue_twice.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`: a stdlib model with two
`MCPServer` objects (read-only and destructive), a `Registry`, `Token`,
`ApprovalRecord`, `policy_decide`, `dispatch` and an audit list. It opens no
socket and has no HTTP layer, so exercises 3 and 7 add a thin
Streamable HTTP front around `dispatch`, and exercise 6 adds two tools through
the lesson's own `register`. Spec facts were read on 2026-09-29 from
modelcontextprotocol.io/specification/2026-07-28 (`basic/transports/streamable-http`,
`basic/patterns/cancellation`, `server/tools`).

### 1 — the lesson's registry checks pass 3 of 3 drifted remote URLs, because nothing probes the URL

**A drift report that probes the published URL names the field, the published
value and the live value for all 3 edits.** Two of the edits (a path typo and
an `http://` downgrade) reach no server. The third, the destructive server's
URL, answers as `com.example/internal-destructive`, so the report also names
the `serverInfo.name` mismatch.

| edit to `remotes[0].url` | lesson validators | drift report | lesson alignment, probed by URL |
|---|---:|---|---|
| `/readonly-v2` | 0 issues | url differs; nothing answers | nothing to compare |
| `http://…/readonly` | 0 issues | url differs; nothing answers | nothing to compare |
| `…/destructive` | 0 issues | url differs; `serverInfo.name` differs | name mismatch |

**The lesson's registry validation cannot see URL drift.** `Registry.register`
calls `discover()` on the in-process object and never uses the URL.
`validate_runtime_alignment` compares only name and version. The doc's Build
It step 7 asks for drift on "the published remote, identity, version".

### 2 — tools/list is byte-stable, but its "private" cache hands a 1-scope caller 2 tools

**Two calls return identical 744-byte results, and so does a server whose tools
were registered in reverse order.** The order is `postgres.readonly`,
`s3.list`. A client cache with an injected clock serves from memory at 0 ms and
59,999 ms, then refetches at 60,000 ms (the result's `ttlMs`). The refreshed
bytes are the same. That is 2 server calls for 4 reads.

**The "private" scope is not per-caller.** `tools_list` takes no token, yet
it returns `cacheScope: "private"`. A token holding only `s3:list` lists both
tools, and `dispatch` then denies one of them. The spec allows the list to vary
by the authorization on the request. The lesson's two models also disagree:
Python uses 60,000 ms and `private`, TypeScript uses 300,000 ms and `public`.

### 3 — the lesson has no header layer, so a mismatched version header runs policy and the tool

**The header front returns HTTP 400 with `-32020`, and policy and the tool run
0 times.**

| request | status | code | policy runs | tool runs |
|---|---:|---|---:|---:|
| header `2025-11-25`, body `2026-07-28` | 400 | -32020 | 0 | 0 |
| header `2099-01-01` (unsupported), body `2026-07-28` | 400 | -32020 | 0 | 0 |
| `Mcp-Name` differs from `params.name` | 400 | -32020 | 0 | 0 |
| headers match (names as written or lower case) | 200 | complete | 1 | 1 |

The unsupported-version header still gets `-32020`, not `-32022`, because the
mismatch is checked first. **Without the front, the lesson runs the
mismatched request.** The same body handed straight to `dispatch` completes,
with policy and tool each run once. `-32020` occurs 0 times in `main.py` and
the 5 TypeScript source files.

### 4 — the audience check stops a read-only token, but tools/list shows it the destructive tool

**A token minted for the read-only server is refused by the destructive
server with "token audience does not match this server", and the `jira.create`
handler runs 0 times.** That holds as minted, with `jira:write` added, and with
`jira:write` plus a valid approval, because audience is checked before scope and
approval. Each denial is audited. The reverse direction is refused in the same
way. The correct token, with the approval, runs the handler once.

**`tools/list` has no authorization.** The refused token still lists
`jira.create` with `destructiveHint: true`. Wrong audience and unknown tool
both come back as the generic `-32000`. The spec treats an unknown tool as a
protocol error and uses `-32602` in its example.

### 5 — a changed field cannot replay the approval, but the unchanged action replays 5 of 5 times

**With `title` changed from `"new bug"` to `"new bug!"`, the call is denied
with "approval arguments do not match requested action" and the handler runs
0 times.** A record bound to another target, or presented by another actor, is
denied with its own reason. The digest does not depend on key order. It does
depend on Unicode form (NFC vs NFD) and on number spelling (`1` vs `1.0`),
which fails closed.

**The approval is not one-time.** The unchanged action is allowed 5 of 5
times, the handler runs 5 times, and every call returns the same hard-coded
`PROJ-99`. The doc lists a "one-time or repeat-use policy"; the model has only
repeat use. **The approval is also the only gate on out-of-schema arguments.**
`dispatch` never checks `inputSchema`. An approval minted for
`{"title": "x", "assignee": "root"}` is honoured, and the handler receives
`assignee` even though the schema sets `additionalProperties: false`.

### 6 — with cursor state in process memory, alternating replicas answer 0 of 75 page reads

**With an explicit shared handle (an opaque UUID4 key in one store both
replicas read), all 75 page reads are correct.** Each of the 25 workflows
opens a cursor and reads 3 pages, which makes 100 calls over two round-robin
replicas.

| cursor state | routing | correct | error | silently wrong |
|---|---|---:|---:|---:|
| shared handle | alternating | 75 | 0 | 0 |
| each replica's own memory | alternating | 0 | 50 | 25 |
| each replica's own memory | affinity | 75 | 0 | 0 |

With hidden memory, the 25 wrong answers are page 1 returned where page 2 was
due, and they carry no error. The lesson's own tools are stateless: 100
alternating calls give results byte-identical to one replica, and both replicas
encode `tools/list` identically. **The TypeScript model does keep this kind
of state.** `incidents_ack` sets `inc.acked = true` on a process-local map, so
behind two replicas one replica's ack is invisible to the other.

### 7 — a new-ID retry never resumes, and after a late break it creates the Jira issue twice

**The broken id-7 stream is logged as cancelled, and the id-8 retry is a fresh
request.** Its stream starts again at "started" and ends with a response whose
id is 8. No stream carries an SSE `id:` line, so there is nothing to resume
from. The retry sends no `Last-Event-ID`. A retry that does send
`Last-Event-ID: 1`, as a 2025-era client would, gets the same stream and runs
the tool again. The spec says resumable streams are not supported and the header
is ignored.

| tool | break after | handler runs before retry | after retry |
|---|---|---:|---:|
| `jira.create` | event 1 (before tool) | 0 | 1 |
| `jira.create` | event 2 (tool done, reply lost) | 1 | 2 |

**A late break makes the retry run the mutation twice.** The cancellation
arrives after the handler has run, and the lesson has no idempotency key: the
reference code mentions "idempot" 0 times. Both calls return `PROJ-99`, so the
duplicate does not show in the reply. The read-only tool is safe to retry,
because its two replies are identical.
