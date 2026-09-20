<!-- generated:start -->
# 13-tools-and-protocols / 23-capstone-tool-ecosystem

Solutions to all 7 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/23-capstone-tool-ecosystem/) · upstream spec
`phases/13-tools-and-protocols/23-capstone-tool-ecosystem/docs/en.md`

```bash
uv run demo practice run 23-capstone-tool-ecosystem --ex 1
uv run demo explain 23-capstone-tool-ecosystem --ex 1
uv run pytest demos/phases/13-tools-and-protocols/23-capstone-tool-ecosystem
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Separate facts proven by the output from production claims that still nee… | code | T0 | `ex01_the_delegation_row_promises_a_sleep_the_code_does_not_have.py` |
| 2 | Add a second static backend and define the collision rule for two tools with the same name. T… | code | T0 | `ex02_the_pin_manifest_already_has_a_namespace_the_tool_list_does_not.py` |
| 3 | Replace the writer stub with an A2A test server. Record the Agent Card, message request, time… | code | T0 | `ex03_the_timeout_destroys_a_result_the_server_had_already_produced.py` |
| 4 | Add a task store that survives a process restart. Prove a client can resume with `tasks/get`,… | code | T0 | `ex04_the_shipped_client_ignores_the_interval_it_was_handed.py` |
| 5 | Build a minimal MCP App and verify `app.callServerTool` in a browser with a restrictive CSP a… | code | T0 | `ex05_the_shipped_app_is_blocked_by_its_own_two_fields.py` |
| 6 | Export the simulated spans through an OTel SDK to a local collector. Assert receipt, trace id… | code | T0 | `ex06_the_denial_never_reaches_the_trace_because_the_span_starts_after_it.py` |
| 7 | Write `AGENTS.md` for repository-wide maintenance rules and a separate skill bundle for the r… | code | T0 | `ex07_authority_is_a_two_table_function_and_neither_table_is_a_document.py` |
<!-- generated:end -->

## Answers

All seven are T0 and stdlib, and all seven ship code.

The capstone's own table splits nine layers into "simulated" and
"production", and says a green local run validates only the first column.
Four of these solutions cross the boundary anyway, because `http.server`,
`urllib` and `subprocess` are stdlib: exercises 2, 3, 5 and 6 run real
servers on loopback and exercise 4 runs a real second interpreter. What that
buys is not fidelity but **failure modes** — a timeout, a restart, a rejected
policy, an abandoned request — none of which the in-process version can have.

The recurring finding is that the simulation's fields are already right and
already unused. `pollIntervalMs` is emitted and never read. `PINNED` is
namespaced and nothing else is. The `ui` block's CSP and permissions are
both present and together they forbid the App from working. A field that
nothing consumes cannot be wrong, which is why promoting a layer finds bugs
that the simulation could not.

### 1 — the delegation row promises a sleep the code does not have

**ANSWER: 6 facts, all re-derived from a fresh run; 9 production layers, all
unevidenced by construction.** The module imports `hashlib`, `json`, `time`,
`uuid`, `copy` and `datetime` — **0** of the 10 modules that could open a
socket, spawn a process or reach a database. No reading of the output could
move a row from the claims column, which is what makes the partition
checkable rather than a judgement.

**FINDING: the delegation row promises a sleep the code does not have.**
`time.sleep` appears **0** times, and the `a2a.SendMessage` span lasts under
a microsecond. The stub does not simulate latency — the one property that
would make a timeout test mean anything.

**FINDING: the audit log and the span list are process-global.** Two runs
leave **4** audit rows and **11** spans under **2** trace ids. "Every span in
one run shares one trace id" is true of the **7** you slice out and is not
what the printed list shows.

**FINDING: opacity here is the absence of a field, not a boundary.** The
delegation span carries peer and skill and no content, but the writer is this
process and the HTML it "returns" is built inline by the caller.

### 2 — the pin manifest already has a namespace the tool list does not

**ANSWER: qualify every tool as `server::name`, and refuse a bare name two
servers answer to.** Two real `tools/list` POSTs return **4** tools under
**3** distinct bare names; qualified, all four are distinct.

**FINDING: `PINNED` is keyed `research::name` and nothing else is.**
`TOOLS`, `REQUIRED_SCOPE` and `gateway_call` all use the bare name. The
namespace the collision rule needs is already in the shipped code, in one of
the four places that would have to agree.

**FINDING: the pin authenticates the description, not the server.** The
archive backend's `generate_report` ships a byte-identical description and
`pin_ok` returns True — a tool passing a manifest it was never in.

**FINDING: the scope table lends its answer to the newcomer.** The archive
tool inherits `research:write`, and `gateway_call`'s `next(...)` returns the
first match without noticing a second exists. Routing and authorization both
decided by list order.

### 3 — the timeout destroys a result the server had already produced

**ANSWER: a real Agent Card, a real `message/send`, a real timeout, and an
artifact the caller did not build.** **3** requests leave the client, **3**
complete on the server, **2** answers come back.

**FINDING: the timeout destroys a result the server had already produced.**
Nothing in the response shape lets the client ask again — which is exactly
what a task id would have given it, and what a synchronous function return
can never need.

**FINDING: the capstone has no failure path for delegation at all.** **0**
`except` handlers and **0** timeouts in the whole module. Adding the wire
adds the first way the step can fail.

**FINDING: the skill name is asserted in a span attribute and validated
nowhere.** The card declares `summarize_papers` and the span records it; the
two agreeing is a coincidence no code checks.

### 4 — the shipped client ignores the interval it was handed

**ANSWER: the task survives a restart, resumes by id, is polled at its
declared interval, and carries its own final result.** Gaps of
`[65, 65, 66]`ms against a declared **60**, and **0** calls to
`tasks/result`.

**FINDING: the shipped client ignores the interval it was handed.**
`pollIntervalMs` is **1000** and `orchestrator` calls `tasks_get` on the next
line; a whole run takes under **0.1**ms. The only client that exists is the
one that violates it.

**FINDING: the volatile store does not survive its own process.** A real
second interpreter answers `-32602 Unknown taskId` for a task the parent
minted. Durability here is a missing file, not a missing feature.

**FINDING: `tasks/result` is absent and unnecessary in the same breath.** The
completed task already carries `result`; a second fetch would only re-deliver
a field the client holds.

### 5 — the shipped App is blocked by its own two fields

**ANSWER: a CSP that names the script by hash and permissions that name the
tool by name.** The header arrives byte-exact over loopback and the bridge
allows **1** of **2** attempted calls. The remaining claim is enforcement —
a browser choosing to honour the header — and nothing here can assert it.

**FINDING: the shipped App is blocked by its own two fields.**
`default-src 'self'` with an inline `<script>` and no hash means a conforming
renderer refuses to run it; `permissions: []` means `app.callServerTool` has
nothing it may call. Both fields look right; together they describe an App
that can neither run nor call.

**FINDING: hashing the script binds its bytes.** One added space flips the
verdict. The hash is the tightest of the three options and it moves the
failure to deploy time — the trade a nonce is usually bought to avoid.

**FINDING: the `ui://` reference names nothing resolvable.** The HTML is
delivered in a sibling key and the module defines **0** ways to read a
resource, so nothing breaks when the URI points nowhere.

### 6 — the denial never reaches the trace, because the span starts after it

**ANSWER: a real collector receives the run, and every assertion reads what
came out of the socket.** **12** spans in **1** POST, 32-hex trace ids,
16-hex span ids, **2** trees, **0** orphans, depth **4**, **1** span at
`status.code` 2.

**FINDING: the denial never reaches the trace.** Bob's refusal adds **1**
audit row and **0** spans, because `gateway_call` returns before `span(...)`
is called. There is no error status to export until the span moves above the
check — the change that makes the trace and the audit log agree.

**FINDING: an exception orphans a span the exporter cannot tell is
unfinished.** `finish()` runs only on the success path, so a raising tool
leaves `end == 0`; exported literally the span ends before it began.

**FINDING: an exchange is two `llm.chat` spans and neither is costable.**
**0** of **4** carry both token counts.

### 7 — authority is a two-table function, and neither table is a document

**ANSWER: both artifacts written, and the gateway's answers are unchanged.**
`gateway_call` reads **4** module globals directly plus `PINNED` through
`pin_ok`, and **0** of the five is a document, a path or a filesystem call.
A Markdown file cannot be in `USERS` or `REQUIRED_SCOPE`, so it cannot grant
anything.

**FINDING: the two documents differ by version, not by content.** The bundle
carries `name`, `description` and `version`; `AGENTS.md` carries **0**
frontmatter fields. The procedure can be pinned, diffed and routed to; the
repository rules can only be read.

**FINDING: a directory bundle is invisible to the course installer.** **0**
of the bundle's **4** files match the flat `skill-*.md` glob — which is why
this lesson's own output artifact is flat.

**FINDING: the minimal parser reads top-level keys.** Flat frontmatter yields
**6** keys; the same fields under `metadata:` yield **2**, one of them
`metadata` itself, and `--tag capstone` then matches nothing.
