<!-- generated:start -->
# 13-tools-and-protocols / 17-mcp-gateways-and-registries

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/17-mcp-gateways-and-registries/) · upstream spec
`phases/13-tools-and-protocols/17-mcp-gateways-and-registries/docs/en.md`

```bash
uv run demo practice run 17-mcp-gateways-and-registries --ex 1
uv run demo explain 17-mcp-gateways-and-registries --ex 1
uv run pytest demos/phases/13-tools-and-protocols/17-mcp-gateways-and-registries
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add trace context to the outer and forwarded request metadata and record the correlation in t… | code | T0 | `ex01_the_audit_row_and_the_forwarded_id_are_made_in_different_functions.py` |
| 2 | Add a Tasks-capable backend and route `tasks/get` by task id in `Mcp-Name`. | code | T0 | `ex02_a_task_id_is_opaque_so_routing_it_costs_the_state_tools_do_not.py` |
| 3 | Change one backend descriptor and prove both discovery and direct call are blocked. | code | T0 | `ex03_discovery_omits_it_silently_and_the_call_says_why.py` |
| 4 | Add a principal-specific server capability and explain why discovery must remain privately ca… | code | T0 | `ex04_the_gateway_already_downgrades_what_the_backend_calls_public.py` |
| 5 | Write a legacy adapter interface without adding any legacy state to the modern `Gateway` class. | code | T0 | `ex05_the_interface_is_already_there_it_is_just_not_written_down.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib, and all five ship code. The scaffold classified
exercise 4 as prose on the word "explain", but it has a code deliverable — a
principal-specific capability — so it is filed as code with the explanation
read off the measurement.

A gateway's whole job is to sit between two parties and account for what
passes, and the recurring gap here is **accounting that stops at the
boundary**: an audit row that cannot name a request, a trace that would miss
every refusal, a listing that omits without saying, an interface that is
wider than the object implementing it.

### 1 — the audit row and the forwarded id are made in different functions

**ANSWER: one trace across both hops, a span each, both in the audit row.**
The row goes from **3** keys to **6**.

**FINDING: two identical calls are indistinguishable in the shipped log.** The
row is `{principal, tool, decision}` — no request id, no timestamp, no trace.

**FINDING: `forwarded_request_ids` is generated and never joined.** `_forward`
appends it; `handle` writes the audit row and never reads it. The correlation
does not exist to be recorded — it has to be created.

**FINDING: three of the four decisions never reach a backend.** `deny`,
`rate_limit` and `pin_mismatch` are appended before `_forward` runs, so a
trace that spans forwards is blind to every refusal.

### 2 — a task id is opaque, so routing it costs the state tools do not

**ANSWER: `tasks/get` routes by the id in `Mcp-Name`.** `validate_wire`
already mirrors `params.taskId` for task methods where it mirrors
`params.name` for tool ones.

**FINDING: the envelope was specified before the router.** `NAMED_METHODS`
lists **6** methods including all three `tasks/*`, and the shipped `handle`
answers `tasks/get` with **-32601**.

**FINDING: routing tools needs no state and routing tasks does.**

| routing | mechanism | state |
|---|---|---|
| `notes.search` | `split(".", 1)` | none |
| a task id | `{task_id: backend}` | grows per task, must be durable |

**FINDING: RBAC has no row a task id could match.** Authorization for tasks
has to be ownership — a second table again.

### 3 — discovery omits it silently and the call says why

**ANSWER: the tool leaves the listing (4 → 3) and the call answers 409
`-32010`.**

**FINDING: the two blocks report differently, and only one is an answer.** A
client working from a cached listing gets the useful message; one that
refreshes gets the silent one.

**FINDING: the pin covers the whole descriptor, so prose is load-bearing.**
Name and schema identical, digest moved.

**FINDING: deny and pin_mismatch are separated in the log and nowhere else.**
From the client's side both simply fail to appear.

### 4 — the gateway already downgrades what the backend calls public

**WHY DISCOVERY MUST STAY PRIVATE:** with a principal-specific capability the
response varies by caller while the requests are byte-identical — a `public`
scope claims the body depends on nothing about the caller, and the first
response cached would be served to the second.

**ANSWER: the capability is added and the scope is already right.** The change
makes the existing `private` *necessary* rather than changing it.

**FINDING: the gateway already downgrades its backends' hint.**
`BackendServer` says `public`/60s; the gateway says `private`/30s. Forwarding
the backend's hint verbatim would have been wrong.

**FINDING: the shorter ttl is on the thing that moves faster.** 10s on
`tools/list` against 30s on discovery — one word, two risks.

### 5 — the interface is already there; it is just not written down

**ANSWER: a `Backend` protocol of `handle` and `tools`.** `BackendServer` and
`LegacyAdapter` both satisfy it; the gateway routes to either unchanged.

**FINDING: the gateway's **5** fields are unchanged, and the session is the
adapter's.** The constraint holds because the adapter is a backend, not a
branch.

**FINDING: the real interface is wider than the protocol.** `tools/list`
answers **400** until `REGISTRY_SERVER_JSON` and `VERIFIED_ADMISSION_STATE`
each carry a row under the same key. The structural interface is a method and
an attribute; the working one adds two module dictionaries nothing declares.

**FINDING: the handshake is invisible from the gateway's side.** 2 legacy
messages on the first call and 1 after, against one forwarded id either way.
