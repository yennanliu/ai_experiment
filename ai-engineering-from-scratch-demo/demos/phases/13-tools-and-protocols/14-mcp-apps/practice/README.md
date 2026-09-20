<!-- generated:start -->
# 13-tools-and-protocols / 14-mcp-apps

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/14-mcp-apps/) · upstream spec
`phases/13-tools-and-protocols/14-mcp-apps/docs/en.md`

```bash
uv run demo practice run 14-mcp-apps --ex 1
uv run demo explain 14-mcp-apps --ex 1
uv run pytest demos/phases/13-tools-and-protocols/14-mcp-apps
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Change the client capability to an empty extension map. Confirm `tools/list` keeps the tool b… | code | T0 | `ex01_the_tool_degrades_where_the_resource_refuses.py` |
| 2 | Send `Mcp-Name: ui://notes/other.html` with a body that reads the timeline. Confirm error `-3… | code | T0 | `ex02_the_header_check_runs_first_so_it_masks_every_other_fault.py` |
| 3 | Change the resource to `cacheScope: private`. Describe the user-specific condition that justi… | code | T0 | `ex03_the_template_is_not_a_template_it_is_the_data.py` |
| 4 | Move the script to `https://static.example.com/app.js`. Add that origin to `resourceDomains`… | code | T0 | `ex04_the_csp_can_name_an_origin_and_cannot_pin_a_build.py` |
| 5 | Add an `notes_open` tool and route the button click through the host. Keep user approval in t… | code | T0 | `ex05_the_app_can_ask_and_only_the_host_can_call.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib, and all five ship code. The scaffold classified
exercises 3 and 4 as prose on the words "Describe" and "explain", but each has
a code deliverable — a changed cache scope, a rehosted script and an opened
CSP list — so both are filed as code with the argument made from the
measurement.

An MCP App puts a server's markup inside a host's frame, and the five
exercises circle one question: **which side of that boundary holds each
guarantee?** The capability is the client's, the CSP is the server's, the
approval is the host's, and three of the five find a case where the boundary
is drawn somewhere other than where it reads.

### 1 — the tool degrades where the resource refuses

**ANSWER: the tool survives and its `_meta` does not.** Same **1** tool, same
name and schema; only `_meta.ui.resourceUri` disappears.

**FINDING: `resources/read` refuses where `tools/list` degrades.** The same
empty map gives **-32021** there. A client that lost the capability between
the two calls gets a tool it can invoke and cannot render.

**FINDING: the capability is read per request, and the result is cached as
`public`.** Both listings carry `ttlMs: 60000, cacheScope: "public"` and they
differ — a cache keyed on the method alone would hand a UI-bound listing to a
client that never asked for one.

**FINDING: the negotiation is "a dict or nothing".** Of `{}`, `{extensions:
{}}`, `{ui: None}` and `{ui: {}}`, exactly **1** enables the binding. A client
sending `true` to mean yes is read as absent, silently.

### 2 — the header check runs first, so it masks every other fault

**ANSWER: `-32020 Mcp-Name header does not match body`, HTTP 400.**

**FINDING: the same mismatch masks three other faults.**

| also wrong | with mismatch | alone |
|---|---|---|
| capability | `-32020` | `-32021` |
| version | `-32020` | `-32022` |
| unknown URI | `-32020` | `-32602` |

Two faults cost two round trips.

**FINDING: `params.name or params.uri` is not `if present`.** A `tools/call`
naming the empty string sends `Mcp-Name: ""` while the server computes `None`.
An empty name is unreachable rather than rejected.

**FINDING: the rule covers exactly three methods.** `tools/list` goes out with
no `Mcp-Name` to mismatch — the header addresses one object, which the
validation implements and never states.

### 3 — the template is not a template, it is the data

**THE USER-SPECIFIC CONDITION:** `timeline_html(NOTES)` interpolates every
title, so two users' reads differ — **3** of one user's titles present, **0**
of the other's. A `public` cache is keyed on a URI that is the same for
everybody. The body varies with the caller and the key does not.

**ANSWER: `cacheScope: "private"`, `ttlMs` unchanged, nothing else moved.**

**FINDING: `resources/list` should stay public.** Its entry carries `uri`,
`name`, `description`, `mimeType` and **0** titles. Scope is a property of
what the bytes contain, not of the server.

**FINDING: the data already travels twice.** `tools/call` returns the same
notes as `structuredContent`, so an empty template filled from the tool result
would be genuinely user-independent — the scope follows the design. (That tool
result also hands back the store's own list object rather than a copy.)

### 4 — the CSP can name an origin and cannot pin a build

**ANSWER: the inline script moves out and `resourceDomains` goes 0 → 1.** The
other three lists stay empty.

**THE RISK: the server vouches for code it does not build and cannot version.**
The CSP grammar has **4** keys and none is an integrity hash, and the URL
carries no version. Whatever the origin serves next runs inside the frame on
the next read — and a `public` resource may keep serving the same HTML while
the script behind it changes.

**FINDING: `connectDomains` empty is a real mitigation and not a complete
one.** No socket out, but the script still executes in a document carrying
**3** of the user's note titles. Confidentiality now rests on `frameDomains`
and `baseUriDomains` staying empty too.

**FINDING: the origin lives in two places nothing cross-checks.** Pointing the
tag at a domain absent from `resourceDomains` still answers HTTP 200. The
failure is silent at the protocol layer and visible only inside the frame.

### 5 — the app can ask, and only the host can call

**ANSWER: intent → approval → call.** Three clicks, **3** prompts, **3**
`notes_open` calls, and **0** direct app-to-server requests.

**FINDING: the approval is the only thing between the frame and the tool.**
Auto-approving gives the same **3** calls and **0** prompts. What makes the
design safe is not that the message goes through the host — it is that the
host stops.

**FINDING: the intent is untrusted input, because the frame runs foreign
code.** **2** of **5** intents (`note-999`, a path traversal) never reach a
human. Exercise 4 put third-party script in this frame; this is the check that
keeps a click from becoming an arbitrary call.

**FINDING: approving is not authorizing.** A host that approves anything still
gets **-32602** from the tool. Two independent checks, neither redundant: the
host cannot know the server's store, and the server cannot know whether a
human clicked.
