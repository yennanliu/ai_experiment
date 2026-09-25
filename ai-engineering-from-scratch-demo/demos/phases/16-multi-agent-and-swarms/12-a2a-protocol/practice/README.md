<!-- generated:start -->
# 16-multi-agent-and-swarms / 12-a2a-protocol

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/12-a2a-protocol/) · upstream spec
`phases/16-multi-agent-and-swarms/12-a2a-protocol/docs/en.md`

```bash
uv run demo practice run 12-a2a-protocol --ex 1
uv run demo explain 12-a2a-protocol --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/12-a2a-protocol
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Confirm the client discovers the server and receives the correct artifact. | code | T0 | `ex01_the_submitted_state_is_already_gone_when_the_201_says_it.py` |
| 2 | Add a second skill to the server (e.g., "summarize"). Update the Agent Card. Write a client t… | code | T0 | `ex02_adding_a_skill_means_replacing_two_module_globals.py` |
| 3 | Implement an SSE streaming endpoint: `/tasks/{id}/events` that emits state changes. What does… | code | T0 | `ex03_one_open_stream_blocks_every_other_request.py` |
| 4 | Read the A2A spec (https://a2a-protocol.org/latest/specification/). Identify three things the… | code | T0 | `ex04_five_of_the_six_core_operations_are_404.py` |
| 5 | Compare A2A (Agent Card discovery) to MCP (server-side capability listing via `listTools`). W… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — the submitted state is already gone when the 201 says it

**The client discovers the server and gets an artifact, but the artifact is
not correct.** The lesson's own `run_client`, pointed at the lesson's own
handler, reaches `completed` and receives:

```python
{'issues': ['no return statement', 'no function definition'], 'lines': 3}
```

The two issues are right for `x = 1\nprint(x)\n`. The line count is wrong:
the code has **2** lines, and `code.count("\n") + 1` counts the empty string
after the trailing newline as a third.

Three more things turn up on the way.

**The 201's `submitted` is a literal, and no client can see that state.**
`do_POST` answers `{"task_id": tid, "state": "submitted"}` without reading
the store. `create()` starts the worker before it returns, and the worker's
first statement sets `working`. In 50 of 50 creates the store already read
`working` when `create()` returned.

**Discovery supplies 1 of the client's 3 URLs, and that one is a literal
too.** The card URL and the poll URL are hardcoded in `run_client`. Only the
submit URL comes from `card["endpoints"]["tasks"]`, and the card itself
hardcodes `http://localhost:8765/tasks`. On any other port the discovered
URL points back at 8765: all 5 of the client's requests had to be redirected
to run this check.

**An unknown skill is accepted, then failed.** `{"skill": "summarize"}` gets
a 201 `submitted` and turns `failed` after the worker's 0.2s sleep. Nothing
compares the request against the card's skill list.

### 2 — adding a skill means replacing two module globals

The solution uses three pieces: a store that dispatches through a registry,
a card whose skills have the shape of the spec's AgentSkill
(`id`, `name`, `description`, `tags`), and a client that picks a skill by
matching the task type against `tags`. `review` routes to `review-python`,
`summarize` routes to `summarize`, and the client never names either skill.

Measured against the lesson's server, that change is harder than it should
be:

- **There is no seam for a skill.** The only dispatch is
  `if t["skill"] == "review-python"` inside `TaskStore._run`, so a second
  skill means overriding `_run` wholesale.
- **The handler owns no state.** `A2AHandler` reads the module globals
  `STORE` and `AGENT_CARD` directly, so serving a new store and card means
  rebinding both globals in the module. Two agents cannot share a process.
- **Validation belongs at submit.** Checking the body's skill against the
  card refuses an unknown skill with **400 and 0 task records**. The shipped
  handler answers 201, creates a record, and fails it later.
- **The shipped card cannot drive a routing client.** `["review-python"]` is
  a list of strings with no description, no tags and no input shape. Only a
  client that already knows the name, and that the payload key is `code`,
  can call it. That is exactly what `run_client` does.

One more point from the spec: A2A's `SendMessageRequest` has no skill field
at all (it holds `message`, `configuration`, `metadata`). A real A2A client
does not pick a skill on the wire. It picks an *agent* whose card advertises
the skill and sends a message, and the agent routes it internally. The
lesson's `{"skill": ...}` body is a design this demo made, not part of the
protocol.

### 3 — one open stream blocks every other request

**What changes for the client:** it opens one connection, reads it line by
line, decodes each `data:` line as JSON, and closes the connection when the
state is terminal. There is no poll interval and each change arrives once.
The new work is managing the connection. A dropped stream looks just like a
quiet one unless the client knows the terminal states. The stream also
starts at `working`, never `submitted` (see exercise 1).

**On the lesson's server, one open stream blocks everything.** `run_server`
uses `HTTPServer`, which handles one request at a time:

| during an open stream | `HTTPServer` (shipped) | `ThreadingHTTPServer` |
|---|---|---|
| GET the Agent Card | waits ~0.18s, until the task ends | ~0.001s |
| second subscriber, same task | receives `['completed']` only | `['working', 'completed']` |

A2A §3.5.2 requires every concurrent stream for a task to receive the same
events. Note also that the spec's own REST subscribe endpoint is
`POST /tasks/{id}:subscribe`, not `GET /tasks/{id}/events`.

Two smaller findings. Before the override, the shipped router answers
`/tasks/abc/events` with 404 `not found`, the *task*-not-found body, because
routing is `path.split("/tasks/", 1)[1]`. And `TaskStore` has no condition
variable, event or callback, so the endpoint can only re-read the store in
a loop. SSE here moves the polling from the client into the server, at the
cost of one server thread per subscriber.

### 4 — five of the six core operations are 404

Checked against the spec's current release, **1.0.0** (the demo's card
claims `a2a-0.3`), and against the running demo rather than its source:

1. **Discovery.** The well-known URI is `/.well-known/agent-card.json`
   (§8.2, §14.3). The demo serves `agent.json`, and the spec's path returns
   **404**. The card it does serve lacks **5 of the 8** AgentCard fields
   that `a2a.proto` marks REQUIRED: `description`, `supportedInterfaces`,
   `capabilities`, `defaultInputModes`, `defaultOutputModes`. Its skills
   are strings, where an AgentSkill requires `id`, `name`, `description`
   and `tags`.
2. **Core operations**, which §3.1 says "all A2A implementations must
   support":

   | operation | REST endpoint (§5.3) | demo |
   |---|---|---:|
   | send message | `POST /message:send` | 404 |
   | streaming message | `POST /message:stream` | 404 |
   | get task | `GET /tasks/{id}` | 200 |
   | list tasks | `GET /tasks` | 404 |
   | cancel task | `POST /tasks/{id}:cancel` | 404 |
   | subscribe | `POST /tasks/{id}:subscribe` | 404 |

   The lesson lists `canceled` as a lifecycle state, but no code in the
   module ever sets it.
3. **Errors.** §3.3.2 says servers MUST return appropriate errors. A
   malformed JSON body makes `do_POST` raise, and the client gets no
   response at all: `RemoteDisconnected`. §3.3.4 says a streaming call to
   an agent that does not declare streaming MUST get UnsupportedOperationError
   (HTTP 400). The demo returns 404.

The wire format is off too. A spec Task requires `id` and `status`, with the
state at `status.state`. The demo returns a flat `state` next to `skill`,
`payload` and `created_at`. Its artifact is `{type, data}`, while a spec
Artifact requires `artifactId` and `parts`. A spec-conformant client reading
`status.state` finds nothing to poll on.

### 5 — the card says what the agent does; listTools says how to call it

*Draws on "The MCP/A2A split".*

The obvious framing is static versus live: a card published in advance
against a list the server returns when asked. The sharper difference is
**what each one describes**.

An MCP `tools/list` entry carries an `inputSchema`, a JSON Schema for the
arguments, and optionally an `outputSchema` that the server MUST then
conform to. A client can build a valid call from the listing alone. An A2A
AgentSkill carries `id`, `name`, `description`, `tags`, `examples`, and
input/output *media types*, but no schema for what goes in. The spec never
needs one, because a client sends a *message* (text, data or file parts)
and the agent decides what to do with it. So a card describes *capability*
for a reader that understands prose, usually an LLM choosing a peer. A tool
listing describes an *interface* for a caller that builds typed calls.

That produces the tradeoff:

| | self-describing card (A2A) | capability probing (MCP) |
|---|---|---|
| when it can be read | before any connection or auth, from a URL | only inside a session, after `initialize` |
| cacheability | a static document that can be indexed and cached (§8.6) | per session; `listChanged` notifications keep it current |
| freshness | can drift from what the agent actually does | reflects the server at call time |
| precision | prose plus tags: good for choosing a peer | JSON Schema: good for building a call |
| trust | a claim, which is why signed cards exist (`AgentCardSignature`) | still a claim, but made by a server you have already connected to |

A2A partly bridges the two. Its **extended agent card** is only returned to
an authenticated client, so it is a public card plus a probe after auth.

The demo shows the cost of the card side in miniature. Its skills are bare
strings, so the one client that works is the one that already knows the
skill name and the `code` payload key (exercise 2). A card with nothing to
read reduces discovery to a hardcoded integration, which is the problem the
lesson says A2A exists to remove.
