<!-- generated:start -->
# 13-tools-and-protocols / 01-the-tool-interface

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/01-the-tool-interface/) · upstream spec
`phases/13-tools-and-protocols/01-the-tool-interface/docs/en.md`

```bash
uv run demo practice run 01-the-tool-interface --ex 1
uv run demo explain 01-the-tool-interface --ex 1
uv run pytest demos/phases/13-tools-and-protocols/01-the-tool-interface
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a fourth tool to `code/main.py` called `get_stock_price(ticker)`. Write its description a… | code | T0 | `ex01_registering_a_tool_does_not_route_to_it.py` |
| 2 | Break the schema validator. Pass a call whose `arguments` object is missing a required field,… | code | T0 | `ex02_the_extra_field_is_ignored_and_a_boolean_is_a_number.py` |
| 3 | Classify each tool in the harness as pure or consequential. Add a `consequential: true` flag… | code | T0 | `ex03_the_gate_is_live_code_that_no_shipped_tool_reaches.py` |
| 4 | Draw the four-step loop on paper with the provider-column table above filled in for your favo… | explain | T0 | prose, below |
| 5 | Read OpenAI's function-calling guide top to bottom. Identify the one field that sits in the r… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

All five are T0 and stdlib. Three ship code; exercises 4 and 5 are prose, below,
each citing the lesson section it rests on.

The three code exercises ask you to extend the harness, and all three find the
same thing one layer down: **the lesson ships the mechanism and not the wiring.**
A registered tool is never routed to, a declared schema field is never enforced,
and a written confirmation gate is never reached.

### 1 — registering a tool does not route to it

`get_stock_price` joins the registry as entry **4**, its schema accepts
`{"ticker": "AAPL"}` and rejects `{}` with `missing required field 'ticker'`, and
it executes to `{"ticker": "AAPL", "price": 231.4}`.

**ANSWER: it registers fine and is never chosen.** `fake_decide` routes **0 of
5** ticker queries to it — every one falls through to "I cannot route that query
to any registered tool."

**FINDING: `fake_decide` never reads the registry.** Stripping its docstring and
walking the AST, the names its code touches are `city, float, history, last,
len, match, msg, n, nums, re, user_msg, uuid` — `REGISTRY` is not among them. It
appears in the function exactly once, in the *docstring*, as the
`tools=[t.input_schema for t in REGISTRY]` line describing what a real provider
would receive. Routing is three hard-coded keyword branches naming `add`,
`get_time` and `get_weather` as string literals.

**FINDING: so "describe" is the step the lesson does not implement.**
`describe_registry` prints the registry to the terminal rather than passing it to
the decider. The registry reaches the human and never reaches the chooser — which
is why adding a tool changes what is printed and not what is called.

### 2 — the extra field is ignored, and a boolean is a number

| call | `validate` returns | executes to |
|---|---|---|
| `{"a": 1}` | `["missing required field 'b'"]` | blocked |
| `{"a": 1, "b": 2, "evil": "x"}` | `[]` | `{"sum": 3}` |
| `{"a": true, "b": false}` | `[]` | **`{"sum": 1}`** |

**ANSWER: missing required is rejected; the extra field is silently ignored.**
`validate` walks the *declared* properties and never enumerates the supplied
keys, so anything undeclared passes through untouched.

**ANSWER: the host should reject, because ignoring is not discarding.** The
validator ignoring a key does not remove it — the same dict is handed to
`executor(call["arguments"])`, so an undeclared key is still there for the
executor to read. Ignoring is safe only if every executor also ignores it, which
is a property of code nobody validated. Rejecting is the one choice whose safety
does not depend on the tool's implementation, and JSON Schema spells it
`additionalProperties: false` — which this validator does not implement.

**FINDING: a boolean validates as a number and the tool adds it.** The check is
`isinstance(value, (int, float))`, and in Python `isinstance(True, int)` is
`True`. So the type gate the exercise calls a safety boundary admits a value JSON
Schema excludes, and `tool_add` does arithmetic on it. That is a worse break than
either input the exercise proposes, and it passes the same "confirm the host
rejects it" test the exercise is built around.

### 3 — the gate is live code that no shipped tool reaches

**ANSWER: all three are pure, and none needs the flag.** `add` is a function of
its arguments and `get_weather` reads a literal dict; neither writes anything and
both replay identically. `consequential` stays false on all three, so **0**
registry entries need it — and the loop already prints the line the exercise asks
you to add.

**FINDING: `get_time` is not consequential, and it is not pure either.** It calls
`datetime.now`, so it reads state outside its arguments and cannot be replayed —
while "The trust split" lists `get_current_time` under "Pure. Read-only,
deterministic, no side effects". The dichotomy collapses two independent axes,
*reads* external state and *mutates* external state, into one; `get_time` is the
tool that reads without mutating. A confirmation gate keys off the mutation axis
and caching keys off the other, and the lesson has neither.

**FINDING: the gate is live code that nothing reaches.** The lesson's own four
demo queries print the `GATE` line **0** times. Flipping
`get_weather.consequential` and re-running the same query prints it **1** time —
so the branch works, and is guarded by a flag no shipped tool sets.

**FINDING: and the circuit breaker is unreachable too.** `MAX_TURNS` is **5**,
but the deepest turn any demo query reaches is **2**: `fake_decide` returns
`content` as soon as the last history entry has role `tool`, so the loop always
exits on its second pass. `LOOP TERMINATED` prints **0** times. Both of the
loop's safety features are written, and neither executes.

### 4 — the columns change and the trust boundary moves

*Cites "Where the loop lives".*

The lesson's table says the column names change and the structure does not. Filled
in for **Claude Code** as the host, and set against the MCP row it tells you to
cross-reference with Phase 13 · 06:

| | Who describes | Who decides | Who executes |
|---|---|---|---|
| Claude Code, built-in tools | Anthropic (shipped in the binary) | Claude | Claude Code, on your machine |
| Claude Code, MCP tools | the MCP server author | Claude | the MCP server |
| Claude Code, skills | whoever wrote `SKILL.md` | Claude | Claude Code, as ordinary tool calls |

Drawing it makes one thing visible that the four-step loop does not: **the four
steps live in one process in the first row and in three parties in the second.**
Describe, decide and execute are one trust domain when the host ships its own
tools, and three when it does not — the tool description is written by a third
party, the decision is made by a model that treats that description as
instructions, and the execution happens somewhere the host cannot inspect.

That is why MCP needs a security chapter and single-turn function calling does
not have one. The structure is the same; the number of parties who have to be
trusted is not, and the loop as presented has no column for "who wrote the
description the model is reading". Exercise 1's finding is the benign version of
the same gap — here the registry never reaches the model; in MCP it reaches the
model from someone else's server.

### 5 — `tool_choice` is a filter you could apply yourself

*Cites "Step two: decide".*

**The field is `tool_choice`.** It appears in every provider's request — OpenAI's
`tool_choice`, Anthropic's `tool_choice`, Gemini's
`functionCallingConfig.mode` — and in none of the four steps as presented here.
`parallel_tool_calls`, `strict` and `responseSchema` all appear in the lesson
text; `tool_choice` appears **0** times.

**What it adds** is control over the *decide* step from the caller's side. Step
two says the model chooses one of three behaviours: answer, call, or refuse.
`tool_choice` lets the request constrain that choice before the model makes it —
`"none"` forbids calls, `"auto"` is the default, `"required"` forces some call,
and `{"type": "function", "name": "get_weather"}` forces a specific one.

**Why it is convenient rather than essential** is that the host can already
produce every one of those outcomes with what the four-step loop gives it. The
describe step is the caller's, so describing zero tools is `"none"` and
describing exactly one is a named forced call. The execute step is the caller's
too, so a call the host did not want is a call the host can decline to run and
feed back as a tool error. `tool_choice` collapses that into one request field
and — this is the part worth having — it moves the constraint *before* the
generation rather than after, which saves the tokens of a call you were going to
throw away and removes a turn of latency.

What it cannot do is add a capability. It is a filter over a choice the host
already controls both ends of, which is exactly why it sits outside the
four-step loop rather than inside it: the loop is the invariant, and
`tool_choice` is an optimisation of one step of it.
